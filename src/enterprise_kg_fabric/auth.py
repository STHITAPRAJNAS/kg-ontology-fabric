from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import HTTPException, Request


class AuthMode(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    JWT = "jwt"


@dataclass(frozen=True)
class AuthPrincipal:
    subject: str
    tenant_id: str | None = None
    acl_groups: list[str] = field(default_factory=list)
    roles: set[str] = field(default_factory=lambda: {"reader"})
    is_admin: bool = False

    def has_role(self, role: str) -> bool:
        return self.is_admin or role in self.roles


@dataclass
class AuthConfig:
    """
    Zero-dependency production authentication & ABAC enforcement config for FastAPI + MCP.

    Supports:
      - `AuthMode.NONE`: Local dev / unit tests (blocked on `0.0.0.0` unless explicitly overridden).
      - `AuthMode.API_KEY`: Static or JSON-mapped API keys (`X-API-Key` or `Authorization: Bearer <key>`).
      - `AuthMode.JWT`: Strict HS256 (stdlib) or PyJWT-verified JWT tokens requiring valid `exp` + `tenant_id`.
    """

    mode: AuthMode = AuthMode.NONE
    api_keys: dict[str, dict[str, Any]] = field(default_factory=dict)
    jwt_secret: str = ""
    jwt_algorithms: list[str] = field(default_factory=lambda: ["HS256"])
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    require_exp: bool = True
    require_tenant_claim: bool = True

    @classmethod
    def from_env(cls) -> AuthConfig:
        raw_mode = os.getenv("KG_FABRIC_AUTH_MODE", "none").strip().lower()
        try:
            mode = AuthMode(raw_mode)
        except ValueError:
            mode = AuthMode.NONE

        api_keys: dict[str, dict[str, Any]] = {}
        raw_keys = os.getenv("KG_FABRIC_API_KEYS", "").strip()
        if raw_keys:
            if raw_keys.startswith("{"):
                parsed = json.loads(raw_keys)
                if isinstance(parsed, dict):
                    for k, meta in parsed.items():
                        api_keys[str(k)] = meta if isinstance(meta, dict) else {}
            else:
                for token in raw_keys.split(","):
                    tok = token.strip()
                    if tok:
                        api_keys[tok] = {
                            "subject": "api-key-user",
                            "tenant_id": None,
                            "acl_groups": [],
                            "roles": ["admin", "writer", "reader"],
                            "is_admin": True,
                        }

        jwt_secret = os.getenv("KG_FABRIC_JWT_SECRET", "").strip()
        jwt_algs_raw = os.getenv("KG_FABRIC_JWT_ALGORITHMS", "HS256").strip()
        jwt_algs = [a.strip() for a in jwt_algs_raw.split(",") if a.strip()] or ["HS256"]

        return cls(
            mode=mode,
            api_keys=api_keys,
            jwt_secret=jwt_secret,
            jwt_algorithms=jwt_algs,
            jwt_issuer=os.getenv("KG_FABRIC_JWT_ISSUER") or None,
            jwt_audience=os.getenv("KG_FABRIC_JWT_AUDIENCE") or None,
        )


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def create_hs256_jwt(payload: dict[str, Any], secret: str) -> str:
    """Helper for issuing HS256 JWTs in tests and local CLI tooling without requiring PyJWT."""
    header = {"alg": "HS256", "typ": "JWT"}
    h_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{h_b64}.{p_b64}".encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{h_b64}.{p_b64}.{_b64url_encode(sig)}"


def verify_jwt_token(token: str, config: AuthConfig) -> dict[str, Any]:
    if not config.jwt_secret:
        raise HTTPException(status_code=500, detail="JWT auth enabled but KG_FABRIC_JWT_SECRET is not configured")

    # Prefer PyJWT if installed (supports RS256/ES256/JWKS/leeway)
    try:
        import jwt as pyjwt  # type: ignore

        options = {"require": ["exp"]} if config.require_exp else {}
        kwargs: dict[str, Any] = {
            "algorithms": config.jwt_algorithms,
            "options": options,
        }
        if config.jwt_issuer:
            kwargs["issuer"] = config.jwt_issuer
        if config.jwt_audience:
            kwargs["audience"] = config.jwt_audience
        claims = pyjwt.decode(token, config.jwt_secret, **kwargs)
        return dict(claims)
    except ImportError:
        pass
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid JWT token: {exc}") from exc

    # Built-in strict HS256 fallback using Python stdlib
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Malformed JWT token")
    try:
        header = json.loads(_b64url_decode(parts[0]).decode("utf-8"))
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
        provided_sig = _b64url_decode(parts[2])
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Corrupt JWT encoding: {exc}") from exc

    alg = str(header.get("alg", ""))
    if alg != "HS256" or "HS256" not in config.jwt_algorithms:
        raise HTTPException(status_code=401, detail=f"Unsupported JWT algorithm: {alg}")

    signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
    expected_sig = hmac.new(config.jwt_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(provided_sig, expected_sig):
        raise HTTPException(status_code=401, detail="Invalid JWT signature")

    now = time.time()
    if config.require_exp:
        exp = payload.get("exp")
        if not isinstance(exp, (int, float)):
            raise HTTPException(status_code=401, detail="JWT missing required 'exp' claim")
        if now >= float(exp):
            raise HTTPException(status_code=401, detail="JWT token has expired")
    if "nbf" in payload and isinstance(payload["nbf"], (int, float)) and now < float(payload["nbf"]):
        raise HTTPException(status_code=401, detail="JWT token not yet valid")
    if config.jwt_issuer and payload.get("iss") != config.jwt_issuer:
        raise HTTPException(status_code=401, detail="Invalid JWT issuer")
    if config.jwt_audience:
        aud = payload.get("aud")
        if isinstance(aud, list):
            if config.jwt_audience not in aud:
                raise HTTPException(status_code=401, detail="Invalid JWT audience")
        elif aud != config.jwt_audience:
            raise HTTPException(status_code=401, detail="Invalid JWT audience")

    return dict(payload)


class AuthEnforcer:
    """Authenticates FastAPI & MCP callers and enforces tenant + ABAC `acl_groups` boundaries."""

    def __init__(self, config: AuthConfig | None = None):
        self.config = config or AuthConfig()

    def authenticate_request(self, request: Request) -> AuthPrincipal | None:
        if self.config.mode == AuthMode.NONE:
            return None

        auth_header = request.headers.get("authorization", "")
        api_key_header = request.headers.get("x-api-key", "").strip()
        bearer_token = ""
        if auth_header.lower().startswith("bearer "):
            bearer_token = auth_header[7:].strip()

        if self.config.mode == AuthMode.API_KEY:
            candidate = api_key_header or bearer_token
            if not candidate:
                raise HTTPException(status_code=401, detail="Missing API key (X-API-Key or Bearer token)")
            matched_meta: dict[str, Any] | None = None
            for configured_key, meta in self.config.api_keys.items():
                if hmac.compare_digest(candidate, configured_key):
                    matched_meta = meta
                    break
            if matched_meta is None:
                raise HTTPException(status_code=401, detail="Invalid API key")

            roles = set(str(r) for r in matched_meta.get("roles", ["reader"]))
            is_admin = bool(matched_meta.get("is_admin", False)) or ("admin" in roles)
            acl_groups = [str(g) for g in matched_meta.get("acl_groups", [])]
            return AuthPrincipal(
                subject=str(matched_meta.get("subject", "api-key-principal")),
                tenant_id=matched_meta.get("tenant_id"),
                acl_groups=acl_groups,
                roles=roles,
                is_admin=is_admin,
            )

        if self.config.mode == AuthMode.JWT:
            if not bearer_token:
                raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <jwt> header")
            claims = verify_jwt_token(bearer_token, self.config)
            tenant_id = claims.get("tenant_id")
            if self.config.require_tenant_claim and not tenant_id:
                raise HTTPException(status_code=401, detail="JWT missing required 'tenant_id' claim")
            raw_roles = claims.get("roles", ["reader"])
            roles = set(str(r) for r in (raw_roles if isinstance(raw_roles, list) else [raw_roles]))
            is_admin = bool(claims.get("is_admin", False)) or ("admin" in roles)
            raw_acls = claims.get("acl_groups", claims.get("groups", []))
            acl_groups = [str(g) for g in (raw_acls if isinstance(raw_acls, list) else [])]
            return AuthPrincipal(
                subject=str(claims.get("sub", "jwt-principal")),
                tenant_id=str(tenant_id) if tenant_id else None,
                acl_groups=acl_groups,
                roles=roles,
                is_admin=is_admin,
            )

        return None

    def enforce_tenant(self, principal: AuthPrincipal | None, requested_tenant: str) -> str:
        if principal is None or principal.is_admin or not principal.tenant_id:
            return requested_tenant
        if requested_tenant and requested_tenant != principal.tenant_id:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Tenant isolation violation: principal bound to '{principal.tenant_id}' "
                    f"cannot access tenant '{requested_tenant}'"
                ),
            )
        return principal.tenant_id

    def enforce_acl_groups(
        self,
        principal: AuthPrincipal | None,
        requested_acls: list[str] | None,
    ) -> list[str] | None:
        """
        Prevents ABAC privilege escalation:
          - When unauthenticated (`AuthMode.NONE`) or admin with no bound ACL restriction, preserves requested ACLs.
          - When authenticated with bound `principal.acl_groups`, never allows caller to claim groups
            outside `principal.acl_groups`. If caller omits `caller_acl_groups`, injects `principal.acl_groups`.
        """
        if principal is None:
            return requested_acls
        if principal.is_admin and not principal.acl_groups:
            return requested_acls
        allowed = set(principal.acl_groups)
        if requested_acls is None:
            return list(principal.acl_groups)
        return [g for g in requested_acls if g in allowed]

    def require_role(self, principal: AuthPrincipal | None, role: str) -> None:
        if principal is None:
            return
        if not principal.has_role(role):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions: role '{role}' required",
            )
