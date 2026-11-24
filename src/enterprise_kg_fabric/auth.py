from __future__ import annotations

import base64
import contextvars
import hashlib
import hmac
import json
import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from fastapi import HTTPException, Request

MIN_JWT_SECRET_BYTES = 32


class AuthMode(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    JWT = "jwt"
    CUSTOM = "custom"


@dataclass(frozen=True)
class AuthPrincipal:
    subject: str
    tenant_id: str | None = None
    acl_groups: list[str] = field(default_factory=list)
    roles: set[str] = field(default_factory=lambda: {"reader"})
    is_admin: bool = False

    @property
    def is_platform_admin(self) -> bool:
        """True ONLY for global platform admins (admin role NOT bound to a single tenant)."""
        return self.is_admin and (not self.tenant_id or self.tenant_id == "*")

    def has_role(self, role: str) -> bool:
        return self.is_admin or role in self.roles


# Service-layer AuthPrincipal propagation via contextvars (closes direct SDK / MCP stdio bypasses)
_CURRENT_PRINCIPAL: contextvars.ContextVar[AuthPrincipal | None] = contextvars.ContextVar(
    "kg_fabric_current_principal",
    default=None,
)


def get_current_principal() -> AuthPrincipal | None:
    return _CURRENT_PRINCIPAL.get()


def set_current_principal(principal: AuthPrincipal | None) -> contextvars.Token[AuthPrincipal | None]:
    return _CURRENT_PRINCIPAL.set(principal)


def reset_current_principal(token: contextvars.Token[AuthPrincipal | None]) -> None:
    _CURRENT_PRINCIPAL.reset(token)


class BoundedApiRateLimiter:
    """
    In-process bounded sliding-window rate limiter (`max_keys=10_000` FIFO/LRU eviction)
    protecting REST & MCP endpoints against runaway agents without unbounded memory growth.
    """

    def __init__(self, requests_per_minute: int = 0, max_keys: int = 10_000):
        self.requests_per_minute = max(0, int(requests_per_minute))
        self.max_keys = max(16, int(max_keys))
        self._buckets: OrderedDict[str, deque[float]] = OrderedDict()

    def check_and_record(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Returns `(allowed, remaining_quota)`."""
        if self.requests_per_minute <= 0:
            return True, -1
        ts = time.monotonic() if now is None else now
        cutoff = ts - 60.0

        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= self.max_keys:
                self._buckets.popitem(last=False)
            bucket = deque()
            self._buckets[key] = bucket
        else:
            self._buckets.move_to_end(key)

        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= self.requests_per_minute:
            return False, 0

        bucket.append(ts)
        return True, max(0, self.requests_per_minute - len(bucket))


@dataclass
class AuthConfig:
    """
    Production authentication, BYOA, rate-limiting & ABAC enforcement config for FastAPI + MCP.

    Supports:
      - `AuthMode.NONE`: Local dev / unit tests (blocked on `0.0.0.0` unless explicitly overridden).
      - `AuthMode.API_KEY`: Static or JSON-mapped API keys (`X-API-Key` or `Authorization: Bearer <key>`).
      - `AuthMode.JWT`: Strict HS256 (stdlib) or PyJWT-verified JWT tokens requiring valid `exp`,
        `tenant_id`, and a >= 32-byte secret.
      - `AuthMode.CUSTOM` / BYOA hooks:
          - `custom_request_authenticator(headers) -> AuthPrincipal | None` (API Gateway / Istio / Envoy mTLS / Kong / IAP)
          - `custom_token_verifier(token) -> AuthPrincipal | None` (OIDC/JWKS/custom IdP)
    """

    mode: AuthMode = AuthMode.NONE
    api_keys: dict[str, dict[str, Any]] = field(default_factory=dict)
    jwt_secret: str = ""
    jwks_url: str | None = None
    jwt_algorithms: list[str] = field(default_factory=lambda: ["HS256"])
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    require_exp: bool = True
    require_tenant_claim: bool = True
    api_rate_limit_rpm: int = 0
    custom_request_authenticator: Callable[[Mapping[str, str]], AuthPrincipal | None] | None = None
    custom_token_verifier: Callable[[str], AuthPrincipal | None] | None = None

    def __post_init__(self) -> None:
        if (
            self.mode == AuthMode.JWT
            and self.custom_token_verifier is None
            and not self.jwks_url
            and self.jwt_secret
            and len(self.jwt_secret.encode("utf-8")) < MIN_JWT_SECRET_BYTES
        ):
            raise ValueError(
                f"KG_FABRIC_JWT_SECRET must be at least {MIN_JWT_SECRET_BYTES} bytes "
                f"(got {len(self.jwt_secret.encode('utf-8'))} bytes)."
            )
        if self.jwks_url and self.jwt_algorithms == ["HS256"]:
            self.jwt_algorithms = ["RS256", "ES256"]

    @classmethod
    def from_env(cls) -> AuthConfig:
        raw_mode = os.getenv("KG_FABRIC_AUTH_MODE", "none").strip().lower()
        try:
            mode = AuthMode(raw_mode)
        except ValueError as exc:
            valid = ", ".join(m.value for m in AuthMode)
            raise ValueError(
                f"Invalid KG_FABRIC_AUTH_MODE='{raw_mode}'. Must be one of: {valid}"
            ) from exc

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
        jwks_url = os.getenv("KG_FABRIC_JWKS_URL", "").strip() or None
        default_algs = "RS256,ES256" if jwks_url else "HS256"
        jwt_algs_raw = os.getenv("KG_FABRIC_JWT_ALGORITHMS", default_algs).strip()
        jwt_algs = [a.strip() for a in jwt_algs_raw.split(",") if a.strip()] or (
            ["RS256", "ES256"] if jwks_url else ["HS256"]
        )
        rate_limit_rpm = int(os.getenv("KG_FABRIC_RATE_LIMIT_RPM", "0"))

        return cls(
            mode=mode,
            api_keys=api_keys,
            jwt_secret=jwt_secret,
            jwks_url=jwks_url,
            jwt_algorithms=jwt_algs,
            jwt_issuer=os.getenv("KG_FABRIC_JWT_ISSUER") or None,
            jwt_audience=os.getenv("KG_FABRIC_JWT_AUDIENCE") or None,
            api_rate_limit_rpm=rate_limit_rpm,
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


_JWKS_CLIENTS: dict[str, Any] = {}


def _get_jwks_client(jwks_url: str) -> Any:
    client = _JWKS_CLIENTS.get(jwks_url)
    if client is None:
        import jwt as pyjwt  # type: ignore

        client = pyjwt.PyJWKClient(jwks_url, cache_keys=True)
        _JWKS_CLIENTS[jwks_url] = client
    return client


def verify_jwt_token(token: str, config: AuthConfig) -> dict[str, Any]:
    # 1. Built-in OIDC / JWKS verification (Auth0, Okta, Cognito, Entra ID, Keycloak)
    if config.jwks_url:
        try:
            import jwt as pyjwt  # type: ignore

            jwk_client = _get_jwks_client(config.jwks_url)
            signing_key = jwk_client.get_signing_key_from_jwt(token)
            options = {"require": ["exp"]} if config.require_exp else {}
            kwargs: dict[str, Any] = {
                "algorithms": config.jwt_algorithms,
                "options": options,
            }
            if config.jwt_issuer:
                kwargs["issuer"] = config.jwt_issuer
            if config.jwt_audience:
                kwargs["audience"] = config.jwt_audience
            claims = pyjwt.decode(token, signing_key.key, **kwargs)
            return dict(claims)
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail="JWKS auth requires PyJWT[crypto]. Install with: pip install 'enterprise-kg-ontology-fabric[auth]'",
            ) from exc
        except Exception as exc:
            raise HTTPException(status_code=401, detail=f"Invalid OIDC/JWKS JWT token: {exc}") from exc

    if not config.jwt_secret:
        raise HTTPException(
            status_code=500,
            detail="JWT auth enabled but KG_FABRIC_JWT_SECRET (or KG_FABRIC_JWKS_URL) is not configured",
        )
    if len(config.jwt_secret.encode("utf-8")) < MIN_JWT_SECRET_BYTES:
        raise HTTPException(
            status_code=500,
            detail=f"KG_FABRIC_JWT_SECRET must be at least {MIN_JWT_SECRET_BYTES} bytes",
        )

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
    """Authenticates FastAPI & MCP callers, rate-limits requests, and enforces tenant + ABAC boundaries."""

    def __init__(self, config: AuthConfig | None = None):
        self.config = config or AuthConfig()
        self.rate_limiter = BoundedApiRateLimiter(
            requests_per_minute=self.config.api_rate_limit_rpm
        )

    def authenticate_request(self, request: Request) -> AuthPrincipal | None:
        # 1. Bring-Your-Own-Auth (BYOA) request-header hook (API Gateway / Envoy mTLS / Kong / IAP)
        if self.config.custom_request_authenticator is not None:
            headers_dict = {k.lower(): v for k, v in request.headers.items()}
            custom_principal = self.config.custom_request_authenticator(headers_dict)
            if custom_principal is not None:
                return custom_principal
            if self.config.mode == AuthMode.CUSTOM:
                raise HTTPException(status_code=401, detail="Custom request authentication failed")

        if self.config.mode == AuthMode.NONE:
            return None

        auth_header = request.headers.get("authorization", "")
        api_key_header = request.headers.get("x-api-key", "").strip()
        bearer_token = ""
        if auth_header.lower().startswith("bearer "):
            bearer_token = auth_header[7:].strip()

        # 2. Bring-Your-Own-Auth token verifier hook (OIDC/JWKS)
        if self.config.custom_token_verifier is not None and bearer_token:
            verified = self.config.custom_token_verifier(bearer_token)
            if verified is None:
                raise HTTPException(status_code=401, detail="Custom token verification failed")
            return verified

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
            if str(tenant_id) == "*" and not is_admin:
                raise HTTPException(
                    status_code=403,
                    detail="Wildcard tenant_id '*' is restricted to admin principals",
                )
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

    def check_rate_limit(self, request: Request, principal: AuthPrincipal | None) -> None:
        if self.config.api_rate_limit_rpm <= 0:
            return
        client_host = request.client.host if request.client else "unknown"
        key = (
            f"sub:{principal.tenant_id}:{principal.subject}"
            if principal is not None
            else f"ip:{client_host}"
        )
        allowed, _remaining = self.rate_limiter.check_and_record(key)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({self.config.api_rate_limit_rpm} req/min)",
            )

    def enforce_tenant(self, principal: AuthPrincipal | None, requested_tenant: str) -> str:
        if principal is None:
            return requested_tenant
        if principal.is_admin and (not principal.tenant_id or principal.tenant_id == "*"):
            return requested_tenant
        if not principal.tenant_id or principal.tenant_id == "*":
            raise HTTPException(
                status_code=403,
                detail="Non-admin principal must be bound to an explicit tenant_id",
            )
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

    def require_platform_admin(self, principal: AuthPrincipal | None) -> None:
        """Requires global platform admin (admin with unbound/wildcard tenant_id), blocking tenant-bound admins."""
        if principal is None:
            return
        if not principal.is_platform_admin:
            raise HTTPException(
                status_code=403,
                detail="Platform admin privileges (cross-tenant admin) are required for this operation.",
            )
