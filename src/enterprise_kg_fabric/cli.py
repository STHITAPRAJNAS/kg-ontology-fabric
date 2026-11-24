from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Sequence

from .api import API_VERSION
from .auth import AuthConfig, AuthMode


def _is_non_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    return normalized not in {"127.0.0.1", "localhost", "::1"}


def validate_serve_security(
    host: str,
    auth_config: AuthConfig,
    allow_unauthenticated: bool = False,
) -> None:
    """
    Prevents accidentally exposing an unauthenticated KG Fabric API/MCP server on a
    public/network interface (`0.0.0.0`, `::`, etc.) unless explicitly acknowledged.
    Note: Secrets (`KG_FABRIC_JWT_SECRET`, `KG_FABRIC_API_KEYS`) are configured strictly
    via environment variables, never CLI flags, to prevent exposure in `ps` / `top`.
    """
    env_override = os.getenv("KG_FABRIC_ALLOW_UNAUTHENTICATED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if (
        _is_non_loopback_host(host)
        and auth_config.mode == AuthMode.NONE
        and not (allow_unauthenticated or env_override)
    ):
        raise SystemExit(
            "REFUSING TO START: Binding to non-loopback host "
            f"'{host}' with KG_FABRIC_AUTH_MODE=none is unsafe.\n"
            "Configure KG_FABRIC_AUTH_MODE=api_key (with KG_FABRIC_API_KEYS) or "
            "KG_FABRIC_AUTH_MODE=jwt (with KG_FABRIC_JWT_SECRET >= 32 bytes), OR pass "
            "--allow-unauthenticated / KG_FABRIC_ALLOW_UNAUTHENTICATED=true for isolated networks."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kg-fabric",
        description="Enterprise Multi-KB Automated Ontology & Knowledge Graph Fabric CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. `kg-fabric serve`
    p_serve = sub.add_parser(
        "serve",
        help="Run the co-hosted FastAPI REST + MCP HTTP server",
    )
    p_serve.add_argument("--host", default=os.getenv("KG_FABRIC_HOST", "127.0.0.1"))
    p_serve.add_argument("--port", type=int, default=int(os.getenv("KG_FABRIC_PORT", "8080")))
    p_serve.add_argument("--workers", type=int, default=int(os.getenv("UVICORN_WORKERS", "1")))
    p_serve.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="Explicitly allow binding non-loopback interfaces with KG_FABRIC_AUTH_MODE=none",
    )
    p_serve.add_argument(
        "--enable-sync-worker",
        action="store_true",
        help="Enable built-in multi-replica-safe background incremental CDC sync worker",
    )
    p_serve.add_argument(
        "--sync-interval-sec",
        type=float,
        default=float(os.getenv("KG_FABRIC_SYNC_INTERVAL_SEC", "300")),
        help="Background incremental sync interval in seconds (default: 300)",
    )

    # 2. `kg-fabric mcp`
    p_mcp = sub.add_parser(
        "mcp",
        help="Run the standalone FastMCP server (stdio or streamable-http)",
    )
    p_mcp.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="MCP transport mode (default: stdio for Claude Desktop / Cursor / Windsurf / ADK)",
    )

    # 3. `kg-fabric init-db` & `kg-fabric migrate-db`
    for cmd_name, help_txt in (
        ("init-db", "Initialize PostgreSQL pgvector schema (registry, watermarks, TBox, jobs, DLQ, RLS)"),
        ("migrate-db", "Idempotently upgrade PostgreSQL schema, acl_groups, RLS policies, and optional NOBYPASSRLS role"),
    ):
        p_db = sub.add_parser(cmd_name, help=help_txt)
        p_db.add_argument(
            "--app-role",
            default=os.getenv("KG_FABRIC_APP_ROLE", ""),
            help="Optional least-privilege PostgreSQL role name (NOBYPASSRLS). Password read from KG_FABRIC_APP_ROLE_PASSWORD env var.",
        )

    # 4. `kg-fabric version`
    sub.add_parser("version", help="Print installed kg-fabric version")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(f"enterprise-kg-ontology-fabric {API_VERSION}")
        return 0

    if args.command == "serve":
        auth_cfg = AuthConfig.from_env()
        validate_serve_security(
            host=args.host,
            auth_config=auth_cfg,
            allow_unauthenticated=args.allow_unauthenticated,
        )
        if args.enable_sync_worker:
            os.environ["KG_FABRIC_ENABLE_SYNC_WORKER"] = "true"
            os.environ["KG_FABRIC_SYNC_INTERVAL_SEC"] = str(args.sync_interval_sec)

        import uvicorn

        uvicorn.run(
            "enterprise_kg_fabric.api:create_kg_fabric_app_from_env",
            factory=True,
            host=args.host,
            port=args.port,
            workers=args.workers,
        )
        return 0

    if args.command == "mcp":
        from .mcp_server import create_fastmcp_server
        from .service import EnterpriseKGOntologyService

        svc = EnterpriseKGOntologyService.from_env()
        try:
            fastmcp = create_fastmcp_server(svc)
        except ImportError:
            print(
                "ERROR: The optional `mcp` package is required for `kg-fabric mcp`.\n"
                'Install it with: pip install "enterprise-kg-ontology-fabric[mcp]"',
                file=sys.stderr,
            )
            return 1
        fastmcp.run(transport=args.transport)
        return 0

    if args.command in {"init-db", "migrate-db"}:
        from .adapters.pgvector_store import PostgresPgVectorStoreAdapter
        from .service import EnterpriseKGOntologyService

        svc = EnterpriseKGOntologyService.from_env()
        if not isinstance(svc.vector_store, PostgresPgVectorStoreAdapter):
            print(
                f"ERROR: Set PGVECTOR_DSN or PGHOST before running `kg-fabric {args.command}`.",
                file=sys.stderr,
            )
            return 1

        app_role = args.app_role.strip() or None
        app_role_pw = os.getenv("KG_FABRIC_APP_ROLE_PASSWORD", "")

        async def _migrate() -> None:
            await svc.vector_store.migrate_schema_and_rls(  # type: ignore[attr-defined]
                app_role_name=app_role,
                app_role_password=app_role_pw,
            )
            await svc.close()

        asyncio.run(_migrate())
        msg = "PostgreSQL pgvector schema & RLS policies migrated successfully."
        if app_role:
            msg += f" Provisioned least-privilege NOBYPASSRLS role '{app_role}'."
        print(msg)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
