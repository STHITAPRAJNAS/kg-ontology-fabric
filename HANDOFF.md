# Session Handoff — `enterprise-kg-ontology-fabric`

1. **Last Updated**: 2026-11-23 15:48 EST (`Windows`)
2. **Current Goal & Status**:
   - Shipped **`v0.3.0`** of `enterprise-kg-ontology-fabric` (`https://github.com/STHITAPRAJNAS/kg-ontology-fabric`) in **Dual-Mode Architecture**: fully **PyPI-ready** (SDK + `kg-fabric` CLI + pure-ASGI JWT/API-key auth + pluggable `DomainRulePack` + durable Postgres Jobs/DLQ + GitHub Actions CI & PyPI Trusted Publishing) while remaining **clone-and-host ready** (`docker compose --profile app up -d --build`, `.env.example`, `Dockerfile`, and Amazon EKS manifests).
   - All **`14 / 14` pytest tests pass** across `tests/test_enterprise_kg_fabric_e2e.py`, `tests/test_v02_enterprise_sota_capabilities.py`, and `tests/test_v03_pypi_and_security_hardening.py` (including full pure-ASGI JWT/API-key auth, expired-token rejection, cross-tenant smuggling `403`, reader write-guard `403`, ABAC anti-spoofing across REST/MCP/OpenCypher Path 4, generic & custom Healthcare `DomainRulePack`, CLI `0.0.0.0` bind guard, `from_env()`, and durable Postgres Jobs/DLQ cold-start hydration).
3. **Recent Changes (`v0.3.0`)**:
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/auth.py`: Added `AuthMode` (`none` | `api_key` | `jwt`), `AuthConfig.from_env()`, `AuthPrincipal`, `AuthEnforcer`, `create_hs256_jwt`, and strict `exp` + `tenant_id` + `acl_groups` verification (PyJWT when installed, pure-stdlib HS256 fallback).
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/domain_packs.py`: Extracted pluggable `DomainRulePack` with built-in `CLOUD_SRE_DOMAIN_PACK` and domain-neutral `GENERIC_DOMAIN_PACK`; wired into `AutomatedOntologyInducer` and `EnterpriseKGOntologyService`.
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/cli.py`: Added `kg-fabric` CLI (`serve`, `mcp`, `init-db`, `version`) with strict non-loopback (`0.0.0.0`) unauthenticated bind guard.
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/api.py`: Added pure-ASGI `_AuthASGIMiddleware`, FastAPI `lifespan` graceful `svc.close()`, `create_kg_fabric_app_from_env()`, and tenant/role/ABAC enforcement across all REST & MCP endpoints.
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/rag_engine.py`: Added Path 4 OpenCypher ABAC post-filtering so restricted entities/edges cannot leak via `opencypher_results`.
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/adapters/pgvector_store.py` & `service.py`: Added durable PostgreSQL `kg_sync_jobs` & `kg_dlq_chunks` tables + `EnterpriseKGOntologyService.from_env()` (supporting `PGVECTOR_DSN` and standard `PGHOST`/`PGPORT`/`PGDATABASE`/`PGUSER`/`PGPASSWORD`) + `close()`.
   - `enterprise-kg-ontology-fabric/pyproject.toml`, `Dockerfile`, `docker-compose.yml`, `.env.example`, `CHANGELOG.md`, `deploy/eks/deployment-api.yaml`, `.github/workflows/ci.yml`, `.github/workflows/release.yml`: Bumped to `0.3.0`, added `[project.scripts]`, full-stack `--profile app` compose service, and CI/PyPI OIDC release workflows.
   - `enterprise-kg-ontology-fabric/tests/test_v03_pypi_and_security_hardening.py`: Added 4 comprehensive `v0.3.0` security, domain-pack, CLI, and durability tests.
4. **Next Steps / Blockers**:
   - To publish `v0.3.0` to PyPI once PyPI Trusted Publishing is linked to `STHITAPRAJNAS/kg-ontology-fabric`, simply tag & push `git tag v0.3.0 && git push origin v0.3.0` (triggers `.github/workflows/release.yml`).
   - To run all 14 offline tests on any OS:
     `PYTHONPATH="enterprise-kg-ontology-fabric/src:enterprise-kg-ontology-fabric" pytest enterprise-kg-ontology-fabric/tests/ -v`
   - To clone & run the full containerized stack:
     `docker compose -f enterprise-kg-ontology-fabric/docker-compose.yml --profile app up -d --build`
