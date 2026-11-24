# Session Handoff — `enterprise-kg-ontology-fabric`

1. **Last Updated**: 2026-11-24 11:26 EST (`Windows`)
2. **Current Goal & Status**:
   - Completed full **`v0.3.0`** PyPI-ready + cloneable-service hardening with full security/durability parity ported from `adk-memory-platform`, **automated 4-dimension Ontology & KG Quality Scorecard (`KGQualityAuditor`)**, **Vector-vs-GraphRAG Lift in `EvalSuiteReport`**, and **live verification on Docker PostgreSQL 16 (`NOBYPASSRLS` RLS) + Neo4j 5 + live Uvicorn HTTP server** (no PyPI tag/push yet per user request).
   - **`16 / 16` pytest tests pass** (`6.10s`).
   - **Live Docker Verification (`examples/live_docker_v03_full_verification.py`) — ALL PASSED**:
     - Fail-closed PostgreSQL 16 Kernel RLS under least-privilege `NOBYPASSRLS` role `kg_app_user` (`fail_closed=0 rows`, `beta_leak=0`)
     - Live PostgreSQL `pg_try_advisory_lock` multi-replica sync coordination
     - Live Uvicorn HTTP server (`32+` byte JWT auth, `401`/`403` anti-smuggling, ABAC zero-leakage across live `pgvector`, live Neo4j 5, Path 4 OpenCypher & live `/mcp`, `14` MCP tools, `/v1/quality/report`, `/metrics`, `/ui`)
     - Live Quality Scorecard on real Postgres + Neo4j graph: **`96.18 / 100 (Grade A+)`** (`ontology_tbox=25.0`, `extraction_fidelity=24.69`, `entity_resolution=20.0`, `graph_topology=26.49`, accurately flagging the 1 simulated DLQ chunk).
3. **Recent Changes**:
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/auth.py`: Added 32-byte minimum JWT secret guard, `contextvars` (`get_current_principal`/`set_current_principal`/`reset_current_principal`), BYOA hooks (`custom_request_authenticator`/`custom_token_verifier`), and `BoundedApiRateLimiter` (`429`).
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/adapters/pgvector_store.py`: Added fail-closed RLS `CREATE POLICY` rules across all 6 tables, `get_rls_app_role_ddl()`, `_scoped_conn(tenant_id)` (`SET LOCAL app.current_tenant_id`), `acl_groups` JSONB column + persistence/restoration/SQL filtering, `migrate_schema_and_rls()`, and `try_acquire_advisory_lock`/`release_advisory_lock`.
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/adapters/graph_stores.py`: Fixed Neo4j 5 `acl_groups` & `qualifiers_json` persistence/retrieval across `upsert_entities_and_bitemporal_edges`, `get_all_entities`, `get_all_edges`, and `expand_k_hop_subgraph`; added composite indexes and batched `UNWIND` for non-functional edges.
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/service.py`, `api.py`, `mcp_server.py`, `cli.py`: Added service-layer & MCP resource `contextvars` auth enforcement, query-vs-body `tenant_id` smuggling rejection (`403`), `check_answerable` (REST + 13th MCP tool), multi-replica-safe background sync worker (`pg_try_advisory_lock`), `GET /metrics`, `GET /v1/metrics`, `GET /ui`, `kg-fabric migrate-db`, and fixed `Neo4jOpenCypherGraphAdapter(username=...)` in `from_env()`.
   - `enterprise-kg-ontology-fabric/examples/live_docker_v03_full_verification.py` & `tests/test_v03_pypi_and_security_hardening.py`: Full live-Docker & unit adversarial suites.
4. **Next Steps / Blockers**:
   - No PyPI release pushed yet (`v0.3.0` tag intentionally NOT pushed). Whenever ready to publish to PyPI, link Trusted Publishing on PyPI for `STHITAPRAJNAS/kg-ontology-fabric` and push `git tag v0.3.0 && git push origin v0.3.0`.
   - Optional next step if desired: live LLM extraction test (`KG_FABRIC_USE_LIVE_LLM=true` with a live Gemini/Bedrock/OpenAI API key) on unstructured prose.
