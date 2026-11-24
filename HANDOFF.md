# Session Handoff — `enterprise-kg-ontology-fabric`

1. **Last Updated**: 2026-11-24 14:20 EST (`Windows`)
2. **Current Goal & Status**:
   - Completed **ALL 4 External Red-Team Security Reviews (30 total findings)** AND **BOTH Feature Milestones (Milestones A & B)** on `v0.3.0` (no PyPI tag pushed yet).
   - **`ruff check src tests` — 0 errors** (`All checks passed!`).
   - **`22 / 22` pytest tests pass**, including `test_fourth_review_remediation_safe_cypher_plan_doc_acls_job_limit_restart_recovery_kb_redaction_and_safe_retract`.
3. **Recent Changes (Review #4)**:
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/rag_engine.py` & `service.py`: Replaced raw LLM-written Cypher with structured JSON `{predicate, entity_hint}` compiled strictly into hardcoded `{tenant_id: $tenant_id}` templates; blocked all `CALL` keywords in `validate_readonly_opencypher`; made Path 4 ABAC post-filter fail-closed on `(source, predicate, target)` columns; gated `synthesize_with_llm=True` and LLM intent calls to `writer`/`admin` (`0` LLM calls for plain `reader`s).
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/adapters/pgvector_store.py`, `service.py`, `api.py`, `mcp_server.py`: Enforced `caller_acl_groups` on `GET /v1/kbs/{kb_id}/documents` and redacted `table_name`/`schema_name`/`postgres_dsn_alias` for non-platform-admins on `GET /v1/kbs` & MCP `kg://kbs/{tenant}/registry`.
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/service.py` & `api.py`: Added per-tenant async build concurrency limit (`409 Conflict` on overlap), automatic reconciliation of stale `QUEUED`/`RUNNING` jobs from dead processes, and moved chunk provenance retraction AFTER extraction succeeds so transient DLQ failures preserve existing graph facts.
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/auth.py` & `adapters/graph_stores.py`: Required `jwt_issuer` + `jwt_audience` when `jwks_url` is set, offloaded JWKS key fetches to `asyncio.to_thread`, added Neo4j `TEXT INDEX` on `:Entity(canonical_name)`, and pushed Neo4j entity search ABAC filtering before `LIMIT`.
4. **Next Steps / Blockers**:
   - Start Docker Desktop and run `python examples/live_docker_v03_full_verification.py` whenever you want to re-run the live Postgres 16 + Neo4j 5 suite.
   - Push `git tag v0.3.0 && git push origin v0.3.0` for PyPI release whenever ready.
