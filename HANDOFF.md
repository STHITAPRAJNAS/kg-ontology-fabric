# Session Handoff — `enterprise-kg-ontology-fabric`

1. **Last Updated**: 2026-11-24 12:27 EST (`Windows`)
2. **Current Goal & Status**:
   - Remediated **ALL 12 findings from Red-Team Review #1 AND ALL 7 findings from Red-Team Review #2** on `v0.3.0` (no PyPI tag pushed yet).
   - **`ruff check src tests` — 0 errors** (`All checks passed!`).
   - **`18 / 18` pytest tests pass (`6.32s`)**, including both adversarial security regression suites (`test_review_remediation_critical_security_neptune_promote_delete_and_auth_fail_closed` and `test_second_review_remediation_legacy_exfil_platform_admin_entity_pk_delete_dlq_and_metrics`).
   - **Live Docker Verification (`examples/live_docker_v03_full_verification.py` & `examples/live_docker_messy_enterprise_test.py`) — ALL PASSED** & live Postgres/Neo4j demo stores cleanly repaired/reset:
     - Fail-closed `EXISTING_PGVECTOR_KB` allowlist (`KG_FABRIC_ALLOWED_LEGACY_TABLES`) + unconditional denylist of `pg_catalog`/`information_schema`/`unified_rag_chunks` + `platform_admin` requirement (`403` verified live) + incremental SHA-256 skip (`delta=0` on 2nd build).
     - Tenant-bound admin blocked (`403`) from raw Cypher (REST + MCP) and cross-tenant `/v1/jobs/dlq`; global `platform_admin` (`tenant_id="*"`, `roles=["admin"]`) enforced.
     - Same-named entity (`payment-router-01`) across `tenant_alpha` & `tenant_beta` isolated via composite `PRIMARY KEY (tenant_id, entity_id)` & tenant-scoped `entity_id` (`ent::<tenant_id>::<class>::<slug>`).
     - Complete `DELETE /v1/documents/{kb}/{doc}` & `DELETE /v1/kbs/{kb}` prunes Postgres chunks, Neo4j 5 graph entities/edges, and `canonical_entity_embeddings` (verified `/v1/rag/query` stops returning deleted facts).
     - `/v1/metrics` and `/metrics` scoped to caller's `tenant_id` (zero cross-tenant ID/score leaks).
     - PostgreSQL 16 Kernel RLS under `NOBYPASSRLS` `kg_app_user` (`fail_closed=0`, `beta_leak=0`), `14` MCP tools, `GET /v1/quality/report` **`96.71 / 100 A+`**.
3. **Recent Changes**:
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/auth.py`: Added `AuthPrincipal.is_platform_admin` and `AuthEnforcer.require_platform_admin`.
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/entity_resolution.py`: Tenant-scoped generated `entity_id` (`ent::<tenant_id>::<class>::<slug>`).
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/adapters/pgvector_store.py`: Added `validate_legacy_table_access`, fail-closed `allowed_legacy_tables` / `KG_FABRIC_ALLOWED_LEGACY_TABLES`, `_scoped_conn` legacy hydration, composite `PRIMARY KEY (tenant_id, entity_id)` DDL + idempotent migration, `delete_canonical_entity_embeddings`, and `unregister_kb_chunks`/`delete_document_chunks` returning deleted `chunk_ids`.
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/adapters/graph_stores.py`: Implemented `delete_chunks_provenance` across In-Memory, Neo4j 5, and Neptune adapters.
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/service.py`, `api.py`, `mcp_server.py`: Wired `require_platform_admin` on raw Cypher & legacy KB registration, incremental legacy SHA-256 skip, complete graph+vector deletion, tenant-bound admin DLQ guard, and tenant-scoped `/v1/metrics` & `/metrics`.
   - `enterprise-kg-ontology-fabric/tests/test_v03_pypi_and_security_hardening.py` & `examples/live_docker_v03_full_verification.py`: Added comprehensive unit and live-Docker adversarial proofs for all 7 second-review findings.
4. **Next Steps / Blockers**:
   - No PyPI release pushed yet (`v0.3.0` tag intentionally NOT pushed). Whenever ready to publish to PyPI, link Trusted Publishing on PyPI for `STHITAPRAJNAS/kg-ontology-fabric` and push `git tag v0.3.0 && git push origin v0.3.0`.
   - Optional next step if desired: live LLM extraction test (`KG_FABRIC_USE_LIVE_LLM=true` with a live Gemini/Bedrock/OpenAI API key) on unstructured prose.
