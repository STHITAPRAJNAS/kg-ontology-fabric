# Session Handoff — `enterprise-kg-ontology-fabric`

1. **Last Updated**: 2026-11-24 13:16 EST (`Windows`)
2. **Current Goal & Status**:
   - Remediated **ALL findings across External Red-Team Reviews #1 (12 findings), #2 (7 findings), and #3 (4 findings)** on `v0.3.0` (no PyPI tag pushed yet).
   - **`ruff check src tests` — 0 errors** (`All checks passed!`).
   - **`19 / 19` pytest tests pass (`6.13s`)**, including all 3 adversarial security regression suites:
     - `test_review_remediation_critical_security_neptune_promote_delete_and_auth_fail_closed`
     - `test_second_review_remediation_legacy_exfil_platform_admin_entity_pk_delete_dlq_and_metrics`
     - `test_third_review_remediation_legacy_repoint_hmac_ingest_acl_narrowing_and_llm_judge_rbac`
   - Note: Live Docker verification (`examples/live_docker_v03_full_verification.py`) passed on Review #2 (`96.71 / 100 A+`), and was extended with Review #3 live checks (legacy re-point `403`, `use_llm_judge=true` `403`, Neo4j 5 entity ACL narrowing); Docker Desktop was stopped on the host during the Review #3 check so re-run `examples/live_docker_v03_full_verification.py` once Docker Desktop is started.
3. **Recent Changes (Review #3)**:
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/service.py` & `adapters/pgvector_store.py`: Blocked tenant writers from re-pointing an existing `EXISTING_PGVECTOR_KB` to another table via `source_type=MANAGED_RAW_DOC_KB` (`403`, requires `platform_admin`), and synced `source_type`/`schema_name`/`column_mapping`/`embedding_model_id` on `ON CONFLICT (kb_id) DO UPDATE`.
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/llama_index_bridge.py` & `service.py`: Wired `service.pii_redactor` into `LlamaIndexHierarchicalNodeParser` so `ingest_raw_documents` honors `PIIRedactionMode.HMAC_PSEUDONYMIZE` (and updates `_pii_redaction_audit_counts`) identically to `hydrate_existing_kb_chunks`.
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/adapters/pgvector_store.py`, `adapters/graph_stores.py`, & `service.py`: Added `fetch_tenant_chunk_acl_map` and narrowed surviving entity/edge `acl_groups` inside `delete_chunks_provenance` across In-Memory, Neo4j 5, and Neptune adapters.
   - `enterprise-kg-ontology-fabric/src/enterprise_kg_fabric/service.py` & `api.py`: Gated `use_llm_judge=true` on `GET /v1/quality/report` and MCP `audit_kg_quality` to `writer`/`admin` (`reader` gets `403`).
4. **Next Steps / Blockers**:
   - No PyPI release pushed yet (`v0.3.0` tag intentionally NOT pushed). Whenever ready to publish to PyPI, link Trusted Publishing on PyPI for `STHITAPRAJNAS/kg-ontology-fabric` and push `git tag v0.3.0 && git push origin v0.3.0`.
   - When Docker Desktop is running, optionally re-run `python examples/live_docker_v03_full_verification.py` to exercise the 3 newly added Review #3 assertions against live Postgres 16 + Neo4j 5.
