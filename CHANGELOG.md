# Changelog

All notable changes to `enterprise-kg-ontology-fabric` are documented in this file.

## [0.3.0] - 2026-11-23
### Added — PyPI Release Readiness, `adk-memory-platform` Hardening Parity & Live-Docker RLS Verification
- **CLI Entrypoint (`kg-fabric`)**: Added `[project.scripts]` CLI (`kg-fabric serve`, `kg-fabric mcp`, `kg-fabric init-db`, `kg-fabric migrate-db --app-role kg_app_user`, `kg-fabric version`) with a strict non-loopback (`0.0.0.0`) unauthenticated bind guard and env-only secrets (no CLI secret flags).
- **Hardened Authentication, BYOA, Rate Limiting & Service-Layer `contextvars` Enforcement (`auth.py`, `api.py`, `service.py`, `mcp_server.py`)**:
  - Supports `KG_FABRIC_AUTH_MODE=api_key`, `jwt` (strict `exp` + `tenant_id` + **minimum 32-byte secret**), and `custom` Bring-Your-Own-Auth (BYOA) hooks (`custom_request_authenticator(headers)` for API Gateway/Envoy/Kong/IAP and `custom_token_verifier(token)`).
  - Enforces bounded API rate limiting (`BoundedApiRateLimiter`, `429`), query-vs-body `tenant_id` smuggling rejection (`403`), and propagates `AuthPrincipal` via `contextvars` (`get_current_principal` / `set_current_principal`) into `EnterpriseKGOntologyService` and `KGFabricMcpServer` (closing direct SDK, MCP `stdio`, and MCP resource-read bypasses).
- **Fail-Closed PostgreSQL 16 Row-Level Security (`NOBYPASSRLS`) & Live Adapter Fixes (`adapters/pgvector_store.py`, `adapters/graph_stores.py`)**:
  - Added fail-closed `CREATE POLICY` RLS rules across all 6 multi-tenant PostgreSQL tables + transaction-local `SET LOCAL app.current_tenant_id` + `get_rls_app_role_ddl()` (`NOBYPASSRLS` role provisioning).
  - Persisted and restored `acl_groups` across PostgreSQL `unified_rag_chunks` (JSONB column + GIN index + SQL filter) and Neo4j 5 nodes/edges, persisted N-ary `qualifiers_json` on live Neo4j 5 edges, added Neo4j composite indexes (`:Entity(tenant_id, entity_id)`), and batched non-functional edge upserts via `UNWIND`.
- **Pre-Flight Answerability Check, Multi-Replica Sync Worker, Metrics & Governance Console**:
  - Added `POST /v1/rag/check-answerable` & 13th MCP tool `check_answerable` (`ANSWERABLE` | `PARTIAL` | `UNANSWERABLE`).
  - Added opt-in multi-replica-safe background incremental CDC sync worker (`start_sync_worker` / `run_scheduled_sync_cycle`) coordinated via `asyncio.Lock` + PostgreSQL `pg_try_advisory_lock`.
  - Added `GET /metrics` (Prometheus exposition), `GET /v1/metrics` (JSON), and lightweight Governance Console SPA at `GET /ui`.
- **Pluggable Domain Packs (`domain_packs.py`)**: Extracted fallback ontology rules into `DomainRulePack` with built-in `CLOUD_SRE_DOMAIN_PACK` and domain-neutral `GENERIC_DOMAIN_PACK` (`KG_FABRIC_DOMAIN_PACK=generic|cloud_sre`).
- **Automated Ontology & KG Quality Scorecard (`quality_auditor.py`, `GET /v1/quality/report`, MCP `audit_kg_quality`) + Vector-vs-GraphRAG Eval Lift (`eval_harness.py`)**:
  - Evaluates 4 quality dimensions (`0–100` grade): **Ontology/TBox** (W3C SHACL, schema utilization, near-duplicate classes, domain/range compliance), **Extraction Fidelity** (zero-extraction chunk rate, DLQ rate, edge provenance, deterministic or optional Gemini LLM-as-a-Judge triple faithfulness), **Entity Resolution** (suspected unmerged duplicates + over-merged alias/sibling conflicts), and **Graph Topology** (orphan entity rate, connected components, GCC ratio, cross-KB linkage coverage, hub skew, Newman-Girvan community modularity).
  - Surfaces quality score in Prometheus (`kg_fabric_quality_score{tenant_id=...}`), Governance Console (`GET /ui`), and 14th MCP tool (`audit_kg_quality`).
  - Enhanced `POST /v1/eval/run` (`EvalSuiteReport`) with automated `vector_only_baseline_recall` and `graphrag_recall_lift`.
- **Live-Docker Adversarial & SOTA Verification (`examples/live_docker_v03_full_verification.py`)**: Verified live on Docker PostgreSQL 16 (`NOBYPASSRLS` `kg_app_user`) + Neo4j 5 + live Uvicorn HTTP server.
### Security & Correctness Hardening (External Red-Team Reviews #1 & #2)
- **Fail-Closed Legacy Table Access (`validate_legacy_table_access`, `KG_FABRIC_ALLOWED_LEGACY_TABLES`)**: Unconditionally denies PostgreSQL system schemas (`pg_catalog`, `information_schema`, `pg_toast`, `pg_temp*`), internal KG Fabric tables (`unified_rag_chunks`, etc.), and invalid SQL identifiers; disables legacy physical table hydration by default unless explicitly allowlisted; requires global `platform_admin` (`roles=["admin"]`, `tenant_id="*"` / unscoped) to register `EXISTING_PGVECTOR_KB` in authenticated mode; executes inside `_scoped_conn(tenant_id)`; and skips unchanged rows incrementally via `content_sha256`.
- **Platform Admin vs. Tenant-Bound Admin Separation (`AuthPrincipal.is_platform_admin`, `require_platform_admin`)**: Raw OpenCypher (`POST /v1/search/cypher`, MCP `execute_readonly_opencypher`) now requires global `platform_admin` (blocking tenant-bound admins with `403`); `GET /v1/jobs/dlq`, `GET /v1/metrics`, and `GET /metrics` strictly isolate non-platform-admins to their own `tenant_id`.
- **Cross-Tenant Entity Embedding Isolation**: Scoped generated `entity_id`s to `ent::<tenant_id>::<class>::<slug>` and upgraded `canonical_entity_embeddings` to composite `PRIMARY KEY (tenant_id, entity_id)` with idempotent migration so same-named entities across tenants never collide.
- **Complete Document & KB Deletion (`delete_chunks_provenance`, `delete_canonical_entity_embeddings`)**: `DELETE /v1/documents/{kb}/{doc}` and `DELETE /v1/kbs/{kb}` now prune graph edge/entity provenance across Neo4j, Neptune, and In-Memory backends and delete orphaned entities, edges, and `canonical_entity_embeddings`.
- **Additional Hardening**: Fixed plural MCP write RBAC + nested tenant validation, cross-tenant KB takeover & chunk overwrite guards (even under DB superuser), real Neptune `/openCypher` & OpenAI/Bedrock LLM extractor implementations, persistent idempotent `promote_quarantined_ontology_concepts`, fail-closed `KG_FABRIC_AUTH_MODE`, and `ruff` linting CI gate.

## [0.2.0] - 2026-11-23
### Added — 2024–2026 SOTA GraphRAG & Enterprise Production Capabilities
- Structured-output LLM extraction (`google-genai`, Bedrock, OpenAI) with `AsyncCircuitBreaker` and N-ary reified claim qualifiers (`qualifiers`).
- HippoRAG 2 Personalized PageRank (PPR) with inverse-degree hub damping (`POST /v1/search/ppr`, MCP `search_ppr_subgraph`).
- Microsoft GraphRAG Leiden/Louvain hierarchical community detection and global sensemaking (`POST /v1/search/global`, MCP `query_global_communities`).
- Adaptive-RAG query router + pluggable Cross-Encoder reranker.
- End-to-end ABAC security trimming (`caller_acl_groups`) across `pgvector` and Neo4j/Neptune.
- IBM Docling multimodal ingestion + Markdown/CSV table-to-triples parser.
- Persistent PostgreSQL TBox store (`enterprise_tbox_registry`), OpenTelemetry tracing, and Continuous GraphRAG Evaluation Harness (`POST /v1/eval/run`).

## [0.1.0] - 2026-11-23
### Added — Initial Enterprise Multi-KB Ontology & KG Fabric
- Multi-KB `pgvector` federation (`384d`/`768d`/`1024d`/`1536d`), Sub-Linear Vector Centroid Sampling, W3C OWL 2 / SHACL compiler (`rdflib`/`pyshacl`), `SiblingCollisionGuard`, bi-temporal functional edges, Microsoft Presidio PII redaction, Neo4j 5 & Amazon Neptune `openCypher` adapters, co-hosted FastAPI + MCP server, and Amazon EKS manifests.
