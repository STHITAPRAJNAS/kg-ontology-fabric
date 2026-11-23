# Changelog

All notable changes to `enterprise-kg-ontology-fabric` are documented in this file.

## [0.3.0] - 2026-11-23
### Added — PyPI Release Readiness & Dual-Mode Self-Hosting
- **CLI Entrypoint (`kg-fabric`)**: Added `[project.scripts]` CLI (`kg-fabric serve`, `kg-fabric mcp`, `kg-fabric init-db`, `kg-fabric version`) with a strict non-loopback (`0.0.0.0`) unauthenticated bind guard (`--allow-unauthenticated` / `KG_FABRIC_ALLOW_UNAUTHENTICATED=true` required when `KG_FABRIC_AUTH_MODE=none`).
- **Authentication, Tenant Isolation & ABAC Enforcement (`auth.py`, `api.py`)**:
  - Supports `KG_FABRIC_AUTH_MODE=api_key` (static or JSON principal map) and `KG_FABRIC_AUTH_MODE=jwt` (strict `exp` + `tenant_id` validation via PyJWT or built-in stdlib HS256).
  - Enforces tenant isolation (`403` on cross-tenant smuggling), role checks (`writer` / `admin`), and caller ABAC `acl_groups` intersection across all FastAPI endpoints and MCP tool invocations (`/mcp`, `/mcp/http`, `/v1/mcp/call`).
- **Pluggable Domain Packs (`domain_packs.py`)**: Extracted fallback ontology rules into `DomainRulePack` with built-in `CLOUD_SRE_DOMAIN_PACK` and domain-neutral `GENERIC_DOMAIN_PACK` (`KG_FABRIC_DOMAIN_PACK=generic|cloud_sre`), plus full SDK extensibility.
- **Env-Auto-Wired Factory & FastAPI `lifespan`**: Added `EnterpriseKGOntologyService.from_env()`, `create_kg_fabric_app_from_env()`, standard `PGHOST`/`PGPORT`/`PGDATABASE`/`PGUSER`/`PGPASSWORD` + `PGVECTOR_DSN` support, and graceful `async def close()` pool/driver shutdown.
- **Durable PostgreSQL Sync Jobs & DLQ**: Added `kg_sync_jobs` and `kg_dlq_chunks` tables to PostgreSQL DDL and wired `save_sync_job`, `get_sync_job_async`, `append_dlq_record`, and `list_dlq_records_async` for multi-pod EKS durability.
- **Self-Hosted Full-Stack Compose & CI/CD**: Added optional `kg-fabric-api` service profile (`docker compose --profile app up -d --build`), `.env.example`, and GitHub Actions CI (`ci.yml`) + PyPI Trusted Publishing release workflow (`release.yml`).

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
