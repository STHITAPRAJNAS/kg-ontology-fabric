# Enterprise Multi-KB Automated Ontology & Knowledge Graph Fabric (`enterprise-kg-ontology-fabric` v0.3.0)

An enterprise-grade **Python SDK, CLI (`kg-fabric`), and self-hostable FastAPI + MCP microservice** that automatically induces, aligns, and governs **formal domain ontologies (W3C OWL 2 / SHACL TBox)** and **bi-temporal, ABAC-trimmed knowledge graphs (ABox)** from **heterogeneous PostgreSQL `pgvector` embedding stores** (`384d` / `768d` / `1024d` / `1536d`) and **multimodal/raw document streams**, backed by **PostgreSQL 16 (`pgvector` + RLS)**, **Neo4j 5 / Amazon Neptune (`openCypher`)**, **HippoRAG 2 PPR**, **Leiden/Louvain Global Search**, **Microsoft Presidio PII**, **IBM Docling**, **Automated KG Quality Scorecard**, and co-hosted **FastAPI + MCP (`2025-11-25`)**.

> **Dual-Mode Distribution**: Use it **as a Python library/CLI** embedded into your existing Python/ADK/LangGraph/FastAPI apps **OR clone & self-host** the full containerized service via `docker compose --profile app up -d --build` or Amazon EKS (`deploy/eks/`).

---

## Documentation Hub

* **[Architecture, Algorithms & Explainer Guide](./docs/architecture_and_explainer_guide.md)** — Deep dive into Sub-Linear Vector Centroid Sampling, Cross-KB Synonym Alignment, `SiblingCollisionGuard`, Bi-Temporal Functional Edges, W3C OWL/SHACL Validation, and Hybrid GraphRAG.
* **[Usage, Multi-Model Vector Federation, Presidio PII & AWS EKS Deployment Guide](./docs/usage_and_eks_deployment_guide.md)** — End-to-end API/SDK usage, Multi-Model Vector Federation, Microsoft Presidio PII configuration, Amazon EKS manifests (`deploy/eks/`), and Enterprise Maturity Roadmap.
* **[Live Docker v0.3.0 Adversarial & RLS Verification](./examples/live_docker_v03_full_verification.py)** — Full live-Docker verification against PostgreSQL 16 (`NOBYPASSRLS` role `kg_app_user`), Neo4j 5, and live Uvicorn HTTP (JWT, anti-smuggling, ABAC, legacy table auto-hydration, `pg_try_advisory_lock`, `KGQualityAuditor`, and cold-start hydration).
* **[Live Docker Messy Cross-KB Benchmark](./examples/live_docker_messy_enterprise_test.py)** & **[Observed Results JSON](./live_docker_observation_report.json)** — Head-to-head synthetic cross-KB comparison against live Docker `pgvector` + Neo4j 5 (`9/9` recall vs. `3/9` Vector-Only RAG).
* **[Live Internet Wikipedia 5-KB Benchmark](./examples/live_internet_wikipedia_kbs_e2e.py)** & **[Report JSON](./live_internet_kbs_report.json)** — 25 real Wikipedia articles across 5 heterogeneous KBs on live Docker `pgvector` + Neo4j 5.
* **[120-KB Enterprise Scale Simulation](./examples/simulate_120_kbs_end_to_end.py)** & **[Benchmark Report JSON](./simulation_benchmark_report.json)**.

---

## Key Capabilities (`v0.3.0`)

Designed to **reuse proven industry libraries** when installed while providing deterministic zero-dependency offline fallbacks for air-gapped CI & tests:

| Capability | Module / Surface | Library / Standard |
| :--- | :--- | :--- |
| **Structured-Output LLM Extraction + Circuit Breaker** | `llm_extractor.py` | `google-genai` (Gemini), `openai`, `boto3` (AWS Bedrock Converse), `AsyncCircuitBreaker` (falls back to pluggable `DomainRulePack` regex extractor when `KG_FABRIC_USE_LIVE_LLM=false`) |
| **HippoRAG 2 Personalized PageRank (PPR) + Hub Damping** | `graph_analytics.py`, `POST /v1/search/ppr`, MCP `search_ppr_subgraph` | `networkx.pagerank` + inverse-degree hub damping (`ppr / sqrt(deg)`) |
| **Microsoft GraphRAG Leiden/Louvain Global Sensemaking** | `graph_analytics.py`, `POST /v1/search/global`, MCP `query_global_communities` | `leidenalg` + `python-igraph` / `networkx.community.louvain_communities` |
| **Adaptive-RAG Query Router + Cross-Encoder Reranker** | `adaptive_router.py`, `rag_engine.py` | `sentence-transformers` (`CrossEncoder`) / Bedrock Rerank |
| **Multi-Tenant Auth, Fail-Closed RLS & ABAC Trimming** | `auth.py`, `adapters/pgvector_store.py`, `adapters/graph_stores.py` | Pure-ASGI JWT (`>=32B` secret), API Key, BYOA hooks, bounded rate limiter (`429`), PostgreSQL `NOBYPASSRLS` RLS policies, cross-tenant KB/chunk overwrite guards, Neo4j `READ_ACCESS` sessions, end-to-end `caller_acl_groups` |
| **4-Dimension Ontology & KG Quality Scorecard** | `quality_auditor.py`, `GET /v1/quality/report`, MCP `audit_kg_quality` | W3C `pyshacl`, schema utilization, domain/range compliance, sampled faithfulness (deterministic or LLM-as-a-Judge), ER duplicate/over-merge detection, orphan/GCC/modularity metrics + Vector-vs-GraphRAG lift in `POST /v1/eval/run` |
| **Multimodal & Tabular Document Ingestion** | `multimodal_ingestor.py` | **IBM Docling** (`DocumentConverter`) + Markdown/CSV table-to-triples parser |
| **Persistent Multi-Pod TBox, Durable Jobs/DLQ & Sync Worker** | `adapters/pgvector_store.py`, `service.py` | PostgreSQL `enterprise_tbox_registry`, `kg_sync_jobs`, `kg_dlq_chunks`, multi-replica `pg_try_advisory_lock` sync worker, `DELETE` purge APIs |
| **Co-Hosted FastAPI + MCP Server (`2025-11-25`)** | `mcp_server.py`, `POST /mcp`, `/mcp/http`, `/v1/mcp/call` | `mcp` FastMCP — **14 MCP Tools, 4 Resources, 2 Prompts**, `GET /metrics` (Prometheus), `GET /ui` Governance Console |

---

## 1. Why This Exists: Verified Live Docker Comparison (`Vector-Only RAG` vs. `KG Fabric`)

When enterprise knowledge is scattered across siloed vector KBs (SRE runbooks, FinOps ledgers, Compliance audits, SecOps CVE bulletins, CRM contracts), **standard vector search fails on multi-hop cross-KB questions** because no single chunk contains the full chain.

On a messy multi-division audit query (*"If VIP customer `Acme-Global-Bank` escalates an audit, what downstream database, compliance mandates, active CVEs, WAF rules, active cloud region, and active cost center are in its blast radius?"*) across **5 real separate legacy PostgreSQL `pgvector` tables (`384d`, `768d`, `1024d`, `1536d`) + raw Markdown postmortems** on live Docker **PostgreSQL 16 + Neo4j 5**:

| Capability Verified Live on Docker Postgres (`pgvector`) & Neo4j 5 | Baseline Vector-Only RAG | Enterprise Ontology + KG Fabric |
| :--- | :---: | :---: |
| **3-Hop Hidden Cross-KB Blast-Radius Recall** | **`3 / 9` (33%)** | **`9 / 9` (100%)** across all 6 siloed KBs |
| **Cross-KB Entity Resolution** (`"Payment Router 01"` ↔ `payment-router-01`) | None | **Merged** into 1 canonical Neo4j bridge vertex |
| **Stale Fact Filtering** (`us-east-1` & `CC-104-Core` superseded) | Returns conflicting chunks | **Closed** automatically (`t_invalid`) |
| **Bi-Temporal Forensic Time-Travel** (`as_of_iso="2026-06-01"`) | Impossible | **Verified** (recalls `us-east-1` & `CC-104-Core`) |
| **Sibling Canary Isolation** (`payment-router-02`, `PCI-DSS-3.2`, `CVE-2026-9982`) | Pollutes context | **Strictly isolated** (`SiblingCollisionGuard`) |
| **Pre-Ingestion Microsoft Presidio PII Redaction** | Raw PII embedded | **`0` PII leaks** (`CVE`/`PCI`/`CC` preserved) |
| **Incremental SHA-256 Watermark Sync (Pass 2)** | Re-extracts everything | **`100%` unchanged chunks skipped** (`0` delta) |
| **Cold Process Restart Persistence (Zero RAM Cache)** | N/A | **Verified** directly from Postgres + Neo4j disk |
| **W3C OWL 2 / SHACL Conformance (`pyshacl`)** | None | **Passed (`true`)** |

---

## 2. Architecture Overview

```text
+---------------------------------------------------------------------------------------+
| 1. MULTI-KB FEDERATION & LLAMAINDEX RAW DOCUMENT INGESTION                            |
|                                                                                       |
|  [100s of Legacy pgvector KBs]                 [Raw Unstructured Docs: MD/PDF/S3]     |
|   - SRE (384-dim BGE)  | FinOps (768-dim)       - Microsoft Presidio PII Redactor     |
|   - Compliance (1536d) | CRM (1024-dim)         - Technical Entity Allow-List Guard   |
|   - Zero re-embedding of legacy rows            - LlamaIndexHierarchicalNodeParser    |
|                         |                                       |                     |
|                         +-------------------+-------------------+                     |
|                                             v                                         |
|                 PostgreSQL `pgvector` Federation + SHA-256 CDC Watermarks             |
+---------------------------------------------+-----------------------------------------+
                                              |
                                              v
+---------------------------------------------------------------------------------------+
| 2. SUB-LINEAR VECTOR-CLUSTER ONTOLOGY INDUCTION & CROSS-KB ALIGNMENT (TBox)           |
|                                                                                       |
|  - SubLinearVectorCentroidSampler: K-Means++ medoids + boundary outliers per KB       |
|  - AutomatedOntologyInducer: Aligns synonyms across divisions into Layer-0 anchored   |
|    `EnterpriseCompiledTBox` (`APPROVED` vs `QUARANTINED_CANDIDATE` Two-Bucket model)  |
|  - OWLSHACLCompiler: Compiles W3C OWL 2 / RDFS / SHACL `.ttl` & validates with pyshacl|
+---------------------------------------------+-----------------------------------------+
                                              |
                                              v
+---------------------------------------------------------------------------------------+
| 3. LLAMAINDEX PROPERTY GRAPH EXTRACTION & CROSS-KB ENTITY RESOLUTION (ABox)           |
|                                                                                       |
|  - Incremental SHA-256 Watermark CDC: Skips 99%+ unchanged chunks across 100s of KBs  |
|  - LlamaIndex SchemaLLMPathExtractor + DynamicLLMPathExtractor + Conjunction CarryOver|
|  - CrossKBEntityResolver + SiblingCollisionGuard + Bi-Temporal Functional Edges       |
+---------------------------------------------+-----------------------------------------+
                                              |
                        +---------------------+---------------------+
                        v                                           v
+-------------------------------------------+ +-----------------------------------------+
| PostgreSQL 16 (`pgvector` HNSW + GIN)     | | Neo4j 5 (`bolt://`) or AWS Neptune      |
| - `kb_federation_registry`                | | (`openCypher` HTTPS SigV4)              |
| - `kb_ingestion_watermarks` (SHA-256 CDC) | | - `:OntologyClass` & `:OntologyRelation`|
| - `unified_rag_chunks` (HNSW + tsvector)  | | - `:Entity` with cross-KB `source_kbs`  |
| - `canonical_entity_embeddings` (HNSW)    | | - Bi-Temporal `:KG_RELATION` edges      |
+-------------------------------------------+ +-----------------------------------------+
                        ^                                           ^
                        +---------------------+---------------------+
                                              |
+---------------------------------------------+-----------------------------------------+
| 4. LLAMAINDEX 4-PATH HYBRID GraphRAG ENGINE (`EnterpriseHybridGraphRAGEngine`)        |
|                                                                                       |
|  Path 1: Multi-Model Native Vector Fan-Out (`FederatedEmbeddingModelRouter` + RRF)    |
|  Path 2: PostgreSQL `tsvector` GIN BM25 Lexical Search                                |
|  Path 3: K-Hop PropertyGraph Expansion + Cross-KB Bridges + `as_of_iso` Time-Travel   |
|  Path 4: Ontology-Grounded Multi-Hop Text-to-OpenCypher Execution                     |
+---------------------------------------------------------------------------------------+
```

---

## 3. How to Use This Service Practically (Real-World Adoption Playbook)

Rather than ripping and replacing existing RAG pipelines, deploy `kg-ontology-fabric` in **4 practical stages**:

1. **Stage 1 — Non-Invasive Overlay on Existing `pgvector` KBs (Week 1)**:
   * Keep all existing team RAG pipelines and `pgvector` tables untouched (regardless of whether they use OpenAI `1536d`, Cohere `1024d`, Gemini `768d`, or BGE `384d`).
   * Register their Postgres tables via `POST /v1/kbs/register-batch` with their `VectorKBColumnMapping`.
   * Run `POST /v1/ontology/induce` to automatically discover a unified **Enterprise TBox Ontology** across all teams at a fraction of LLM cost (centroid sampling), and export W3C OWL/SHACL Turtle (`GET /v1/ontology/export.ttl`) for Data Governance review.
2. **Stage 2 — Continuous Incremental Sync Worker (Week 2)**:
   * Deploy the **EKS CronJob** ([`deploy/eks/cronjob-incremental-sync.yaml`](./deploy/eks/cronjob-incremental-sync.yaml)) calling `POST /v1/graph/build` (`incremental=true`) every 15 minutes.
   * Only new or modified chunks (detected via PostgreSQL `kb_ingestion_watermarks` SHA-256 hashes) are scrubbed by Microsoft Presidio and extracted into Neo4j / Amazon Neptune.
3. **Stage 3 — Power Enterprise AI Agents via `POST /v1/rag/query` (Week 3)**:
   * Wire your **Google ADK**, **LangGraph**, or **LlamaIndex** agents to call `POST /v1/rag/query` as a unified cross-enterprise retrieval tool.
   * Agents immediately gain **multi-hop blast-radius analysis**, **cross-division entity resolution**, **bi-temporal audit history (`as_of_iso`)**, and **exact KB/chunk provenance citations**.
4. **Stage 4 — Governed Bottom-Up Ontology Evolution (Ongoing)**:
   * As new KBs introduce novel concepts (e.g., new compliance mandates or Zero-Day classes), they land safely in `QUARANTINED_CANDIDATE` without polluting production graphs, and stewards promote validated concepts via `POST /v1/ontology/promote`.

---

## 4. Quickstart — Two Ways to Use It

### Mode 1: Install as a Python Package / CLI (PyPI-Ready SDK)
Embed the engine in your own FastAPI/ADK/LangGraph app, plug in a custom `DomainRulePack` (or `GENERIC_DOMAIN_PACK`), or launch via the built-in `kg-fabric` CLI:
```bash
# Install core SDK + CLI (or add optional extras: pii,aws,mcp,llm,graph-analytics,multimodal,rerank,eval,otel,auth)
pip install -e ".[mcp,auth,graph-analytics]"

# 1. Start co-hosted FastAPI REST + MCP HTTP server on loopback (or set KG_FABRIC_AUTH_MODE=jwt|api_key for 0.0.0.0)
kg-fabric serve --host 127.0.0.1 --port 8080

# 2. Or start standalone FastMCP over stdio for Claude Desktop / Cursor / Windsurf / ADK
kg-fabric mcp --transport stdio

# 3. Initialize PostgreSQL pgvector schema (registry, watermarks, persistent TBox, durable jobs & DLQ)
kg-fabric init-db
```

### Mode 2: Clone & Self-Host as a Full Service (Docker Compose / Amazon EKS)
```bash
git clone https://github.com/STHITAPRAJNAS/kg-ontology-fabric.git
cd kg-ontology-fabric
cp .env.example .env

# Start PostgreSQL 16 (pgvector) + Neo4j 5 + FastAPI/MCP API container in one command:
docker compose --profile app up -d --build

# Or start only Postgres + Neo4j for local benchmarks & tests:
docker compose up -d --wait
python examples/live_docker_messy_enterprise_test.py

# Run full offline pytest suite (23 tests, no Docker required) & ruff lint:
pip install -e ".[dev,mcp,auth]"
ruff check src tests
pytest tests/ -v

# Amazon EKS Deployment (see docs/usage_and_eks_deployment_guide.md)
kubectl apply -f deploy/eks/deployment-api.yaml
kubectl apply -f deploy/eks/cronjob-incremental-sync.yaml
```
Inspect the live induced ontology and knowledge graph in Neo4j Browser at `http://localhost:7475` (`bolt://localhost:7688`).

### Security, Multi-Tenant Isolation & Honest Implementation Scope
- **Auth, OIDC/JWKS & Isolation**: Supports `KG_FABRIC_AUTH_MODE=jwt` (strict `exp`, `tenant_id`, minimum `32`-byte HS256 secret, or built-in OIDC/JWKS RS256/ES256 verification via `KG_FABRIC_JWKS_URL` + `KG_FABRIC_JWT_ISSUER`/`AUDIENCE`), `api_key`, and `custom` BYOA hooks. Unknown `KG_FABRIC_AUTH_MODE` values fail closed (`ValueError`). Global **platform admins** (`roles=["admin"]` with `tenant_id="*"` or unscoped) are distinguished from **tenant-bound admins** (`roles=["admin"]` with a specific `tenant_id`).
- **Tenant & ABAC Guards**: Enforces tenant boundaries across REST, MCP (15 tools with full HTTP + `stdio` parity, including nested `documents`/`registrations` and MCP resources), PostgreSQL `NOBYPASSRLS` RLS policies (`kg-fabric migrate-db --app-role kg_app_user`), cross-tenant KB/chunk overwrite guards (even under DB superuser), composite `(tenant_id, entity_id)` primary keys & tenant-scoped `entity_id`s in `canonical_entity_embeddings`, tenant-scoped `/v1/jobs`, `/v1/jobs/dlq`, `/v1/metrics`, and `/metrics`, and Path 4 OpenCypher post-filtering. Arbitrary raw Cypher (`POST /v1/search/cypher` & MCP `execute_readonly_opencypher`) requires **global platform admin** in authenticated mode and runs inside Neo4j `READ_ACCESS` sessions.
- **Fail-Closed Legacy Table Hydration, Incremental Sync & Complete Deletion**: Reading physical `EXISTING_PGVECTOR_KB` tables is **disabled by default** — operators must explicitly allowlist tables via `KG_FABRIC_ALLOWED_LEGACY_TABLES` (comma-separated `schema.table` globs) and register/modify them as a global platform admin; PostgreSQL system catalogs and internal KG Fabric tables are unconditionally denied. Incremental sync supports `updated_at_column` cursors, multi-DB `KG_FABRIC_PG_DSN_ALIASES`, SHA-256 skip, and stale-fact retraction when modified chunks are re-extracted. `DELETE /v1/documents/{kb}/{doc}`, `DELETE /v1/kbs/{kb}`, and `DELETE /v1/tenants/{tenant_id}` prune chunks, graph provenance/entities/edges, narrow surviving entity/edge `acl_groups`, invalidate cached communities, and delete orphaned `canonical_entity_embeddings`.
- **Core Pipeline, Scale & Governance**:
  - **Embeddings & Live AI**: `from_env()` wires `KG_FABRIC_EMBEDDING_PROVIDER` (`hash` zero-key default | `gemini` | `openai` | `bedrock` | `sentence_transformers`). When `KG_FABRIC_USE_LIVE_LLM=true` (`gemini`, `openai`, or AWS `bedrock`), extraction runs concurrently (`KG_FABRIC_LLM_CONCURRENCY=8`, non-blocking `asyncio.to_thread` + semaphore), and optional LLM-assisted ontology induction, structured JSON intent-to-Cypher template compilation (never executing raw model-written Cypher), and writer/admin-gated cited answer synthesis (`synthesize_with_llm=True`) activate cleanly. Plain `reader` callers trigger zero LLM calls and see redacted physical DB fields on `GET /v1/kbs`.
  - **Scalable Queries & Async Jobs**: Query paths use indexed entity lookup (`canonical_entity_embeddings` HNSW + Neo4j `TEXT INDEX` with Cypher-level ABAC filtering before `LIMIT`), bounded K-hop ego-subgraphs for PPR, and build-time persisted `kg_community_summaries` (with ABAC filtering) rather than full-tenant graph scans. `POST /v1/jobs/build` returns `202 Accepted` immediately on a per-tenant concurrency-limited background worker (`409` on overlap, automatic orphan recovery on restart) alongside `GET /v1/jobs`, `POST /v1/jobs/{id}/cancel`, `GET /v1/kbs`, ABAC-filtered `GET /v1/kbs/{kb}/documents`, and `GET /v1/entities/{entity_id}`.
  - **Per-Concept Ontology Governance**: `POST /v1/ontology/review` (`approve` / `reject` / `merge` propagating to Neo4j/Neptune) + immutable `GET /v1/ontology/audit` trail + interactive `/ui` Governance Console controls.
- **Storage Backend Notes**: PostgreSQL 16 (`pgvector`, supporting native-vector RRF fan-out over allowlisted legacy tables + `KG_FABRIC_PG_DSN_ALIASES`) + Neo4j 5 are live-Docker verified. Amazon Neptune (`boto3` `neptunedata` `/openCypher`) is implemented for full read/write/governance operations and unit-tested, but not tested against a live AWS Neptune cluster in this repo.
- **Per-Worker State**: Compiled TBoxes refresh from PostgreSQL on `load_or_get_tbox`, while the built-in `BoundedApiRateLimiter` and Prometheus counters are per-worker (front multi-worker/EKS pods with API Gateway/Redis rate limiting and Prometheus scrape aggregation).
- See [CHANGELOG.md](./CHANGELOG.md) for full release history and `.github/workflows/release.yml` for tag-triggered PyPI Trusted Publishing (`v*.*.*`).

---

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
