# Enterprise Multi-KB Automated Ontology & Knowledge Graph Fabric (`kg-ontology-fabric` v0.2.0)

An enterprise-grade service that automatically induces, aligns, and governs **formal domain ontologies (W3C OWL 2 / SHACL TBox)** and **bi-temporal, ABAC-trimmed knowledge graphs (ABox)** from **100s of heterogeneous PostgreSQL `pgvector` embedding stores** (`384d` / `768d` / `1024d` / `1536d`) and **multimodal/raw document streams**, powered by **LlamaIndex**, **Neo4j 5 / Amazon Neptune (`openCypher`)**, **PostgreSQL (`pgvector`)**, **Microsoft Presidio**, **HippoRAG 2 PPR**, **Leiden Global Search**, **IBM Docling**, and co-hosted **FastAPI + MCP (`2026-07-28`)**.

---

## Documentation Hub

* **[Architecture, Algorithms & Explainer Guide](./docs/architecture_and_explainer_guide.md)** — Deep dive into Sub-Linear Vector Centroid Sampling, Cross-KB Synonym Alignment, `SiblingCollisionGuard`, Bi-Temporal Functional Edges, W3C OWL/SHACL Validation, and Hybrid GraphRAG.
* **[Usage, Multi-Model Vector Federation, Presidio PII & AWS EKS Deployment Guide](./docs/usage_and_eks_deployment_guide.md)** — End-to-end API/SDK usage, Zero-Re-Embedding Multi-Model Query Fan-Out, Microsoft Presidio PII configuration, Amazon EKS manifests (`deploy/eks/`), and Enterprise Maturity Roadmap.
* **[Live Docker Messy Cross-KB Benchmark](./examples/live_docker_messy_enterprise_test.py)** & **[Observed Results JSON](./live_docker_observation_report.json)** — Head-to-head comparison against live Docker `pgvector` + Neo4j 5 (`9/9` recall vs. `3/9` Vector-Only RAG).
* **[Live Internet Wikipedia 5-KB Benchmark](./examples/live_internet_wikipedia_kbs_e2e.py)** & **[Report JSON](./live_internet_kbs_report.json)** — 25 real Wikipedia articles across 5 heterogeneous KBs on live Docker `pgvector` + Neo4j 5.
* **[120-KB Enterprise Scale Simulation](./examples/simulate_120_kbs_end_to_end.py)** & **[Benchmark Report JSON](./simulation_benchmark_report.json)**.

---

## What's New in `v0.2.0` (2024–2026 SOTA GraphRAG & Enterprise Production Hardening)

Designed to **reuse proven industry & workspace libraries** instead of reinventing the wheel (with deterministic zero-dependency fallbacks when optional extras are not installed):

| Capability | Module / Surface | Reused Library / Standard |
| :--- | :--- | :--- |
| **Live Structured-Output LLM Extraction + Circuit Breaker** | `llm_extractor.py` | `google-genai` (Structured JSON), `boto3` Bedrock, `openai`, `agentic_memory_sdk.governance.AsyncCircuitBreaker` |
| **HippoRAG 2 Personalized PageRank (PPR) + Hub Damping** | `graph_analytics.py`, `POST /v1/search/ppr`, MCP `search_ppr_subgraph` | `networkx.pagerank` + inverse-degree hub damping (`ppr / sqrt(deg)`) |
| **Microsoft GraphRAG Leiden/Louvain Global Sensemaking** | `graph_analytics.py`, `POST /v1/search/global`, MCP `query_global_communities` | `leidenalg` + `python-igraph` / `networkx.community.louvain_communities` |
| **Adaptive-RAG Query Router + Cross-Encoder Reranker** | `adaptive_router.py`, `rag_engine.py` | `sentence-transformers` (`CrossEncoder`) / Bedrock Rerank |
| **Document/Entity/Edge ABAC Security Trimming + N-ary Qualifiers** | `adapters/pgvector_store.py`, `adapters/graph_stores.py`, `models.py` | End-to-end `caller_acl_groups` enforcement (zero-leakage verified) + `qualifiers` (`condition`, `scope`, `exception`) |
| **Multimodal & Tabular Document Ingestion** | `multimodal_ingestor.py` | **IBM Docling** (`docling.document_converter.DocumentConverter`) + Markdown/CSV table-to-triples parser |
| **Persistent Multi-Pod TBox Store** | `adapters/pgvector_store.py` (`enterprise_tbox_registry`) | PostgreSQL JSONB persistence for cold-start EKS HPA pods |
| **Async Sync Jobs, DLQ, OpenTelemetry & Eval Harness** | `service.py`, `api.py`, `eval_harness.py` (`POST /v1/jobs/build`, `GET /v1/jobs/dlq`, `POST /v1/eval/run`) | `opentelemetry-api` / `opentelemetry-sdk`, `ragas`-compatible Continuous GraphRAG Eval |
| **Co-Hosted FastAPI + MCP Server (`2026-07-28`)** | `mcp_server.py`, `POST /mcp`, `/mcp/http`, `/v1/mcp/call` | `mcp` FastMCP — **12 MCP Tools, 4 Resources, 2 Prompts** |

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

## 4. Quickstart

### A. Run Unit Tests & 120-KB Scale Simulation (No Docker Required)
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### B. Run Live Docker Messy Cross-KB Benchmark (`pgvector` + `Neo4j 5`)
```bash
docker compose up -d --wait
python examples/live_docker_messy_enterprise_test.py
```
Inspect the live induced ontology and knowledge graph in Neo4j Browser at `http://localhost:7475` (`bolt://localhost:7688`).

### C. Run FastAPI + MCP Microservice Locally or Deploy to Amazon EKS
```bash
# Local FastAPI + MCP server (install optional SOTA extras as needed)
pip install -e ".[pii,aws,mcp,llm,graph-analytics,multimodal,rerank,eval,otel]"
python -m spacy download en_core_web_lg
uvicorn enterprise_kg_fabric.api:create_kg_fabric_app --factory --host 0.0.0.0 --port 8080

# Amazon EKS Deployment (see docs/usage_and_eks_deployment_guide.md)
kubectl apply -f deploy/eks/deployment-api.yaml
kubectl apply -f deploy/eks/cronjob-incremental-sync.yaml
```

---

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
