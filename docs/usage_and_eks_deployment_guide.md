# Enterprise Usage Guide, Multi-Model Vector Federation, Presidio PII & AWS EKS Deployment

This guide covers:
1. **Microsoft Presidio PII & Secret Redaction** (including protecting technical KG entities from false positives)
2. **How the Service Handles 100s of KBs with Different Embedding Models** (Zero-Re-Embedding Dual-Space Architecture)
3. **Developer & Operator Usage Guide** (Python SDK & FastAPI REST API)
4. **Production Deployment on Amazon EKS** (with Amazon Neptune, Amazon Aurora PostgreSQL `pgvector`, and IRSA)
5. **Enterprise Maturity & Product Improvement Blueprint**

---

## 1. Robust PII & Secret Redaction with Microsoft Presidio

Every chunk hydrated from legacy `pgvector` KBs and every raw document ingested via LlamaIndex passes through [`PresidioPIIRedactor`](../src/enterprise_kg_fabric/pii_redaction.py) **before** embeddings, ontology induction, or graph writes occur.

### Key Capabilities
* **Native Microsoft Presidio Engine (`presidio-analyzer` + `presidio-anonymizer`)**:
  Install the `pii` extra (`pip install "enterprise-kg-ontology-fabric[pii]"` + `python -m spacy download en_core_web_lg`). In production EKS containers ([`Dockerfile`](../Dockerfile)), the spaCy model is **pre-baked** into the Docker image so pods start in private subnets with zero internet egress.
* **Custom Enterprise Secret `PatternRecognizer`s**:
  Stock Presidio detects `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`, `CREDIT_CARD`, `IBAN_CODE`, and `IP_ADDRESS`. We register additional custom Presidio `PatternRecognizer`s for enterprise secrets:
  * `BEARER_JWT_TOKEN` (HTTP Authorization JWT tokens)
  * `AWS_ACCESS_KEY_ID` (IAM access key identifiers)
  * `PRIVATE_KEY_HEADER` (PEM RSA/EC/OpenSSH private key headers)
  * `GENERIC_API_SECRET` (inline key-value credentials)
* **Technical KG Entity Allow-List Guard (Critical for Knowledge Graphs!)**:
  Standard NLP PII models frequently misclassify technical identifiers as phone numbers or SSNs (e.g., `CVE-2026-9981`, `PCI-DSS-4.0`, `CC-209-Global`, `INC-99421`, `WAF-Shield-Rule-904`, `us-east-1`). `PresidioPIIRedactor` computes protected technical spans **first** and suppresses overlapping PII false positives so your Knowledge Graph topology stays 100% intact.
* **3 Redaction Modes (`PIIRedactionMode`)**:
  1. `REPLACE_TAG` (default): Replaces PII with `<US_SSN>`, `<EMAIL_ADDRESS>`, `<BEARER_JWT_TOKEN>`.
  2. `HMAC_PSEUDONYMIZE`: Uses a KMS-managed HMAC secret key to replace PII with deterministic tokens like `<EMAIL_ADDRESS_a91f3c2d>` — enabling cross-KB graph linking across multiple KBs *without ever storing raw PII in Neo4j/Neptune or `pgvector`*.
  3. `MASK`: Partial masking (`***-**-6789`).

### Example: Configuring Presidio Redaction
```python
from enterprise_kg_fabric import (
    EnterpriseKGOntologyService,
    PIIRedactionMode,
    PresidioPIIRedactor,
)

redactor = PresidioPIIRedactor(
    mode=PIIRedactionMode.HMAC_PSEUDONYMIZE,
    hmac_secret_key="loaded-from-aws-secrets-manager",
    score_threshold=0.70,
    custom_allow_list=["InternalProject-4091"],
)

result = redactor.analyze_and_redact(
    "Contact alice@bank.com (SSN 123-45-6789) regarding CVE-2026-9981 on payment-router-01."
)
print(result.scrubbed_text)
# -> Contact <EMAIL_ADDRESS_...> (SSN <US_SSN_...>) regarding CVE-2026-9981 on payment-router-01.
print(result.redaction_counts)
# -> {'EMAIL_ADDRESS': 1, 'US_SSN': 1}
```

---

## 2. How the Service Handles 100s of KBs with Different Embedding Models

In an enterprise with 100+ existing `pgvector` stores, you will inevitably have **incompatible embedding spaces**:
* SRE KBs (`kb_001..020`): `bge-small-en-v1.5` (`384-dim`)
* FinOps & SecOps KBs (`kb_021..040`, `kb_061..080`): `text-embedding-004` (`768-dim`)
* Customer SLA KBs (`kb_081..100`): `embed-english-v3.0` (`1024-dim`)
* Compliance KBs (`kb_041..060`): `text-embedding-3-large` (`1536-dim`)

**Golden Rule of Vector Math**: You can **never** directly compare a cosine score from a `384-dim` BGE vector against a `1536-dim` OpenAI vector (even with padding/PCA, their latent manifolds and similarity score distributions are incompatible). And re-embedding 50 million legacy chunks into one model is slow and expensive.

`enterprise-kg-ontology-fabric` solves this with a **3-Layer Dual-Space Federation Architecture**:

```text
                       USER QUERY: "Which PCI standard governs payment-router-01?"
                                                |
                   +----------------------------+----------------------------+
                   |                                                         |
                   v                                                         v
+----------------------------------------------------+   +-----------------------------------------+
| A. MULTI-MODEL NATIVE QUERY FAN-OUT                |   | B. UNIFIED CANONICAL ENTITY INDEX       |
|    (`FederatedEmbeddingModelRouter`)               |   |    (`canonical_entity_embeddings`)      |
|                                                    |   |                                         |
| 120 KBs collapse to ONLY 4 distinct model groups:  |   | ONLY extracted canonical entities       |
|  - Group 1: bge-small-en-v1.5 (384d) -> 40 KBs     |   | (~1-3% of total chunk volume!) are      |
|  - Group 2: text-embedding-004 (768d) -> 40 KBs    |   | embedded using ONE enterprise canonical |
|  - Group 3: embed-english-v3.0 (1024d) -> 20 KBs   |   | model (e.g. Amazon Titan v2 / Gecko).   |
|  - Group 4: text-embedding-3-large (1536d) -> 20KB |   |                                         |
|                                                    |   | Used for:                               |
| Embeds query ONCE per model group (4 tiny calls!)  |   |  - Cross-KB Entity Resolution           |
| Searches each group in its OWN native pgvector     |   |  - Seed Entity Matching -> Neo4j/Neptune|
| HNSW index (ZERO chunk re-embedding required!)     |   +-----------------------------------------+
+-------------------------+--------------------------+
                          |
                          v
+--------------------------------------------------------------------------------------------------+
| C. SCALE-INVARIANT RANK-BASED RECIPROCAL RANK FUSION (RRF)                                       |
|    RRF_score(chunk) = SUM_channel [ 1 / (60 + rank_in_channel) ]                                 |
|    Uses ONLY rank positions (1st, 2nd, 3rd) across each model group + BM25, completely           |
|    eliminating raw cosine scale bias between OpenAI, Cohere, BGE, and Gemini embeddings!         |
+--------------------------------------------------------------------------------------------------+
```

1. **Ontology Induction (Intra-KB Native Space)**: `SubLinearVectorCentroidSampler` runs K-Means++ medoid & boundary outlier sampling **inside each KB's native vector space** (`384d`, `768d`, `1024d`, `1536d`). No cross-KB vector comparison is needed during centroid sampling!
2. **Chunk Retrieval (Zero-Re-Embedding Multi-Model Fan-Out)**: `FederatedEmbeddingModelRouter` groups all 120 KBs by `(embedding_model_id, embedding_dim)` (usually only 3–5 distinct models), embeds the query once per model family in parallel (`<20ms`), searches each family's native `pgvector` HNSW index, and merges results via **scale-invariant rank-based RRF**.
3. **Cross-KB Entity Resolution (Tiny Canonical Entity Space)**: Only extracted entities (e.g., 5,000 canonical nodes instead of 5,000,000 raw chunks) are embedded with one canonical enterprise embedding model into `canonical_entity_embeddings` for Cross-KB deduplication.

---

## 3. Usage Guide (Local, SDK & REST API)

### A. Running Locally (FastAPI Server)
```bash
cd enterprise-kg-ontology-fabric

# Option 1: Start local PostgreSQL (pgvector:pg16) + Neo4j 5 via Docker Compose
docker compose up -d --wait

# Option 2: Start FastAPI microservice directly
PYTHONPATH="src" uvicorn enterprise_kg_fabric.api:create_kg_fabric_app --factory --host 0.0.0.0 --port 8080
```

### B. REST API Workflow (`curl`)
```bash
# 1. Register existing pgvector KBs (can batch-register 100s at once)
curl -X POST http://localhost:8080/v1/kbs/register-batch \
  -H "Content-Type: application/json" \
  -d '{
    "registrations": [
      {"kb_id": "kb_001_sre", "tenant_id": "acme", "domain_tag": "sre", "table_name": "vec_sre", "embedding_dim": 384, "embedding_model_id": "bge-small-en-v1.5"},
      {"kb_id": "kb_002_compliance", "tenant_id": "acme", "domain_tag": "compliance", "table_name": "vec_comp", "embedding_dim": 1536, "embedding_model_id": "text-embedding-3-large"}
    ]
  }'

# 2. Ingest raw Markdown/runbook documents (Presidio PII scrubbing + LlamaIndex hierarchical chunking)
curl -X POST http://localhost:8080/v1/documents/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "doc_id": "runbook_payments",
      "kb_id": "kb_001_sre",
      "tenant_id": "acme",
      "domain_tag": "sre",
      "title": "Payments Topology",
      "content": "# Core Topology\npayment-router-01 depends on aurora-ledger-01 and is deployed in primary eu-central-1."
    }]
  }'

# 3. Induce & align enterprise ontology (TBox) across all registered KBs
curl -X POST http://localhost:8080/v1/ontology/induce \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "acme", "tbox_version": "1.0.0"}'

# 4. Build / incrementally update Knowledge Graph (SHA-256 watermarks + Cross-KB ER + SHACL)
curl -X POST http://localhost:8080/v1/graph/build \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "acme", "incremental": true, "validate_shacl": true}'

# 5. Export W3C OWL 2 / SHACL Turtle (.ttl)
curl http://localhost:8080/v1/ontology/export.ttl?tenant_id=acme

# 6. Execute LlamaIndex 4-Path Hybrid GraphRAG Query
curl -X POST http://localhost:8080/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme",
    "query": "Which compliance standards govern the downstream database of payment-router-01?",
    "max_graph_hops": 2
  }'
```

---

## 4. Production Deployment Guide for Amazon EKS

### Architecture on AWS
* **Compute**: Amazon EKS (managed node groups or Karpenter) running:
  * `kg-fabric-api` Deployment (stateless FastAPI + LlamaIndex GraphRAG + pre-baked Presidio, autoscaled 3–15 pods via HPA)
  * `kg-fabric-incremental-sync` CronJob (or SQS/Kafka consumer workers) executing incremental SHA-256 watermark delta builds every 15 minutes
* **Graph Store**: **Amazon Neptune Database** or **Amazon Neptune Analytics** (OpenCypher over HTTPS `:8182`, authenticated via EKS IRSA SigV4 — zero static passwords!)
* **Vector & Watermark Store**: **Amazon Aurora PostgreSQL 16 (`pgvector` 0.7+ HNSW + GIN + RLS)** with RDS Proxy for connection pooling across EKS pods.
* **LLM & Canonical Embeddings**: **Amazon Bedrock** (`Claude 3.7 Sonnet` / `Nova Pro` + `amazon.titan-embed-text-v2:0`) via VPC PrivateLink endpoint.

### Step-by-Step EKS Deployment
1. **Build & Push Pre-Baked Presidio Image to Amazon ECR**:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
   docker build -t enterprise-kg-ontology-fabric:0.1.0 .
   docker tag enterprise-kg-ontology-fabric:0.1.0 123456789012.dkr.ecr.us-east-1.amazonaws.com/enterprise-kg-ontology-fabric:0.1.0
   docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/enterprise-kg-ontology-fabric:0.1.0
   ```

2. **Create EKS IRSA Role (`eks-kg-ontology-fabric-role`)** with least-privilege permissions:
   * `neptune-db:ReadDataViaQuery`, `neptune-db:WriteDataViaQuery` (scoped to your Neptune cluster ARN)
   * `bedrock:InvokeModel` (scoped to canonical embedding & extraction model ARNs)
   * `rds-db:connect` (for Aurora PostgreSQL IAM auth)
   * `secretsmanager:GetSecretValue` (for `pii_hmac_key` and DSNs via External Secrets Operator / CSI Driver)

3. **Initialize Aurora PostgreSQL `pgvector` Schema**:
   Run the DDL produced by `get_production_pgvector_ddl()` once against Aurora PostgreSQL (creates `kb_federation_registry`, `kb_ingestion_watermarks`, `unified_rag_chunks` with HNSW + GIN + RLS, and `canonical_entity_embeddings`).

4. **Apply Kubernetes Manifests**:
   ```bash
   kubectl apply -f deploy/eks/deployment-api.yaml
   kubectl apply -f deploy/eks/cronjob-incremental-sync.yaml

   # Verify rollout and health
   kubectl -n kg-ontology-fabric rollout status deployment/kg-fabric-api
   kubectl -n kg-ontology-fabric get hpa,ingress,cronjob
   ```

---

## 5. Additional Enterprise-Grade Improvements (Product Roadmap)

To harden this from an enterprise v1 platform into a **Tier-0 Fortune 500 Knowledge Fabric**, prioritize these 8 architectural upgrades:

| # | Enterprise Capability | Why It Matters at 100s of KBs Scale | Recommended Implementation Pattern |
| :--- | :--- | :--- | :--- |
| **1** | **Document & Entity-Level ABAC / RBAC Security Trimming** | If `KB_045_M&A_Legal` is `CONFIDENTIAL` and `KB_001_SRE` is `INTERNAL`, GraphRAG traversal must **never** leak a `CONFIDENTIAL` edge or entity to an unauthorized caller. | Stamp `acl_groups` / `security_clearance` on every `KGEntityNode`, `KGRelationEdge`, and `VectorChunkRecord`; inject `WHERE ALL(g IN r.acl_groups WHERE g IN $user_groups)` into every Neo4j/Neptune OpenCypher query and PostgreSQL RLS policy. |
| **2** | **Event-Driven Streaming CDC (Kafka / Debezium / AWS SQS)** | Polling 500 Postgres tables every 15 minutes adds DB load and 15-minute staleness. | Attach **Debezium PostgreSQL Logical Replication** or **AWS DMS → EventBridge → SQS** on source `pgvector` tables so only changed chunk IDs trigger real-time KEDA-scaled EKS extraction workers. |
| **3** | **Dead-Letter Queue (DLQ) + Async Circuit Breaker** | At 100s of KBs, 0.5% of chunks will fail LLM JSON parsing or hit rate limits; a single poison-pill chunk must never abort a 100-KB sync job. | Route failed chunks to a PostgreSQL `kg_extraction_dlq` table / SQS DLQ with exponential backoff + wrap Neo4j/Neptune queries in an `AsyncCircuitBreaker` that degrades gracefully to `pgvector` Hybrid RRF when the graph DB spikes. |
| **4** | **Human-in-the-Loop (HITL) Ontology Governance & Drift UI** | Automated promotion of emergent classes is great for agility, but regulated banks/healthcare need a Data Steward approval gate before mutating the Tier-1 Enterprise TBox. | Expose a review workflow for `QUARANTINED_CANDIDATE` classes/predicates (Slack/Jira webhook or Web UI showing supporting KB count, sample evidence chunks, and SHACL diff) with RBAC-gated `Approve` / `Merge-As-Synonym` / `Reject` actions. |
| **5** | **Cross-Encoder / BGE Reranking & Graph Path Pruning** | 2-hop graph expansion on hub nodes (like `SOC2-Type-II` or `us-east-1`) can return 150+ edges ("supernode explosion"), diluting LLM synthesis context. | Add degree-capped traversal (`LIMIT 20` per hop ordered by edge confidence & recency) + a **Cross-Encoder Reranker** (Cohere Rerank 3.5 / AWS Bedrock Rerank / `bge-reranker-v2-m3`) over fused chunk + linearized graph-fact candidates within a strict `max_context_tokens` budget. |
| **6** | **FinOps Cost Governance & Tiered Model Routing** | Running frontier models (Claude Sonnet / Gemini Pro) on 1,000,000 routine chunks burns budget unnecessarily. | Use **Model Tiering**: route high-signal/novel KBs to frontier models for TBox induction and complex extraction, and route routine/high-volume KBs to fast, cheap models (Gemini Flash / Claude Haiku / Nova Micro) with per-tenant token quotas and cost attribution tags. |
| **7** | **Multi-Tenant Graph Isolation & GDPR Right-to-Be-Forgotten** | Regulated tenants demand cryptographic isolation and auditable deletion of any user/customer lineage across both `pgvector` and Neo4j/Neptune. | Enforce strict `tenant_id` partitioning (or dedicated Neptune clusters/Neo4j databases for Tier-0 tenants) + a `purge_source_lineage(kb_id, doc_id)` API that cascades deletes across `unified_rag_chunks`, `kb_ingestion_watermarks`, and `source_chunk_ids` on graph nodes/edges. |
| **8** | **Continuous GraphRAG Evaluation & OpenTelemetry Observability** | You cannot improve ontology or GraphRAG precision if you don't measure extraction F1, entity resolution precision/recall, and hallucination rate on every TBox upgrade. | Emit **OpenTelemetry** traces/metrics (Prometheus + Grafana: per-KB sync lag, Presidio PII scrub counts, ER merge rate, SHACL failure rate, p95 GraphRAG latency) + run an automated golden-dataset **RAGAS / Faithfulness & Multi-Hop Recall Eval Suite** in CI before promoting any new TBox version. |
