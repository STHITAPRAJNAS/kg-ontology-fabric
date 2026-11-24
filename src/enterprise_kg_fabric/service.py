from __future__ import annotations

import asyncio
import hashlib
import os
import re
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

from .adapters.graph_stores import (
    InMemoryOpenCypherGraphEmulator,
    Neo4jOpenCypherGraphAdapter,
    NeptuneOpenCypherGraphAdapter,
    PropertyGraphStorePort,
)
from .adapters.pgvector_store import (
    InMemoryPgVectorStoreEmulator,
    PgVectorStorePort,
    PostgresPgVectorStoreAdapter,
    validate_legacy_table_access,
)
from .adaptive_router import PluggableCrossEncoderReranker
from .auth import AuthEnforcer, get_current_principal
from .domain_packs import DomainRulePack, resolve_domain_pack
from .entity_resolution import CrossKBEntityResolver
from .eval_harness import GraphRAGEvaluationHarness
from .graph_analytics import GraphAnalyticsEngine, _is_acl_allowed
from .llama_index_bridge import LlamaIndexHierarchicalNodeParser, LlamaIndexSchemaAndDynamicExtractor
from .llm_extractor import StructuredLLMExtractorAdapter
from .models import (
    AnswerabilityCheckRequest,
    AnswerabilityCheckResponse,
    ChunkExtractionResult,
    ChunkSearchRequest,
    DLQFailedChunkRecord,
    EnterpriseCompiledTBox,
    EntitySearchMatch,
    EntitySearchRequest,
    EvalSuiteReport,
    EvalTestCase,
    GlobalCommunitySearchRequest,
    GlobalCommunitySearchResponse,
    GraphRAGQueryRequest,
    GraphRAGQueryResponse,
    KBSourceType,
    KBWatermarkState,
    KGGraphBuildReport,
    KGQualityScorecardReport,
    OntologyInductionReport,
    OntologyStatus,
    PPRSearchRequest,
    PPRSearchResponse,
    RawDocumentInput,
    ReadOnlyCypherSearchRequest,
    ReadOnlyCypherSearchResponse,
    RetrievedChunkContext,
    SubgraphSearchRequest,
    SubgraphSearchResponse,
    SyncJobRecord,
    SyncJobStatus,
    VectorChunkRecord,
    VectorKBRegistration,
    utc_now_iso,
)
from .multimodal_ingestor import MultimodalDocumentIngestor
from .ontology_inducer import AutomatedOntologyInducer, OWLSHACLCompiler
from .pii_redaction import PIIRedactionMode, PresidioPIIRedactor
from .quality_auditor import KGQualityAuditor
from .rag_engine import EnterpriseHybridGraphRAGEngine
from .vector_clustering import (
    EmbeddingProvider,
    FederatedEmbeddingModelRouter,
    SemanticFeatureHashEmbeddingProvider,
    cosine_similarity,
)

try:
    from opentelemetry import trace as _otel_trace

    _TRACER = _otel_trace.get_tracer("enterprise-kg-ontology-fabric")
except ImportError:
    _TRACER = None


def _otel_span(name: str):
    if _TRACER is not None:
        return _TRACER.start_as_current_span(name)
    return nullcontext()


class EnterpriseKGOntologyService:
    """
    Enterprise Orchestrator (`v0.2.0`) for Automated Ontology Induction & Knowledge Graph Construction
    across 100s of PostgreSQL `pgvector` Embedding Stores and Multimodal / Raw Document Streams:
      - Structured LLM Extraction (`google-genai` / Bedrock / OpenAI + `AsyncCircuitBreaker` + N-ary Qualifiers)
      - Persistent Multi-Pod TBox Store (`enterprise_tbox_registry`)
      - HippoRAG 2 Personalized PageRank (PPR) Hub-Damped Subgraph Retrieval
      - Microsoft GraphRAG Leiden/Louvain Hierarchical Community Detection & Global Sensemaking
      - Adaptive-RAG Query Routing + Pluggable Cross-Encoder Reranking
      - Document/Entity/Edge-Level ABAC Security Trimming (`acl_groups`)
      - Dead-Letter Queue (DLQ) + Async Sync Job Control Plane + OpenTelemetry Tracing
    """

    def __init__(
        self,
        vector_store: PgVectorStorePort | None = None,
        graph_store: PropertyGraphStorePort | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        model_router: FederatedEmbeddingModelRouter | None = None,
        pii_redactor: PresidioPIIRedactor | None = None,
        llm_extractor: StructuredLLMExtractorAdapter | None = None,
        reranker: PluggableCrossEncoderReranker | None = None,
        canonical_embedding_dim: int = 256,
        centroids_per_kb: int = 3,
        boundary_outliers_per_kb: int = 1,
        min_kb_support_for_approval: int = 2,
        min_chunk_support_for_approval: int = 2,
        use_live_llm: bool = False,
        domain_pack: DomainRulePack | str | None = None,
    ):
        self.canonical_dim = canonical_embedding_dim
        self.domain_pack = resolve_domain_pack(domain_pack)
        self.embedding_provider = embedding_provider or SemanticFeatureHashEmbeddingProvider(
            default_dim=canonical_embedding_dim
        )
        self.model_router = model_router or FederatedEmbeddingModelRouter(self.embedding_provider)
        self.pii_redactor = pii_redactor or PresidioPIIRedactor(mode=PIIRedactionMode.REPLACE_TAG)
        self.vector_store = vector_store or InMemoryPgVectorStoreEmulator(
            embedding_provider=self.embedding_provider,
            canonical_dim=canonical_embedding_dim,
            model_router=self.model_router,
        )
        self.graph_store = graph_store or InMemoryOpenCypherGraphEmulator()

        self.node_parser = LlamaIndexHierarchicalNodeParser(
            embedding_provider=self.embedding_provider,
            canonical_embedding_dim=canonical_embedding_dim,
        )
        self.ontology_inducer = AutomatedOntologyInducer(
            embedding_provider=self.embedding_provider,
            centroids_per_kb=centroids_per_kb,
            boundary_outliers_per_kb=boundary_outliers_per_kb,
            min_kb_support_for_approval=min_kb_support_for_approval,
            min_chunk_support_for_approval=min_chunk_support_for_approval,
            domain_pack=self.domain_pack,
        )
        self.kg_extractor = llm_extractor or StructuredLLMExtractorAdapter(
            use_live_llm=use_live_llm,
            fallback_extractor=LlamaIndexSchemaAndDynamicExtractor(),
        )
        self.entity_resolver = CrossKBEntityResolver(
            embedding_provider=self.embedding_provider,
            canonical_embedding_dim=canonical_embedding_dim,
        )
        self.graph_analytics = GraphAnalyticsEngine(
            embedding_provider=self.embedding_provider,
            canonical_dim=canonical_embedding_dim,
        )
        self.reranker = reranker or PluggableCrossEncoderReranker()
        self.rag_engine = EnterpriseHybridGraphRAGEngine(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            embedding_provider=self.embedding_provider,
            graph_analytics=self.graph_analytics,
            reranker=self.reranker,
            canonical_dim=canonical_embedding_dim,
        )
        self._tboxes: dict[str, EnterpriseCompiledTBox] = {}
        self._cumulative_sibling_collisions_prevented: dict[str, int] = {}
        self._pii_redaction_audit_counts: dict[str, int] = {}
        self._sync_jobs: dict[str, SyncJobRecord] = {}
        self._dlq_records: list[DLQFailedChunkRecord] = []
        self._sync_worker_task: asyncio.Task[Any] | None = None
        self._sync_worker_lock = asyncio.Lock()
        self._sync_worker_interval_sec: float = 300.0
        self._sync_worker_last_cycle: dict[str, Any] | None = None
        self._metrics: dict[str, int] = {
            "queries_total": 0,
            "builds_total": 0,
            "dlq_total": 0,
            "ppr_searches_total": 0,
            "global_searches_total": 0,
            "answerability_checks_total": 0,
            "quality_audits_total": 0,
        }
        self._last_quality_scores: dict[str, float] = {}
        self._principal_enforcer = AuthEnforcer()

    def _enforce_service_principal(
        self,
        tenant_id: str,
        requested_acls: list[str] | None = None,
        required_role: str | None = None,
    ) -> tuple[str, list[str] | None]:
        """Enforces contextvars `AuthPrincipal` inside the service layer (blocks SDK/stdio bypasses)."""
        principal = get_current_principal()
        if principal is None:
            return tenant_id, requested_acls
        if required_role:
            self._principal_enforcer.require_role(principal, required_role)
        safe_tenant = self._principal_enforcer.enforce_tenant(principal, tenant_id)
        safe_acls = self._principal_enforcer.enforce_acl_groups(principal, requested_acls)
        return safe_tenant, safe_acls

    @classmethod
    def from_env(cls) -> EnterpriseKGOntologyService:
        """
        Environment-driven factory for both PyPI CLI (`kg-fabric serve`) and Docker/EKS pods.
        Auto-wires PostgreSQL `pgvector`, Neo4j 5 or Amazon Neptune, Microsoft Presidio PII,
        optional live LLM extraction, and pluggable DomainRulePack from environment variables.
        """
        canonical_dim = int(os.getenv("KG_FABRIC_CANONICAL_DIM", "256"))
        embedding_provider = SemanticFeatureHashEmbeddingProvider(default_dim=canonical_dim)
        model_router = FederatedEmbeddingModelRouter(embedding_provider)

        pii_mode_raw = os.getenv("PII_REDACTION_MODE", "replace_tag").strip().lower()
        try:
            pii_mode = PIIRedactionMode(pii_mode_raw)
        except ValueError:
            pii_mode = PIIRedactionMode.REPLACE_TAG
        pii_redactor = PresidioPIIRedactor(
            mode=pii_mode,
            hmac_secret_key=os.getenv("PII_HMAC_SECRET_KEY") or None,
        )

        pg_dsn = (
            os.getenv("PGVECTOR_DSN")
            or os.getenv("KG_FABRIC_PGVECTOR_DSN")
            or os.getenv("DATABASE_URL")
            or ""
        ).strip()
        if not pg_dsn and os.getenv("PGHOST"):
            pg_user = os.getenv("PGUSER", "postgres")
            pg_pass = os.getenv("PGPASSWORD", "")
            pg_host = os.getenv("PGHOST", "127.0.0.1")
            pg_port = os.getenv("PGPORT", "5432")
            pg_db = os.getenv("PGDATABASE", "enterprise_kg_fabric")
            creds = f"{pg_user}:{pg_pass}@" if pg_pass else f"{pg_user}@"
            pg_dsn = f"postgresql://{creds}{pg_host}:{pg_port}/{pg_db}"
        if pg_dsn:
            vector_store: PgVectorStorePort = PostgresPgVectorStoreAdapter(
                dsn=pg_dsn,
                embedding_provider=embedding_provider,
                canonical_dim=canonical_dim,
            )
        else:
            vector_store = InMemoryPgVectorStoreEmulator(
                embedding_provider=embedding_provider,
                canonical_dim=canonical_dim,
                model_router=model_router,
            )

        graph_backend = os.getenv("GRAPH_BACKEND", "").strip().lower()
        neo4j_uri = os.getenv("NEO4J_URI", "").strip()
        neptune_endpoint = os.getenv("NEPTUNE_ENDPOINT", "").strip()
        if not graph_backend:
            if neptune_endpoint:
                graph_backend = "neptune"
            elif neo4j_uri:
                graph_backend = "neo4j"
            else:
                graph_backend = "memory"

        if graph_backend == "neptune" and neptune_endpoint:
            graph_store: PropertyGraphStorePort = NeptuneOpenCypherGraphAdapter(
                endpoint_url=neptune_endpoint,
                region_name=os.getenv("AWS_REGION", "us-east-1"),
            )
        elif graph_backend == "neo4j" and neo4j_uri:
            graph_store = Neo4jOpenCypherGraphAdapter(
                uri=neo4j_uri,
                username=os.getenv("NEO4J_USER", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD") or os.getenv("KG_FABRIC_NEO4J_PASS", ""),
            )
        else:
            graph_store = InMemoryOpenCypherGraphEmulator()

        use_live_llm = os.getenv("KG_FABRIC_USE_LIVE_LLM", "false").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        llm_provider = os.getenv("KG_FABRIC_LLM_PROVIDER", "gemini").strip().lower()
        llm_model = os.getenv("KG_FABRIC_LLM_MODEL", "gemini-2.5-flash").strip()
        llm_extractor = StructuredLLMExtractorAdapter(
            provider=llm_provider,
            model_name=llm_model,
            use_live_llm=use_live_llm,
            fallback_extractor=LlamaIndexSchemaAndDynamicExtractor(),
        )

        domain_pack_name = os.getenv("KG_FABRIC_DOMAIN_PACK", "cloud_sre").strip().lower()
        return cls(
            vector_store=vector_store,
            graph_store=graph_store,
            embedding_provider=embedding_provider,
            model_router=model_router,
            pii_redactor=pii_redactor,
            llm_extractor=llm_extractor,
            canonical_embedding_dim=canonical_dim,
            use_live_llm=use_live_llm,
            domain_pack=domain_pack_name,
        )

    async def close(self) -> None:
        """Gracefully stops background workers and closes database pools and graph drivers."""
        await self.stop_sync_worker()
        await self.vector_store.close()
        await self.graph_store.close()

    # -------------------------------------------------------------------------
    # Multi-Replica-Safe Built-In Incremental Sync Worker (pg_try_advisory_lock)
    # -------------------------------------------------------------------------

    @staticmethod
    def _sync_lock_key(tenant_id: str) -> int:
        digest = hashlib.sha256(f"kg_fabric_sync:{tenant_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:7], byteorder="big", signed=False)

    async def run_scheduled_sync_cycle(
        self, tenant_ids: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """
        Executes one multi-replica-safe incremental CDC sync pass across registered tenants.
        Uses both an in-process `asyncio.Lock` and a PostgreSQL non-blocking advisory lock
        (`pg_try_advisory_lock`) so autoscaled EKS replicas never race on the same batch.
        """
        async with self._sync_worker_lock:
            if tenant_ids is not None:
                target_tenants = list(tenant_ids)
            else:
                kbs = await self.vector_store.list_registered_kbs(tenant_id=None)
                target_tenants = sorted({kb.tenant_id for kb in kbs})

            synced: dict[str, Any] = {}
            skipped_locked: list[str] = []
            for tid in target_tenants:
                lock_key = self._sync_lock_key(tid)
                acquired = await self.vector_store.try_acquire_advisory_lock(lock_key)
                if not acquired:
                    skipped_locked.append(tid)
                    continue
                try:
                    tbox = await self.vector_store.load_tbox(tid)
                    if tbox is None and tid not in self._tboxes:
                        await self.induce_and_align_ontology(tenant_id=tid)
                    job = await self.execute_tracked_build_job(
                        tenant_id=tid,
                        incremental=True,
                        validate_shacl=True,
                    )
                    synced[tid] = {
                        "job_id": job.job_id,
                        "status": job.status.value,
                        "delta_chunks": job.report.delta_chunks_extracted if job.report else 0,
                        "dlq_count": job.dlq_count,
                    }
                finally:
                    await self.vector_store.release_advisory_lock(lock_key)

            summary = {
                "ran_at": utc_now_iso(),
                "tenants_synced": synced,
                "tenants_skipped_locked": skipped_locked,
            }
            self._sync_worker_last_cycle = summary
            return summary

    def start_sync_worker(
        self,
        interval_sec: float = 300.0,
        tenant_ids: Sequence[str] | None = None,
    ) -> None:
        if self._sync_worker_task is not None and not self._sync_worker_task.done():
            return
        self._sync_worker_interval_sec = max(5.0, float(interval_sec))

        async def _loop() -> None:
            while True:
                try:
                    await self.run_scheduled_sync_cycle(tenant_ids=tenant_ids)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._sync_worker_last_cycle = {
                        "ran_at": utc_now_iso(),
                        "error": str(exc),
                    }
                await asyncio.sleep(self._sync_worker_interval_sec)

        self._sync_worker_task = asyncio.create_task(_loop())

    async def stop_sync_worker(self) -> None:
        if self._sync_worker_task is not None:
            self._sync_worker_task.cancel()
            try:
                await self._sync_worker_task
            except asyncio.CancelledError:
                pass
            self._sync_worker_task = None

    def sync_worker_status(self) -> dict[str, Any]:
        running = self._sync_worker_task is not None and not self._sync_worker_task.done()
        return {
            "running": running,
            "interval_sec": self._sync_worker_interval_sec,
            "last_cycle": self._sync_worker_last_cycle,
        }

    async def register_vector_kb(self, registration: VectorKBRegistration) -> VectorKBRegistration:
        safe_tenant, _ = self._enforce_service_principal(
            registration.tenant_id, required_role="writer"
        )
        registration.tenant_id = safe_tenant

        # Critical #1: Block PostgreSQL system catalogs, internal KG Fabric tables, and unallowlisted
        # legacy tables at registration time. Physical legacy table registration requires platform_admin
        # when authenticated against PostgreSQL.
        if registration.source_type == KBSourceType.EXISTING_PGVECTOR_KB:
            cm = registration.column_mapping
            is_pg = isinstance(self.vector_store, PostgresPgVectorStoreAdapter)
            if is_pg and get_current_principal() is not None:
                self._principal_enforcer.require_platform_admin(get_current_principal())
            validate_legacy_table_access(
                schema_name=registration.schema_name,
                table_name=registration.table_name,
                column_names=[
                    cm.id_column,
                    cm.doc_id_column,
                    cm.text_column,
                    cm.embedding_column,
                    cm.metadata_column,
                ],
                allowed_patterns=getattr(self.vector_store, "allowed_legacy_tables", None),
                enforce_allowlist=bool(is_pg and get_current_principal() is not None),
            )

        await self.vector_store.register_kb(registration)
        return registration

    async def register_vector_kbs_batch(
        self, registrations: Sequence[VectorKBRegistration]
    ) -> int:
        for reg in registrations:
            await self.register_vector_kb(reg)
        return len(registrations)

    async def unregister_kb(self, tenant_id: str, kb_id: str) -> dict[str, Any]:
        safe_tenant, _ = self._enforce_service_principal(tenant_id, required_role="admin")
        deleted_chunk_ids = await self.vector_store.unregister_kb_chunks(safe_tenant, kb_id)
        deleted_ent_ids, deleted_edges = await self.graph_store.delete_chunks_provenance(
            tenant_id=safe_tenant, chunk_ids=deleted_chunk_ids, kb_id=kb_id
        )
        await self.vector_store.delete_canonical_entity_embeddings(safe_tenant, deleted_ent_ids)
        return {
            "tenant_id": safe_tenant,
            "kb_id": kb_id,
            "chunks_removed": len(deleted_chunk_ids),
            "graph_entities_removed": len(deleted_ent_ids),
            "graph_edges_removed": deleted_edges,
        }

    async def delete_document(self, tenant_id: str, kb_id: str, doc_id: str) -> dict[str, Any]:
        safe_tenant, _ = self._enforce_service_principal(tenant_id, required_role="writer")
        deleted_chunk_ids = await self.vector_store.delete_document_chunks(
            safe_tenant, kb_id, doc_id
        )
        deleted_ent_ids, deleted_edges = await self.graph_store.delete_chunks_provenance(
            tenant_id=safe_tenant, chunk_ids=deleted_chunk_ids
        )
        await self.vector_store.delete_canonical_entity_embeddings(safe_tenant, deleted_ent_ids)
        return {
            "tenant_id": safe_tenant,
            "kb_id": kb_id,
            "doc_id": doc_id,
            "chunks_removed": len(deleted_chunk_ids),
            "graph_entities_removed": len(deleted_ent_ids),
            "graph_edges_removed": deleted_edges,
        }

    async def purge_tenant(self, tenant_id: str) -> dict[str, Any]:
        """Complete GDPR / tenant offboarding purge across PostgreSQL/pgvector and Neo4j/Neptune."""
        safe_tenant, _ = self._enforce_service_principal(tenant_id, required_role="admin")
        vec_stats = await self.vector_store.purge_tenant(safe_tenant)
        graph_nodes_removed = await self.graph_store.purge_tenant(safe_tenant)
        self._tboxes.pop(safe_tenant, None)
        self._last_quality_scores.pop(safe_tenant, None)
        self._sync_jobs = {
            jid: j for jid, j in self._sync_jobs.items() if j.tenant_id != safe_tenant
        }
        self._dlq_records = [r for r in self._dlq_records if r.tenant_id != safe_tenant]
        return {
            "tenant_id": safe_tenant,
            "graph_nodes_removed": graph_nodes_removed,
            **vec_stats,
        }

    async def hydrate_existing_kb_chunks(
        self,
        kb_id: str,
        chunks: Sequence[VectorChunkRecord],
    ) -> int:
        scrubbed_chunks: list[VectorChunkRecord] = []
        for chk in chunks:
            safe_tenant, _ = self._enforce_service_principal(
                chk.tenant_id, required_role="writer"
            )
            chk.tenant_id = safe_tenant
            res = self.pii_redactor.analyze_and_redact(chk.text)
            chk.text = res.scrubbed_text
            if "acl_groups" not in chk.metadata:
                chk.metadata["acl_groups"] = list(chk.acl_groups)
            for k, v in res.redaction_counts.items():
                self._pii_redaction_audit_counts[k] = self._pii_redaction_audit_counts.get(k, 0) + v
            scrubbed_chunks.append(chk)
        return await self.vector_store.upsert_kb_chunks(kb_id, scrubbed_chunks)

    async def hydrate_from_registered_legacy_kbs(
        self,
        tenant_id: str = "enterprise_default",
        kb_ids: list[str] | None = None,
    ) -> int:
        """
        Incrementally reads explicitly allowlisted legacy PostgreSQL `pgvector` tables
        registered as `EXISTING_PGVECTOR_KB` via `reverse_hydrate_from_legacy_pgvector_table`,
        skipping rows whose `content_sha256` already matches `unified_rag_chunks`, scrubs PII,
        and upserts only genuinely new/changed delta rows.
        """
        if not isinstance(self.vector_store, PostgresPgVectorStoreAdapter):
            return 0
        if not self.vector_store.allowed_legacy_tables:
            return 0
        registered_kbs = await self.vector_store.list_registered_kbs(tenant_id=tenant_id)
        target_ids = set(kb_ids) if kb_ids else None
        hydrated_total = 0
        for kb in registered_kbs:
            if kb.source_type != KBSourceType.EXISTING_PGVECTOR_KB:
                continue
            if target_ids is not None and kb.kb_id not in target_ids:
                continue
            try:
                legacy_chunks = await self.vector_store.reverse_hydrate_from_legacy_pgvector_table(
                    kb, enforce_allowlist=True
                )
            except (PermissionError, Exception):
                continue
            if not legacy_chunks:
                continue
            existing = await self.vector_store.fetch_kb_chunks(kb.kb_id, tenant_id=tenant_id)
            existing_hashes = {c.chunk_id: c.content_sha256 for c in existing}
            delta_legacy = [
                chk
                for chk in legacy_chunks
                if existing_hashes.get(chk.chunk_id) != chk.content_sha256
            ]
            if delta_legacy:
                hydrated_total += await self.hydrate_existing_kb_chunks(kb.kb_id, delta_legacy)
        return hydrated_total

    async def ingest_raw_documents(
        self,
        documents: Sequence[RawDocumentInput],
        native_kb_dim: int = 768,
    ) -> list[VectorChunkRecord]:
        all_chunks: list[VectorChunkRecord] = []
        by_kb: dict[str, list[VectorChunkRecord]] = {}

        for doc in documents:
            safe_tenant, _ = self._enforce_service_principal(
                doc.tenant_id, required_role="writer"
            )
            doc.tenant_id = safe_tenant
            existing_kbs = {
                kb.kb_id for kb in await self.vector_store.list_registered_kbs(doc.tenant_id)
            }
            if doc.kb_id not in existing_kbs:
                await self.register_vector_kb(
                    VectorKBRegistration(
                        kb_id=doc.kb_id,
                        tenant_id=doc.tenant_id,
                        domain_tag=doc.domain_tag,
                        source_type=KBSourceType.MANAGED_RAW_DOC_KB,
                        table_name=f"pgvec_raw_{doc.kb_id}",
                        embedding_dim=native_kb_dim,
                    )
                )
            # Extract structured triples from any embedded Markdown tables automatically
            table_triples = MultimodalDocumentIngestor.extract_triples_from_markdown_tables(doc.content)
            if table_triples:
                doc.metadata = {
                    **doc.metadata,
                    "structured_triples": list(doc.metadata.get("structured_triples", [])) + table_triples,
                }
            parsed = self.node_parser.parse_document(doc, native_kb_dim=native_kb_dim)
            for chk in parsed:
                chk.metadata["acl_groups"] = list(chk.acl_groups)
            all_chunks.extend(parsed)
            by_kb.setdefault(doc.kb_id, []).extend(parsed)

        for kb_id, kb_chunks in by_kb.items():
            await self.vector_store.upsert_kb_chunks(kb_id, kb_chunks)

        return all_chunks

    async def ingest_multimodal_files(
        self,
        file_paths: Sequence[str | Path],
        kb_id: str,
        tenant_id: str = "enterprise_default",
        domain_tag: str = "general",
        acl_groups: list[str] | None = None,
        native_kb_dim: int = 768,
    ) -> list[VectorChunkRecord]:
        """Reuses IBM Docling / CSV / Markdown table parsers to ingest PDFs, DOCX, CSVs, and MD files."""
        docs = [
            MultimodalDocumentIngestor.parse_file_to_raw_document(
                file_path=fp,
                doc_id=f"file_{Path(fp).stem}",
                kb_id=kb_id,
                tenant_id=tenant_id,
                domain_tag=domain_tag,
                acl_groups=acl_groups,
            )
            for fp in file_paths
        ]
        return await self.ingest_raw_documents(docs, native_kb_dim=native_kb_dim)

    async def load_or_get_tbox(self, tenant_id: str = "enterprise_default") -> EnterpriseCompiledTBox:
        """
        Multi-pod safe TBox lookup: checks persistent `enterprise_tbox_registry` first so
        promotions or inductions performed by another worker/replica are visible immediately.
        """
        persisted = await self.vector_store.load_tbox(tenant_id)
        if persisted is not None:
            self._tboxes[tenant_id] = persisted
            return persisted
        if tenant_id in self._tboxes:
            return self._tboxes[tenant_id]
        return EnterpriseCompiledTBox(tenant_id=tenant_id)

    def get_tbox(self, tenant_id: str = "enterprise_default") -> EnterpriseCompiledTBox:
        if tenant_id not in self._tboxes:
            self._tboxes[tenant_id] = EnterpriseCompiledTBox(tenant_id=tenant_id)
        return self._tboxes[tenant_id]

    async def induce_and_align_ontology(
        self,
        tenant_id: str = "enterprise_default",
        tbox_version: str = "1.0.0",
        kb_ids: list[str] | None = None,
    ) -> tuple[EnterpriseCompiledTBox, OntologyInductionReport]:
        with _otel_span("kg_fabric.induce_and_align_ontology"):
            safe_tenant, _ = self._enforce_service_principal(tenant_id, required_role="writer")
            await self.hydrate_from_registered_legacy_kbs(tenant_id=safe_tenant, kb_ids=kb_ids)
            registered_kbs = await self.vector_store.list_registered_kbs(tenant_id=safe_tenant)
            target_kb_ids = set(kb_ids) if kb_ids else {kb.kb_id for kb in registered_kbs}

            chunks_by_kb: dict[str, list[VectorChunkRecord]] = {}
            for kb in registered_kbs:
                if kb.kb_id not in target_kb_ids:
                    continue
                chks = await self.vector_store.fetch_kb_chunks(
                    kb.kb_id, only_unindexed_or_changed=False, tenant_id=safe_tenant
                )
                if chks:
                    chunks_by_kb[kb.kb_id] = chks

            tbox, report = self.ontology_inducer.induce_enterprise_tbox(
                chunks_by_kb=chunks_by_kb,
                tenant_id=safe_tenant,
                tbox_version=tbox_version,
            )
            self._tboxes[safe_tenant] = tbox
            await self.vector_store.save_tbox(tbox)
            await self.graph_store.persist_ontology_tbox(tbox)
            return tbox, report

    async def build_or_update_knowledge_graph(
        self,
        tenant_id: str = "enterprise_default",
        incremental: bool = True,
        validate_shacl: bool = True,
        allow_dynamic_emergent: bool = True,
    ) -> KGGraphBuildReport:
        with _otel_span("kg_fabric.build_or_update_knowledge_graph"):
            safe_tenant, _ = self._enforce_service_principal(tenant_id, required_role="writer")
            await self.hydrate_from_registered_legacy_kbs(tenant_id=safe_tenant)
            tbox = await self.load_or_get_tbox(safe_tenant)
            if not tbox.classes:
                tbox, _ = await self.induce_and_align_ontology(tenant_id=safe_tenant)

            registered_kbs = await self.vector_store.list_registered_kbs(tenant_id=safe_tenant)
            total_chunks_inspected = 0
            delta_chunks_to_extract: list[VectorChunkRecord] = []

            for kb in registered_kbs:
                all_kb_chunks = await self.vector_store.fetch_kb_chunks(
                    kb.kb_id, only_unindexed_or_changed=False, tenant_id=safe_tenant
                )
                total_chunks_inspected += len(all_kb_chunks)
                if incremental:
                    delta = await self.vector_store.fetch_kb_chunks(
                        kb.kb_id,
                        only_unindexed_or_changed=True,
                        target_tbox_version=tbox.version,
                        tenant_id=safe_tenant,
                    )
                    delta_chunks_to_extract.extend(delta)
                else:
                    delta_chunks_to_extract.extend(all_kb_chunks)

            skipped_chunks = total_chunks_inspected - len(delta_chunks_to_extract)

            # Extract entities and relation triples with per-chunk DLQ fault isolation
            extractions: list[ChunkExtractionResult] = []
            succeeded_chunks: list[VectorChunkRecord] = []
            raw_entity_mentions = 0
            for chk in delta_chunks_to_extract:
                try:
                    res = self.kg_extractor.extract_from_chunk(
                        chk, tbox=tbox, allow_dynamic_emergent=allow_dynamic_emergent
                    )
                    raw_entity_mentions += len(res.entities) + (2 * len(res.triples))
                    extractions.append(res)
                    succeeded_chunks.append(chk)
                except Exception as exc:
                    dlq_rec = DLQFailedChunkRecord(
                        dlq_id=f"dlq_{uuid.uuid4().hex[:10]}",
                        tenant_id=safe_tenant,
                        kb_id=chk.kb_id,
                        chunk_id=chk.chunk_id,
                        error_message=str(exc),
                    )
                    self._dlq_records.append(dlq_rec)
                    await self.vector_store.append_dlq_record(dlq_rec)

            existing_entities = await self.graph_store.get_all_entities(safe_tenant)
            (
                canonical_entities,
                resolved_edges,
                cross_kb_merged_count,
                sibling_collisions_prevented,
            ) = self.entity_resolver.resolve_extractions(
                extractions=extractions,
                tbox=tbox,
                existing_entities=existing_entities,
            )

            self._cumulative_sibling_collisions_prevented[safe_tenant] = (
                self._cumulative_sibling_collisions_prevented.get(safe_tenant, 0)
                + sibling_collisions_prevented
            )

            shacl_ok = True
            if validate_shacl and canonical_entities:
                shacl_ok, _ = OWLSHACLCompiler.validate_abox_with_shacl(
                    tbox=tbox,
                    entities=list(canonical_entities.values()),
                    edges=resolved_edges,
                )

            superseded_closed = await self.graph_store.upsert_entities_and_bitemporal_edges(
                tenant_id=safe_tenant,
                entities=list(canonical_entities.values()),
                edges=resolved_edges,
            )
            await self.vector_store.upsert_canonical_entity_embeddings(
                list(canonical_entities.values())
            )

            watermarks = [
                KBWatermarkState(
                    kb_id=chk.kb_id,
                    chunk_id=chk.chunk_id,
                    content_sha256=chk.content_sha256,
                    tbox_version=tbox.version,
                )
                for chk in succeeded_chunks
            ]
            if watermarks:
                await self.vector_store.mark_chunks_watermarked(watermarks)

            active_edges = await self.graph_store.get_all_edges(safe_tenant, active_only=True)

            return KGGraphBuildReport(
                tenant_id=safe_tenant,
                tbox_version=tbox.version,
                kbs_processed=len(registered_kbs),
                total_chunks_inspected=total_chunks_inspected,
                delta_chunks_extracted=len(succeeded_chunks),
                skipped_unchanged_chunks_via_watermark=skipped_chunks,
                raw_entity_mentions=raw_entity_mentions,
                canonical_entities_after_er=len(canonical_entities),
                cross_kb_merged_entities=cross_kb_merged_count,
                sibling_collisions_prevented=self._cumulative_sibling_collisions_prevented.get(
                    safe_tenant, 0
                ),
                active_edges_upserted=len(active_edges),
                superseded_bitemporal_edges_closed=superseded_closed,
                shacl_conformance_passed=shacl_ok,
            )

    # -------------------------------------------------------------------------
    # Async Sync Job Control Plane & Dead-Letter Queue (DLQ)
    # -------------------------------------------------------------------------

    async def execute_tracked_build_job(
        self,
        tenant_id: str = "enterprise_default",
        incremental: bool = True,
        validate_shacl: bool = True,
    ) -> SyncJobRecord:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = SyncJobRecord(
            job_id=job_id,
            tenant_id=tenant_id,
            status=SyncJobStatus.RUNNING,
            incremental=incremental,
        )
        self._sync_jobs[job_id] = job
        await self.vector_store.save_sync_job(job)
        dlq_before = len(self._dlq_records)
        try:
            report = await self.build_or_update_knowledge_graph(
                tenant_id=tenant_id,
                incremental=incremental,
                validate_shacl=validate_shacl,
            )
            job.report = report
            job.dlq_count = len(self._dlq_records) - dlq_before
            job.status = SyncJobStatus.COMPLETED
            job.completed_at = utc_now_iso()
        except Exception as exc:
            job.status = SyncJobStatus.FAILED
            job.error = str(exc)
            job.completed_at = utc_now_iso()
        await self.vector_store.save_sync_job(job)
        return job

    def get_sync_job(self, job_id: str) -> SyncJobRecord | None:
        return self._sync_jobs.get(job_id)

    async def get_sync_job_async(self, job_id: str) -> SyncJobRecord | None:
        if job_id in self._sync_jobs:
            return self._sync_jobs[job_id]
        persisted = await self.vector_store.get_sync_job(job_id)
        if persisted is not None:
            self._sync_jobs[job_id] = persisted
        return persisted

    def list_dlq_records(self, tenant_id: str | None = None) -> list[DLQFailedChunkRecord]:
        if tenant_id is None:
            return list(self._dlq_records)
        return [r for r in self._dlq_records if r.tenant_id == tenant_id]

    async def list_dlq_records_async(
        self, tenant_id: str | None = None
    ) -> list[DLQFailedChunkRecord]:
        persisted = await self.vector_store.list_dlq_records(tenant_id=tenant_id)
        if persisted:
            return persisted
        return self.list_dlq_records(tenant_id=tenant_id)

    async def promote_quarantined_ontology_concepts(
        self,
        tenant_id: str = "enterprise_default",
        min_confidence: float = 0.80,
    ) -> dict[str, Any]:
        safe_tenant, _ = self._enforce_service_principal(tenant_id, required_role="admin")
        tbox = await self.load_or_get_tbox(safe_tenant)
        promoted_classes: list[str] = []
        promoted_relations: list[str] = []

        for cls_name, cls_def in tbox.classes.items():
            if (
                cls_def.status == OntologyStatus.QUARANTINED_CANDIDATE
                and cls_def.confidence >= min_confidence
            ):
                cls_def.status = OntologyStatus.APPROVED
                promoted_classes.append(cls_name)

        for pred_name, rel_def in tbox.relations.items():
            if (
                rel_def.status == OntologyStatus.QUARANTINED_CANDIDATE
                and rel_def.confidence >= min_confidence
            ):
                rel_def.status = OntologyStatus.APPROVED
                promoted_relations.append(pred_name)

        if promoted_classes or promoted_relations:
            parts = tbox.version.split(".")
            if len(parts) == 3 and parts[1].isdigit():
                tbox.version = f"{parts[0]}.{int(parts[1]) + 1}.0"
            self._tboxes[safe_tenant] = tbox
            await self.vector_store.save_tbox(tbox)
            await self.graph_store.persist_ontology_tbox(tbox)

        entities = await self.graph_store.get_all_entities(safe_tenant)
        edges = await self.graph_store.get_all_edges(safe_tenant, active_only=False)
        promoted_entities_list = []
        promoted_edges_list = []

        for ent in entities.values():
            cls_def = tbox.classes.get(ent.ontology_class)
            if cls_def and cls_def.status == OntologyStatus.APPROVED:
                if ent.governance_bucket != OntologyStatus.APPROVED:
                    ent.governance_bucket = OntologyStatus.APPROVED
                    promoted_entities_list.append(ent)

        for edge in edges:
            rel_def = tbox.relations.get(edge.predicate)
            s_ent = entities.get(edge.source_entity_id)
            t_ent = entities.get(edge.target_entity_id)
            if (
                rel_def
                and rel_def.status == OntologyStatus.APPROVED
                and s_ent
                and s_ent.governance_bucket == OntologyStatus.APPROVED
                and t_ent
                and t_ent.governance_bucket == OntologyStatus.APPROVED
            ):
                if edge.governance_bucket != OntologyStatus.APPROVED:
                    edge.governance_bucket = OntologyStatus.APPROVED
                    promoted_edges_list.append(edge)

        # Persist promoted entity/edge governance buckets back to Neo4j / Neptune!
        if promoted_entities_list or promoted_edges_list:
            await self.graph_store.upsert_entities_and_bitemporal_edges(
                tenant_id=safe_tenant,
                entities=promoted_entities_list,
                edges=promoted_edges_list,
            )

        return {
            "tenant_id": safe_tenant,
            "new_tbox_version": tbox.version,
            "promoted_classes": promoted_classes,
            "promoted_relations": promoted_relations,
            "promoted_entities": len(promoted_entities_list),
            "promoted_edges": len(promoted_edges_list),
        }

    def export_ontology_turtle(self, tenant_id: str = "enterprise_default") -> str:
        tbox = self.get_tbox(tenant_id)
        return OWLSHACLCompiler.export_turtle(tbox)

    async def query_graph_rag(self, request: GraphRAGQueryRequest) -> GraphRAGQueryResponse:
        with _otel_span("kg_fabric.query_graph_rag"):
            safe_tenant, safe_acls = self._enforce_service_principal(
                request.tenant_id, request.caller_acl_groups
            )
            scoped_req = request.model_copy(
                update={"tenant_id": safe_tenant, "caller_acl_groups": safe_acls}
            )
            self._metrics["queries_total"] += 1
            tbox = await self.load_or_get_tbox(safe_tenant)
            return await self.rag_engine.query(request=scoped_req, tbox=tbox)

    async def check_answerable(
        self, request: AnswerabilityCheckRequest
    ) -> AnswerabilityCheckResponse:
        """
        Pre-flight Ontology & Multi-KB Graph Coverage Check (`ANSWERABLE` | `PARTIAL` | `UNANSWERABLE`).
        Prevents hallucinated answers when requested concepts or entities are missing or
        outside the caller's ABAC `acl_groups` scope.
        """
        safe_tenant, safe_acls = self._enforce_service_principal(
            request.tenant_id, request.caller_acl_groups
        )
        self._metrics["answerability_checks_total"] += 1
        tbox = await self.load_or_get_tbox(safe_tenant)
        all_entities = await self.graph_store.get_all_entities(safe_tenant)

        q_lower = request.query.lower()
        stop_words = {
            "what", "which", "where", "when", "who", "how", "does", "are", "the",
            "and", "for", "with", "from", "that", "this", "into", "about", "across",
            "depend", "depends", "connected", "impact", "blast", "radius", "show",
            "list", "find", "tell", "give", "have", "been", "being", "their",
        }
        tokens = [
            tok for tok in re.findall(r"[a-z0-9][a-z0-9_\-\.]{2,}", q_lower)
            if tok not in stop_words
        ]

        matched_classes: list[str] = []
        for cls_name, cls_def in tbox.classes.items():
            terms = [cls_name.lower()] + [s.lower() for s in cls_def.synonyms]
            if any(t in q_lower for t in terms if len(t) >= 3):
                matched_classes.append(cls_name)

        matched_entities: list[str] = []
        covered_tokens: set[str] = set()
        for ent in all_entities.values():
            if ent.governance_bucket != OntologyStatus.APPROVED:
                continue
            if not _is_acl_allowed(ent.acl_groups, safe_acls):
                continue
            names = [ent.canonical_name.lower()] + [a.lower() for a in ent.aliases]
            if any(n in q_lower for n in names if len(n) >= 3):
                matched_entities.append(ent.canonical_name)
                for tok in tokens:
                    if any(tok in n or n in tok for n in names):
                        covered_tokens.add(tok)

        q_vec = self.embedding_provider.embed_text(request.query, dim=self.canonical_dim)
        chunk_hits = await self.vector_store.hybrid_search_chunks(
            tenant_id=safe_tenant,
            query_text=request.query,
            query_embedding=q_vec,
            top_k=5,
            caller_acl_groups=safe_acls,
        )
        for chk in chunk_hits:
            chk_l = chk.text.lower()
            for tok in tokens:
                if tok in chk_l:
                    covered_tokens.add(tok)

        for cls_name in matched_classes:
            cls_l = cls_name.lower()
            for tok in tokens:
                if tok in cls_l or cls_l in tok:
                    covered_tokens.add(tok)

        unmatched = [t for t in dict.fromkeys(tokens) if t not in covered_tokens]
        if tokens:
            coverage = round(len(covered_tokens) / len(set(tokens)), 3)
        else:
            coverage = 1.0 if (matched_entities or chunk_hits) else 0.0

        if (matched_entities or covered_tokens) and coverage >= 0.60:
            status = "ANSWERABLE"
            answerable = True
            hint = None
        elif matched_entities or matched_classes or covered_tokens:
            status = "PARTIAL"
            answerable = False
            hint = (
                f"Partial graph/KB coverage ({coverage:.0%}). Missing or restricted concepts: "
                f"{', '.join(unmatched[:5]) or 'none'}."
            )
        else:
            status = "UNANSWERABLE"
            answerable = False
            hint = (
                "No authorized KB chunks, entities, or ontology classes cover this query. "
                "Verify KB ingestion, ontology induction, or caller ACL groups."
            )

        return AnswerabilityCheckResponse(
            query=request.query,
            tenant_id=safe_tenant,
            answerable=answerable,
            status=status,
            coverage_score=coverage,
            matched_tbox_classes=sorted(set(matched_classes)),
            matched_entities=sorted(set(matched_entities)),
            unmatched_terms=unmatched[:10],
            evidence_counts={
                "matched_entities": len(set(matched_entities)),
                "matched_tbox_classes": len(set(matched_classes)),
                "accessible_chunks": len(chunk_hits),
            },
            clarification_hint=hint,
        )

    # -------------------------------------------------------------------------
    # Granular Agentic Search Primitives + HippoRAG PPR + Leiden Global Search
    # -------------------------------------------------------------------------

    async def search_vector_chunks(
        self, request: ChunkSearchRequest
    ) -> list[RetrievedChunkContext]:
        safe_tenant, safe_acls = self._enforce_service_principal(
            request.tenant_id, request.caller_acl_groups
        )
        q_vec = self.embedding_provider.embed_text(request.query, dim=self.canonical_dim)
        chunks = await self.vector_store.hybrid_search_chunks(
            tenant_id=safe_tenant,
            query_text=request.query,
            query_embedding=q_vec,
            top_k=request.top_k,
            kb_filter=request.kb_filter,
            domain_filter=request.domain_filter,
            caller_acl_groups=safe_acls,
        )
        return self.reranker.rerank_chunks(request.query, chunks, top_k=request.top_k)

    async def search_kg_entities(
        self, request: EntitySearchRequest
    ) -> list[EntitySearchMatch]:
        safe_tenant, safe_acls = self._enforce_service_principal(
            request.tenant_id, request.caller_acl_groups
        )
        all_entities = await self.graph_store.get_all_entities(safe_tenant)
        q_lower = request.query.strip().lower()
        q_vec = self.embedding_provider.embed_text(request.query, dim=self.canonical_dim)

        matches: list[EntitySearchMatch] = []
        for ent in all_entities.values():
            if not request.include_quarantined and ent.governance_bucket != OntologyStatus.APPROVED:
                continue
            if not _is_acl_allowed(ent.acl_groups, safe_acls):
                continue
            if (
                request.ontology_class_filter
                and ent.ontology_class.lower() != request.ontology_class_filter.lower()
            ):
                continue

            names = [ent.canonical_name] + list(ent.aliases)
            lexical_boost = 0.0
            for n in names:
                nl = n.lower()
                if nl == q_lower:
                    lexical_boost = max(lexical_boost, 1.0)
                elif q_lower in nl or nl in q_lower:
                    lexical_boost = max(lexical_boost, 0.75)

            ent_vec = ent.embedding or self.embedding_provider.embed_text(
                f"{ent.ontology_class} {ent.canonical_name}", dim=self.canonical_dim
            )
            vec_sim = max(0.0, cosine_similarity(q_vec, ent_vec))
            score = round(0.65 * lexical_boost + 0.35 * vec_sim, 4) if lexical_boost > 0 else round(vec_sim, 4)

            if score >= 0.15 or lexical_boost > 0:
                matches.append(
                    EntitySearchMatch(
                        entity_id=ent.entity_id,
                        canonical_name=ent.canonical_name,
                        ontology_class=ent.ontology_class,
                        upper_category=ent.upper_category,
                        aliases=ent.aliases,
                        source_kb_ids=ent.source_kb_ids,
                        acl_groups=ent.acl_groups,
                        is_cross_kb_bridge=(len(ent.source_kb_ids) >= 2),
                        match_score=score,
                    )
                )

        matches.sort(key=lambda m: (-m.match_score, -len(m.source_kb_ids)))
        return matches[: request.top_k]

    async def traverse_entity_subgraph(
        self, request: SubgraphSearchRequest
    ) -> SubgraphSearchResponse:
        safe_tenant, safe_acls = self._enforce_service_principal(
            request.tenant_id, request.caller_acl_groups
        )
        all_entities = await self.graph_store.get_all_entities(safe_tenant)
        resolved_ids = self._resolve_seed_names_to_ids(all_entities, request.seed_entities)

        facts, bridges = await self.graph_store.expand_k_hop_subgraph(
            tenant_id=safe_tenant,
            seed_entity_ids=resolved_ids,
            max_hops=request.max_hops,
            include_quarantined=request.include_quarantined,
            as_of_iso=request.as_of_iso,
            caller_acl_groups=safe_acls,
        )
        if request.predicate_filter:
            allowed_preds = {p.upper() for p in request.predicate_filter}
            facts = [f for f in facts if f.predicate.upper() in allowed_preds]

        return SubgraphSearchResponse(
            tenant_id=safe_tenant,
            resolved_seed_entity_ids=resolved_ids,
            cross_kb_bridge_entities=bridges,
            facts=facts,
        )

    async def search_ppr_subgraph(self, request: PPRSearchRequest) -> PPRSearchResponse:
        """HippoRAG 2 Personalized PageRank hub-damped subgraph retrieval."""
        safe_tenant, safe_acls = self._enforce_service_principal(
            request.tenant_id, request.caller_acl_groups
        )
        self._metrics["ppr_searches_total"] += 1
        all_entities = await self.graph_store.get_all_entities(safe_tenant)
        all_edges = await self.graph_store.get_all_edges(
            safe_tenant, active_only=(request.as_of_iso is None)
        )
        resolved_ids = self._resolve_seed_names_to_ids(all_entities, request.seed_entities)
        return self.graph_analytics.search_ppr_subgraph(
            tenant_id=safe_tenant,
            entities=all_entities,
            edges=all_edges,
            seed_entity_ids=resolved_ids,
            top_k_facts=request.top_k_facts,
            alpha=request.alpha,
            as_of_iso=request.as_of_iso,
            caller_acl_groups=safe_acls,
        )

    async def query_global_communities(
        self, request: GlobalCommunitySearchRequest
    ) -> GlobalCommunitySearchResponse:
        """Microsoft GraphRAG Leiden/Louvain Hierarchical Community Global Sensemaking."""
        safe_tenant, safe_acls = self._enforce_service_principal(
            request.tenant_id, request.caller_acl_groups
        )
        self._metrics["global_searches_total"] += 1
        all_entities = await self.graph_store.get_all_entities(safe_tenant)
        all_edges = await self.graph_store.get_all_edges(safe_tenant, active_only=True)
        return self.graph_analytics.query_global_communities(
            tenant_id=safe_tenant,
            query=request.query,
            entities=all_entities,
            edges=all_edges,
            top_k_communities=request.top_k_communities,
            caller_acl_groups=safe_acls,
        )

    async def run_eval_suite(
        self, tenant_id: str, cases: Sequence[EvalTestCase]
    ) -> EvalSuiteReport:
        safe_tenant, _ = self._enforce_service_principal(tenant_id)
        return await GraphRAGEvaluationHarness.evaluate_suite(
            service=self, tenant_id=safe_tenant, cases=cases
        )

    async def audit_kg_quality(
        self,
        tenant_id: str = "enterprise_default",
        sample_size_faithfulness: int = 25,
        use_llm_judge: bool = False,
    ) -> KGQualityScorecardReport:
        """
        Runs the automated 4-dimension Ontology (TBox) & Knowledge Graph (ABox) Quality Scorecard
        (`0-100` grade + actionable findings across SHACL, utilization, extraction faithfulness,
        entity resolution duplicates/over-merges, orphan nodes, GCC, cross-KB linkage, modularity).
        """
        safe_tenant, _ = self._enforce_service_principal(tenant_id)
        self._metrics["quality_audits_total"] += 1
        report = await KGQualityAuditor.audit_tenant(
            service=self,
            tenant_id=safe_tenant,
            sample_size_faithfulness=sample_size_faithfulness,
            use_llm_judge=use_llm_judge,
        )
        self._last_quality_scores[safe_tenant] = report.overall_score
        return report

    @staticmethod
    def _resolve_seed_names_to_ids(
        all_entities: dict[str, Any], seed_entities: Sequence[str]
    ) -> list[str]:
        resolved_ids: list[str] = []
        for seed in seed_entities:
            s_clean = seed.strip()
            if s_clean in all_entities:
                if s_clean not in resolved_ids:
                    resolved_ids.append(s_clean)
                continue
            s_lower = s_clean.lower()
            for eid, ent in all_entities.items():
                names = [ent.canonical_name.lower()] + [a.lower() for a in ent.aliases]
                if s_lower in names or any(s_lower in n for n in names):
                    if eid not in resolved_ids:
                        resolved_ids.append(eid)
        return resolved_ids

    @staticmethod
    def validate_readonly_opencypher(cypher_query: str) -> None:
        import re

        forbidden = re.compile(
            r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s+dbms)\b",
            re.I,
        )
        m = forbidden.search(cypher_query)
        if m:
            raise ValueError(
                f"Mutating OpenCypher keyword '{m.group(1)}' is forbidden on read-only search endpoint."
            )

    def get_metrics_snapshot(self, tenant_id: str | None = None) -> dict[str, Any]:
        if tenant_id is not None:
            scores = (
                {tenant_id: self._last_quality_scores[tenant_id]}
                if tenant_id in self._last_quality_scores
                else {}
            )
        else:
            scores = dict(self._last_quality_scores)
        return {
            **self._metrics,
            "sync_jobs_cached": len(self._sync_jobs),
            "dlq_records_cached": len(self._dlq_records),
            "sync_worker": self.sync_worker_status(),
            "last_quality_scores": scores,
            "pii_redaction_audit_counts": dict(self._pii_redaction_audit_counts),
        }

    def get_prometheus_metrics_text(self, tenant_id: str | None = None) -> str:
        snap = self.get_metrics_snapshot(tenant_id=tenant_id)
        lines = [
            "# HELP kg_fabric_queries_total Total GraphRAG queries processed",
            "# TYPE kg_fabric_queries_total counter",
            f"kg_fabric_queries_total {snap['queries_total']}",
            "# HELP kg_fabric_ppr_searches_total Total HippoRAG PPR subgraph searches",
            "# TYPE kg_fabric_ppr_searches_total counter",
            f"kg_fabric_ppr_searches_total {snap['ppr_searches_total']}",
            "# HELP kg_fabric_global_searches_total Total Leiden global community searches",
            "# TYPE kg_fabric_global_searches_total counter",
            f"kg_fabric_global_searches_total {snap['global_searches_total']}",
            "# HELP kg_fabric_answerability_checks_total Total pre-flight answerability checks",
            "# TYPE kg_fabric_answerability_checks_total counter",
            f"kg_fabric_answerability_checks_total {snap['answerability_checks_total']}",
            "# HELP kg_fabric_quality_audits_total Total KG quality scorecard audits run",
            "# TYPE kg_fabric_quality_audits_total counter",
            f"kg_fabric_quality_audits_total {snap['quality_audits_total']}",
            "# HELP kg_fabric_dlq_records_cached Current DLQ failed chunk count",
            "# TYPE kg_fabric_dlq_records_cached gauge",
            f"kg_fabric_dlq_records_cached {snap['dlq_records_cached']}",
            "# HELP kg_fabric_sync_worker_running Whether background incremental sync worker is active",
            "# TYPE kg_fabric_sync_worker_running gauge",
            f"kg_fabric_sync_worker_running {1 if snap['sync_worker']['running'] else 0}",
        ]
        if snap["last_quality_scores"]:
            lines.append("# HELP kg_fabric_quality_score Latest 0-100 KG & Ontology quality scorecard by tenant")
            lines.append("# TYPE kg_fabric_quality_score gauge")
            for tid, score in sorted(snap["last_quality_scores"].items()):
                safe_tid = tid.replace('"', '\\"')
                lines.append(f'kg_fabric_quality_score{{tenant_id="{safe_tid}"}} {score}')
        return "\n".join(lines) + "\n"

    async def execute_readonly_opencypher(
        self, request: ReadOnlyCypherSearchRequest
    ) -> ReadOnlyCypherSearchResponse:
        # High #2: Raw Cypher over a shared graph DB can omit `$tenant_id` or read other tenants'
        # nodes. Require global `platform_admin` when authenticated (blocks tenant-bound admins too!),
        # reject literal tenant_id overrides, and run inside Neo4j READ_ACCESS sessions.
        self._principal_enforcer.require_platform_admin(get_current_principal())
        safe_tenant, _ = self._enforce_service_principal(
            request.tenant_id, required_role="admin"
        )
        self.validate_readonly_opencypher(request.cypher_query)
        if re.search(r"tenant_id\s*[:=]\s*['\"]", request.cypher_query, re.I):
            raise ValueError(
                "Hardcoded literal tenant_id is forbidden in OpenCypher; bind $tenant_id parameter instead."
            )
        params = {**request.params, "tenant_id": safe_tenant}
        rows = await self.graph_store.execute_opencypher(
            tenant_id=safe_tenant,
            cypher_query=request.cypher_query,
            params=params,
        )
        return ReadOnlyCypherSearchResponse(
            tenant_id=safe_tenant,
            cypher_query=request.cypher_query,
            rows=rows,
            row_count=len(rows),
        )

    async def get_enterprise_telemetry(
        self, tenant_id: str = "enterprise_default"
    ) -> dict[str, Any]:
        kbs = await self.vector_store.list_registered_kbs(tenant_id=tenant_id)
        entities = await self.graph_store.get_all_entities(tenant_id)
        active_edges = await self.graph_store.get_all_edges(tenant_id, active_only=True)
        all_edges = await self.graph_store.get_all_edges(tenant_id, active_only=False)
        tbox = await self.load_or_get_tbox(tenant_id)

        domains: dict[str, int] = {}
        dims: set[int] = set()
        for kb in kbs:
            domains[kb.domain_tag] = domains.get(kb.domain_tag, 0) + 1
            dims.add(kb.embedding_dim)

        cross_kb_entities = [
            e.canonical_name for e in entities.values() if len(e.source_kb_ids) >= 2
        ]

        return {
            "tenant_id": tenant_id,
            "registered_kbs_count": len(kbs),
            "kb_domains_distribution": domains,
            "heterogeneous_embedding_dims_supported": sorted(dims),
            "tbox_version": tbox.version,
            "approved_classes_count": sum(
                1 for c in tbox.classes.values() if c.status == OntologyStatus.APPROVED
            ),
            "quarantined_classes_count": sum(
                1 for c in tbox.classes.values() if c.status == OntologyStatus.QUARANTINED_CANDIDATE
            ),
            "approved_relations_count": sum(
                1 for r in tbox.relations.values() if r.status == OntologyStatus.APPROVED
            ),
            "canonical_entities_count": len(entities),
            "cross_kb_bridge_entities_count": len(cross_kb_entities),
            "active_graph_edges_count": len(active_edges),
            "historical_superseded_edges_count": len(all_edges) - len(active_edges),
            "pii_engine_used": self.pii_redactor.engine_name,
            "pii_redaction_audit_counts": dict(self._pii_redaction_audit_counts),
            "llm_extractor_backend": self.kg_extractor.active_backend,
            "llm_circuit_breaker_state": self.kg_extractor.circuit_breaker.state,
            "reranker_engine": self.reranker.engine_name,
            "docling_multimodal_available": MultimodalDocumentIngestor.docling_available(),
            "dlq_failed_chunks_count": len(self.list_dlq_records(tenant_id)),
            "sync_jobs_tracked": len(self._sync_jobs),
        }
