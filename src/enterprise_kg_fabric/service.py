from __future__ import annotations

from typing import Any, Sequence

from .adapters.graph_stores import InMemoryOpenCypherGraphEmulator, PropertyGraphStorePort
from .adapters.pgvector_store import InMemoryPgVectorStoreEmulator, PgVectorStorePort
from .entity_resolution import CrossKBEntityResolver
from .llama_index_bridge import (
    LlamaIndexHierarchicalNodeParser,
    LlamaIndexSchemaAndDynamicExtractor,
    PIIRedactor,
)
from .models import (
    ChunkExtractionResult,
    ChunkSearchRequest,
    EnterpriseCompiledTBox,
    EntitySearchMatch,
    EntitySearchRequest,
    GraphRAGQueryRequest,
    GraphRAGQueryResponse,
    KBSourceType,
    KBWatermarkState,
    KGGraphBuildReport,
    OntologyInductionReport,
    OntologyStatus,
    RawDocumentInput,
    ReadOnlyCypherSearchRequest,
    ReadOnlyCypherSearchResponse,
    RetrievedChunkContext,
    SubgraphSearchRequest,
    SubgraphSearchResponse,
    VectorChunkRecord,
    VectorKBRegistration,
)
from .ontology_inducer import AutomatedOntologyInducer, OWLSHACLCompiler
from .pii_redaction import PIIRedactionMode, PresidioPIIRedactor
from .rag_engine import EnterpriseHybridGraphRAGEngine
from .vector_clustering import (
    EmbeddingProvider,
    FederatedEmbeddingModelRouter,
    SemanticFeatureHashEmbeddingProvider,
    cosine_similarity,
)


class EnterpriseKGOntologyService:
    """
    Enterprise Orchestrator for Automated Ontology Induction & Knowledge Graph Construction
    across 100s of PostgreSQL `pgvector` Embedding Stores and Raw Document Streams,
    backed by LlamaIndex, Neo4j / Amazon Neptune, and PostgreSQL `pgvector`.
    """

    def __init__(
        self,
        vector_store: PgVectorStorePort | None = None,
        graph_store: PropertyGraphStorePort | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        model_router: FederatedEmbeddingModelRouter | None = None,
        pii_redactor: PresidioPIIRedactor | None = None,
        canonical_embedding_dim: int = 256,
        centroids_per_kb: int = 3,
        boundary_outliers_per_kb: int = 1,
        min_kb_support_for_approval: int = 2,
        min_chunk_support_for_approval: int = 2,
    ):
        self.canonical_dim = canonical_embedding_dim
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
        )
        self.kg_extractor = LlamaIndexSchemaAndDynamicExtractor()
        self.entity_resolver = CrossKBEntityResolver(
            embedding_provider=self.embedding_provider,
            canonical_embedding_dim=canonical_embedding_dim,
        )
        self.rag_engine = EnterpriseHybridGraphRAGEngine(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            embedding_provider=self.embedding_provider,
            canonical_dim=canonical_embedding_dim,
        )
        self._tboxes: dict[str, EnterpriseCompiledTBox] = {}
        self._cumulative_sibling_collisions_prevented: dict[str, int] = {}
        self._pii_redaction_audit_counts: dict[str, int] = {}

    async def register_vector_kb(self, registration: VectorKBRegistration) -> VectorKBRegistration:
        await self.vector_store.register_kb(registration)
        return registration

    async def register_vector_kbs_batch(
        self, registrations: Sequence[VectorKBRegistration]
    ) -> int:
        for reg in registrations:
            await self.vector_store.register_kb(reg)
        return len(registrations)

    async def hydrate_existing_kb_chunks(
        self,
        kb_id: str,
        chunks: Sequence[VectorChunkRecord],
    ) -> int:
        scrubbed_chunks: list[VectorChunkRecord] = []
        for chk in chunks:
            res = self.pii_redactor.analyze_and_redact(chk.text)
            chk.text = res.scrubbed_text
            for k, v in res.redaction_counts.items():
                self._pii_redaction_audit_counts[k] = self._pii_redaction_audit_counts.get(k, 0) + v
            scrubbed_chunks.append(chk)
        return await self.vector_store.upsert_kb_chunks(kb_id, scrubbed_chunks)

    async def ingest_raw_documents(
        self,
        documents: Sequence[RawDocumentInput],
        native_kb_dim: int = 768,
    ) -> list[VectorChunkRecord]:
        all_chunks: list[VectorChunkRecord] = []
        by_kb: dict[str, list[VectorChunkRecord]] = {}

        for doc in documents:
            # Ensure managed raw doc KB is registered
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
            parsed = self.node_parser.parse_document(doc, native_kb_dim=native_kb_dim)
            all_chunks.extend(parsed)
            by_kb.setdefault(doc.kb_id, []).extend(parsed)

        for kb_id, kb_chunks in by_kb.items():
            await self.vector_store.upsert_kb_chunks(kb_id, kb_chunks)

        return all_chunks

    async def induce_and_align_ontology(
        self,
        tenant_id: str = "enterprise_default",
        tbox_version: str = "1.0.0",
        kb_ids: list[str] | None = None,
    ) -> tuple[EnterpriseCompiledTBox, OntologyInductionReport]:
        registered_kbs = await self.vector_store.list_registered_kbs(tenant_id=tenant_id)
        target_kb_ids = set(kb_ids) if kb_ids else {kb.kb_id for kb in registered_kbs}

        chunks_by_kb: dict[str, list[VectorChunkRecord]] = {}
        for kb in registered_kbs:
            if kb.kb_id not in target_kb_ids:
                continue
            chks = await self.vector_store.fetch_kb_chunks(
                kb.kb_id, only_unindexed_or_changed=False
            )
            if chks:
                chunks_by_kb[kb.kb_id] = chks

        tbox, report = self.ontology_inducer.induce_enterprise_tbox(
            chunks_by_kb=chunks_by_kb,
            tenant_id=tenant_id,
            tbox_version=tbox_version,
        )
        self._tboxes[tenant_id] = tbox
        await self.graph_store.persist_ontology_tbox(tbox)
        return tbox, report

    def get_tbox(self, tenant_id: str = "enterprise_default") -> EnterpriseCompiledTBox:
        if tenant_id not in self._tboxes:
            self._tboxes[tenant_id] = EnterpriseCompiledTBox(tenant_id=tenant_id)
        return self._tboxes[tenant_id]

    async def build_or_update_knowledge_graph(
        self,
        tenant_id: str = "enterprise_default",
        incremental: bool = True,
        validate_shacl: bool = True,
        allow_dynamic_emergent: bool = True,
    ) -> KGGraphBuildReport:
        if tenant_id not in self._tboxes:
            await self.induce_and_align_ontology(tenant_id=tenant_id)
        tbox = self._tboxes[tenant_id]

        registered_kbs = await self.vector_store.list_registered_kbs(tenant_id=tenant_id)
        total_chunks_inspected = 0
        delta_chunks_to_extract: list[VectorChunkRecord] = []

        for kb in registered_kbs:
            all_kb_chunks = await self.vector_store.fetch_kb_chunks(
                kb.kb_id, only_unindexed_or_changed=False
            )
            total_chunks_inspected += len(all_kb_chunks)
            if incremental:
                delta = await self.vector_store.fetch_kb_chunks(
                    kb.kb_id,
                    only_unindexed_or_changed=True,
                    target_tbox_version=tbox.version,
                )
                delta_chunks_to_extract.extend(delta)
            else:
                delta_chunks_to_extract.extend(all_kb_chunks)

        skipped_chunks = total_chunks_inspected - len(delta_chunks_to_extract)

        # Extract entities and relation triples from delta chunks via LlamaIndex extractors
        extractions: list[ChunkExtractionResult] = []
        raw_entity_mentions = 0
        for chk in delta_chunks_to_extract:
            res = self.kg_extractor.extract_from_chunk(
                chk, tbox=tbox, allow_dynamic_emergent=allow_dynamic_emergent
            )
            raw_entity_mentions += len(res.entities) + (2 * len(res.triples))
            extractions.append(res)

        existing_entities = await self.graph_store.get_all_entities(tenant_id)
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

        self._cumulative_sibling_collisions_prevented[tenant_id] = (
            self._cumulative_sibling_collisions_prevented.get(tenant_id, 0)
            + sibling_collisions_prevented
        )

        # Validate against W3C OWL / SHACL shapes
        shacl_ok = True
        if validate_shacl and canonical_entities:
            shacl_ok, _ = OWLSHACLCompiler.validate_abox_with_shacl(
                tbox=tbox,
                entities=list(canonical_entities.values()),
                edges=resolved_edges,
            )

        # Persist to Neo4j / Neptune & PostgreSQL pgvector canonical entity table
        superseded_closed = await self.graph_store.upsert_entities_and_bitemporal_edges(
            tenant_id=tenant_id,
            entities=list(canonical_entities.values()),
            edges=resolved_edges,
        )
        await self.vector_store.upsert_canonical_entity_embeddings(
            list(canonical_entities.values())
        )

        # Commit SHA-256 watermarks for all processed delta chunks
        watermarks = [
            KBWatermarkState(
                kb_id=chk.kb_id,
                chunk_id=chk.chunk_id,
                content_sha256=chk.content_sha256,
                tbox_version=tbox.version,
            )
            for chk in delta_chunks_to_extract
        ]
        if watermarks:
            await self.vector_store.mark_chunks_watermarked(watermarks)

        active_edges = await self.graph_store.get_all_edges(tenant_id, active_only=True)

        return KGGraphBuildReport(
            tenant_id=tenant_id,
            tbox_version=tbox.version,
            kbs_processed=len(registered_kbs),
            total_chunks_inspected=total_chunks_inspected,
            delta_chunks_extracted=len(delta_chunks_to_extract),
            skipped_unchanged_chunks_via_watermark=skipped_chunks,
            raw_entity_mentions=raw_entity_mentions,
            canonical_entities_after_er=len(canonical_entities),
            cross_kb_merged_entities=cross_kb_merged_count,
            sibling_collisions_prevented=self._cumulative_sibling_collisions_prevented.get(
                tenant_id, 0
            ),
            active_edges_upserted=len(active_edges),
            superseded_bitemporal_edges_closed=superseded_closed,
            shacl_conformance_passed=shacl_ok,
        )

    async def promote_quarantined_ontology_concepts(
        self,
        tenant_id: str = "enterprise_default",
        min_confidence: float = 0.80,
    ) -> dict[str, Any]:
        """
        Bottom-Up Ontology Evolution: promotes quarantined candidate classes, relations,
        and graph vertices/edges that meet confidence threshold into `APPROVED` status.
        """
        tbox = self.get_tbox(tenant_id)
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

        # Bump minor TBox version
        parts = tbox.version.split(".")
        if len(parts) == 3 and parts[1].isdigit():
            tbox.version = f"{parts[0]}.{int(parts[1]) + 1}.0"

        await self.graph_store.persist_ontology_tbox(tbox)

        # Promote matching entities and edges in the graph store
        entities = await self.graph_store.get_all_entities(tenant_id)
        edges = await self.graph_store.get_all_edges(tenant_id, active_only=False)
        promoted_entity_count = 0
        promoted_edge_count = 0

        for ent in entities.values():
            cls_def = tbox.classes.get(ent.ontology_class)
            if cls_def and cls_def.status == OntologyStatus.APPROVED:
                if ent.governance_bucket != OntologyStatus.APPROVED:
                    ent.governance_bucket = OntologyStatus.APPROVED
                    promoted_entity_count += 1

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
                    promoted_edge_count += 1

        return {
            "tenant_id": tenant_id,
            "new_tbox_version": tbox.version,
            "promoted_classes": promoted_classes,
            "promoted_relations": promoted_relations,
            "promoted_entities": promoted_entity_count,
            "promoted_edges": promoted_edge_count,
        }

    def export_ontology_turtle(self, tenant_id: str = "enterprise_default") -> str:
        tbox = self.get_tbox(tenant_id)
        return OWLSHACLCompiler.export_turtle(tbox)

    async def query_graph_rag(self, request: GraphRAGQueryRequest) -> GraphRAGQueryResponse:
        tbox = self.get_tbox(request.tenant_id)
        return await self.rag_engine.query(request=request, tbox=tbox)

    # -------------------------------------------------------------------------
    # Granular Agentic Search Primitives (FastAPI `/v1/search/*` + MCP Tools)
    # -------------------------------------------------------------------------

    async def search_vector_chunks(
        self, request: ChunkSearchRequest
    ) -> list[RetrievedChunkContext]:
        """Granular Multi-Model Federated pgvector + BM25 RRF chunk search across KBs."""
        q_vec = self.embedding_provider.embed_text(request.query, dim=self.canonical_dim)
        return await self.vector_store.hybrid_search_chunks(
            tenant_id=request.tenant_id,
            query_text=request.query,
            query_embedding=q_vec,
            top_k=request.top_k,
            kb_filter=request.kb_filter,
            domain_filter=request.domain_filter,
        )

    async def search_kg_entities(
        self, request: EntitySearchRequest
    ) -> list[EntitySearchMatch]:
        """Granular Knowledge Graph canonical entity lookup by name, alias, class, or vector similarity."""
        all_entities = await self.graph_store.get_all_entities(request.tenant_id)
        q_lower = request.query.strip().lower()
        q_vec = self.embedding_provider.embed_text(request.query, dim=self.canonical_dim)

        matches: list[EntitySearchMatch] = []
        for ent in all_entities.values():
            if not request.include_quarantined and ent.governance_bucket != OntologyStatus.APPROVED:
                continue
            if (
                request.ontology_class_filter
                and ent.ontology_class.lower() != request.ontology_class_filter.lower()
            ):
                continue

            # Compute hybrid lexical + vector score
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
                        is_cross_kb_bridge=(len(ent.source_kb_ids) >= 2),
                        match_score=score,
                    )
                )

        matches.sort(key=lambda m: (-m.match_score, -len(m.source_kb_ids)))
        return matches[: request.top_k]

    async def traverse_entity_subgraph(
        self, request: SubgraphSearchRequest
    ) -> SubgraphSearchResponse:
        """Granular K-hop bi-temporal subgraph expansion around specific entity IDs or names."""
        all_entities = await self.graph_store.get_all_entities(request.tenant_id)
        resolved_ids: list[str] = []

        for seed in request.seed_entities:
            s_clean = seed.strip()
            if s_clean in all_entities:
                resolved_ids.append(s_clean)
                continue
            s_lower = s_clean.lower()
            for eid, ent in all_entities.items():
                names = [ent.canonical_name.lower()] + [a.lower() for a in ent.aliases]
                if s_lower in names or any(s_lower in n for n in names):
                    if eid not in resolved_ids:
                        resolved_ids.append(eid)

        facts, bridges = await self.graph_store.expand_k_hop_subgraph(
            tenant_id=request.tenant_id,
            seed_entity_ids=resolved_ids,
            max_hops=request.max_hops,
            include_quarantined=request.include_quarantined,
            as_of_iso=request.as_of_iso,
        )
        if request.predicate_filter:
            allowed_preds = {p.upper() for p in request.predicate_filter}
            facts = [f for f in facts if f.predicate.upper() in allowed_preds]

        return SubgraphSearchResponse(
            tenant_id=request.tenant_id,
            resolved_seed_entity_ids=resolved_ids,
            cross_kb_bridge_entities=bridges,
            facts=facts,
        )

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

    async def execute_readonly_opencypher(
        self, request: ReadOnlyCypherSearchRequest
    ) -> ReadOnlyCypherSearchResponse:
        """Executes a guarded read-only OpenCypher query on Neo4j / Amazon Neptune."""
        self.validate_readonly_opencypher(request.cypher_query)
        params = {**request.params, "tenant_id": request.tenant_id}
        rows = await self.graph_store.execute_opencypher(
            tenant_id=request.tenant_id,
            cypher_query=request.cypher_query,
            params=params,
        )
        return ReadOnlyCypherSearchResponse(
            tenant_id=request.tenant_id,
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
        tbox = self.get_tbox(tenant_id)

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
        }
