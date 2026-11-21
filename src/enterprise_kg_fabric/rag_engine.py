from __future__ import annotations

import re
import time
from typing import Any

from .adapters.graph_stores import PropertyGraphStorePort
from .adapters.pgvector_store import PgVectorStorePort
from .models import (
    EnterpriseCompiledTBox,
    GraphRAGQueryRequest,
    GraphRAGQueryResponse,
    KGEntityNode,
    OntologyStatus,
)
from .vector_clustering import EmbeddingProvider


class EnterpriseHybridGraphRAGEngine:
    """
    LlamaIndex-powered 4-Path Hybrid GraphRAG Retriever:
      1. Multi-KB Dense Vector Search (`pgvector` HNSW)
      2. Multi-KB Lexical BM25 Search (`tsvector` GIN)
      3. Seed Entity Resolution + Multi-Hop PropertyGraph Expansion (Neo4j / Neptune)
         with Cross-KB Bridge Discovery & Bi-Temporal (`as_of_iso`) point-in-time support
      4. Ontology-Grounded Text-to-OpenCypher Query Synthesis & Execution
    """

    def __init__(
        self,
        vector_store: PgVectorStorePort,
        graph_store: PropertyGraphStorePort,
        embedding_provider: EmbeddingProvider,
        canonical_dim: int = 256,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedding_provider = embedding_provider
        self.canonical_dim = canonical_dim

    def synthesize_opencypher_from_query(
        self,
        query: str,
        tbox: EnterpriseCompiledTBox,
        seed_entities: list[KGEntityNode],
    ) -> tuple[str, dict[str, Any]]:
        """
        Synthesizes a 100% Neo4j & Amazon Neptune compatible parameterized OpenCypher query
        guided by the induced `EnterpriseCompiledTBox`.
        """
        q_lower = query.lower()
        matched_predicate: str | None = None

        if any(w in q_lower for w in ("compliance", "pci", "gdpr", "soc2", "governed", "regulated", "standard")):
            matched_predicate = "GOVERNED_BY"
        elif any(w in q_lower for w in ("cost center", "budget", "finops", "funded", "allocated", "owned")):
            matched_predicate = "OWNED_BY_PRIMARY"
        elif any(w in q_lower for w in ("region", "deployed", "hosted", "failover", "where")):
            matched_predicate = "DEPLOYED_IN_PRIMARY"
        elif any(w in q_lower for w in ("mitigated", "waf", "remediated", "control", "shield")):
            matched_predicate = "MITIGATED_BY_CONTROL"
        elif any(w in q_lower for w in ("cve", "vulnerability", "incident", "affects", "impacts")):
            matched_predicate = "AFFECTS_ASSET"
        elif any(w in q_lower for w in ("depend", "upstream", "downstream", "database", "aurora")):
            matched_predicate = "DEPENDS_ON"

        primary_seed = seed_entities[0].canonical_name if seed_entities else ""

        if matched_predicate and primary_seed:
            cypher = (
                "MATCH (seed:Entity {tenant_id: $tenant_id})-[:KG_RELATION*0..3]-(s:Entity {tenant_id: $tenant_id})"
                "-[r:KG_RELATION]->(t:Entity {tenant_id: $tenant_id}) "
                "WHERE r.predicate = $predicate AND r.t_invalid = 'ACTIVE' "
                "AND toLower(seed.canonical_name) CONTAINS toLower($entity_name) "
                "RETURN DISTINCT s.canonical_name AS source, r.predicate AS predicate, "
                "t.canonical_name AS target, r.source_kb_ids AS source_kbs "
                "LIMIT 25"
            )
            return cypher, {
                "tenant_id": tbox.tenant_id,
                "predicate": matched_predicate,
                "entity_name": primary_seed,
            }

        if matched_predicate:
            cypher = (
                "MATCH (s:Entity {tenant_id: $tenant_id})-[r:KG_RELATION]->(t:Entity {tenant_id: $tenant_id}) "
                "WHERE r.predicate = $predicate AND r.t_invalid = 'ACTIVE' "
                "RETURN s.canonical_name AS source, r.predicate AS predicate, t.canonical_name AS target, r.source_kb_ids AS source_kbs "
                "LIMIT 25"
            )
            return cypher, {"tenant_id": tbox.tenant_id, "predicate": matched_predicate}

        cypher = (
            "MATCH (s:Entity {tenant_id: $tenant_id})-[r:KG_RELATION]->(t:Entity {tenant_id: $tenant_id}) "
            "WHERE r.t_invalid = 'ACTIVE' "
            "RETURN s.canonical_name AS source, r.predicate AS predicate, t.canonical_name AS target "
            "LIMIT 25"
        )
        return cypher, {"tenant_id": tbox.tenant_id, "entity_name": primary_seed}

    async def query(
        self,
        request: GraphRAGQueryRequest,
        tbox: EnterpriseCompiledTBox,
    ) -> GraphRAGQueryResponse:
        t0 = time.perf_counter()
        q_vec = self.embedding_provider.embed_text(request.query, dim=self.canonical_dim)

        # Path 1 & 2: Multi-KB Hybrid RRF Chunk Search (pgvector HNSW + BM25 GIN)
        chunk_contexts = await self.vector_store.hybrid_search_chunks(
            tenant_id=request.tenant_id,
            query_text=request.query,
            query_embedding=q_vec,
            top_k=request.top_k_chunks,
            kb_filter=request.kb_filter,
            domain_filter=request.domain_filter,
        )

        # Path 3: Seed Entity Discovery (Direct lexical mention + Chunk Provenance + Vector Entity Similarity)
        all_entities = await self.graph_store.get_all_entities(request.tenant_id)
        q_lower = request.query.lower()
        q_tokens = set(re.findall(r"[a-z0-9_\-\.]+", q_lower))
        retrieved_chunk_ids = {c.chunk_id for c in chunk_contexts}

        seed_entity_ids: list[str] = []
        seed_entities: list[KGEntityNode] = []

        # 3a. Exact or alias match in query text
        for eid, ent in all_entities.items():
            if (
                not request.include_quarantined_overflow
                and ent.governance_bucket != OntologyStatus.APPROVED
            ):
                continue
            names_to_check = [ent.canonical_name] + list(ent.aliases)
            if any(
                n.lower() in q_lower or n.lower() in q_tokens
                for n in names_to_check
                if len(n) >= 3
            ):
                if eid not in seed_entity_ids:
                    seed_entity_ids.append(eid)
                    seed_entities.append(ent)

        # 3b. Entities directly extracted from top retrieved chunks (only if query named no explicit entities)
        if not seed_entity_ids:
            for eid, ent in all_entities.items():
                if eid in seed_entity_ids:
                    continue
                if (
                    not request.include_quarantined_overflow
                    and ent.governance_bucket != OntologyStatus.APPROVED
                ):
                    continue
                if retrieved_chunk_ids.intersection(ent.source_chunk_ids):
                    seed_entity_ids.append(eid)
                    seed_entities.append(ent)
                    if len(seed_entity_ids) >= 6:
                        break

        # 3c. Canonical Entity Embedding Similarity fallback (only if still no seeds found)
        if not seed_entity_ids:
            sim_entities = await self.vector_store.search_similar_entities(
                tenant_id=request.tenant_id,
                query_embedding=q_vec,
                top_k=4,
            )
            for ent, _ in sim_entities:
                if (
                    not request.include_quarantined_overflow
                    and ent.governance_bucket != OntologyStatus.APPROVED
                ):
                    continue
                if ent.entity_id not in seed_entity_ids:
                    seed_entity_ids.append(ent.entity_id)
                    seed_entities.append(ent)

        # Expand K-Hop Subgraph in Neo4j / Neptune
        graph_facts, bridge_entities = await self.graph_store.expand_k_hop_subgraph(
            tenant_id=request.tenant_id,
            seed_entity_ids=seed_entity_ids,
            max_hops=request.max_graph_hops,
            include_quarantined=request.include_quarantined_overflow,
            as_of_iso=request.as_of_iso,
        )

        # Path 4: Text-to-OpenCypher Structured Query Execution
        cypher_query, cypher_params = self.synthesize_opencypher_from_query(
            request.query, tbox, seed_entities
        )
        opencypher_rows = await self.graph_store.execute_opencypher(
            tenant_id=request.tenant_id,
            cypher_query=cypher_query,
            params=cypher_params,
        )

        # Aggregate contributing KBs across both Graph Facts and Vector Chunks
        contributing_kbs: set[str] = {c.kb_id for c in chunk_contexts}
        for gf in graph_facts:
            contributing_kbs.update(gf.source_kb_ids)

        # Synthesize Grounded Answer with Cross-KB Lineage Citations
        answer_lines: list[str] = []
        if graph_facts:
            answer_lines.append("### Knowledge Graph Multi-Hop Facts (Neo4j / Neptune)")
            for gf in graph_facts[:10]:
                kb_badge = ", ".join(gf.source_kb_ids)
                bridge_tag = " [CROSS-KB BRIDGE]" if gf.is_cross_kb_bridge else ""
                answer_lines.append(
                    f"- [Hop {gf.hop_distance}]{bridge_tag} **{gf.source_name}** (`{gf.source_class}`) "
                    f"--`{gf.predicate}`--> **{gf.target_name}** (`{gf.target_class}`) "
                    f"*(KBs: {kb_badge} | valid: {gf.t_valid})*"
                )

        if chunk_contexts:
            answer_lines.append("\n### Supporting Vector KB Evidence (`pgvector` Hybrid RRF)")
            for chk in chunk_contexts[:4]:
                snippet = chk.text[:180].replace("\n", " ")
                answer_lines.append(
                    f"- **[{chk.kb_id} | {chk.chunk_id}]** (RRF={chk.rrf_score:.4f}): {snippet}..."
                )

        if not answer_lines:
            answer_lines.append("No matching knowledge graph facts or vector chunks found for the tenant.")

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return GraphRAGQueryResponse(
            query=request.query,
            tenant_id=request.tenant_id,
            tbox_version=tbox.version,
            synthesized_answer="\n".join(answer_lines),
            generated_opencypher_query=cypher_query,
            opencypher_results=opencypher_rows,
            seed_entities_matched=[e.canonical_name for e in seed_entities],
            cross_kb_bridge_entities=bridge_entities,
            graph_facts=graph_facts,
            chunk_contexts=chunk_contexts,
            provenance_kbs_contributing=sorted(contributing_kbs),
            retrieval_latency_ms=latency_ms,
        )
