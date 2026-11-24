from __future__ import annotations

import re
import time
from typing import Any

from .adapters.graph_stores import PropertyGraphStorePort
from .adapters.pgvector_store import PgVectorStorePort
from .adaptive_router import AdaptiveQueryRouter, PluggableCrossEncoderReranker
from .graph_analytics import GraphAnalyticsEngine, _is_acl_allowed
from .models import (
    EnterpriseCompiledTBox,
    GraphRAGQueryRequest,
    GraphRAGQueryResponse,
    KGEntityNode,
    OntologyStatus,
    QueryRouteDecision,
)
from .vector_clustering import EmbeddingProvider


class EnterpriseHybridGraphRAGEngine:
    """
    LlamaIndex-powered Adaptive Hybrid GraphRAG Engine (`v0.2.0`):
      - Adaptive Query Router (`Adaptive-RAG`): routes between Factual Vector, Local Multi-Hop
        HippoRAG 2 PPR, Structured OpenCypher, and Microsoft GraphRAG Leiden Global Sensemaking.
      - ABAC Security Trimming (`caller_acl_groups`) enforced across pgvector + Neo4j/Neptune.
      - HippoRAG 2 Personalized PageRank (PPR) hub-damping over multi-hop subgraphs.
      - Pluggable Cross-Encoder Reranking over retrieved vector chunks.
      - N-ary Reified Claim Qualifiers rendered alongside provenance citations.
    """

    def __init__(
        self,
        vector_store: PgVectorStorePort,
        graph_store: PropertyGraphStorePort,
        embedding_provider: EmbeddingProvider,
        graph_analytics: GraphAnalyticsEngine | None = None,
        reranker: PluggableCrossEncoderReranker | None = None,
        canonical_dim: int = 256,
        llm_extractor: Any | None = None,
        answer_synthesizer: Any | None = None,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedding_provider = embedding_provider
        self.canonical_dim = canonical_dim
        self.graph_analytics = graph_analytics or GraphAnalyticsEngine(
            embedding_provider=embedding_provider, canonical_dim=canonical_dim
        )
        self.reranker = reranker or PluggableCrossEncoderReranker()
        self.llm_extractor = llm_extractor
        self.answer_synthesizer = answer_synthesizer

    async def synthesize_opencypher_async(
        self,
        query: str,
        tbox: EnterpriseCompiledTBox,
        seed_entities: list[KGEntityNode],
        allow_llm_plan: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """
        Optional LLM-assisted structured intent planning when `llm_extractor.use_live_llm=True`
        and `allow_llm_plan=True`.

        Security invariant: NEVER executes raw model-written Cypher strings on the shared graph!
        The LLM may ONLY propose a JSON `{predicate, entity_hint}` grounded in `tbox`, which is
        validated and compiled strictly into hardcoded `{tenant_id: $tenant_id}` templates.
        """
        matched_predicate: str | None = None
        primary_seed = seed_entities[0].canonical_name if seed_entities else ""

        if (
            allow_llm_plan
            and self.llm_extractor is not None
            and getattr(self.llm_extractor, "use_live_llm", False)
        ):
            gen_async = getattr(self.llm_extractor, "generate_text_async", None)
            if gen_async is not None:
                preds = list(tbox.relations.keys())[:30]
                seeds = [e.canonical_name for e in seed_entities[:5]]
                prompt = (
                    "Classify the graph lookup intent as strict JSON: "
                    '{"predicate": "UPPER_SNAKE_OR_EMPTY", "entity_hint": "entity_or_empty"}.\n'
                    f"Allowed Predicates: {preds}\nCandidate Entities: {seeds}\nQuestion: {query}"
                )
                raw = await gen_async(
                    prompt, "Return ONLY valid JSON with keys predicate and entity_hint."
                )
                if raw:
                    try:
                        import json as _json

                        m = re.search(r"\{.*\}", raw, re.S)
                        plan = _json.loads(m.group(0) if m else raw)
                        cand_pred = str(plan.get("predicate", "")).strip()
                        if cand_pred:
                            resolved_p = tbox.resolve_predicate(cand_pred)
                            if resolved_p in tbox.relations:
                                matched_predicate = resolved_p
                        cand_ent = str(plan.get("entity_hint", "")).strip()
                        if cand_ent and not primary_seed:
                            primary_seed = cand_ent[:120]
                    except Exception:
                        pass

        if matched_predicate:
            return self._compile_safe_opencypher_template(
                tenant_id=tbox.tenant_id,
                matched_predicate=matched_predicate,
                primary_seed=primary_seed,
            )
        return self.synthesize_opencypher_from_query(query, tbox, seed_entities)

    @staticmethod
    def _compile_safe_opencypher_template(
        tenant_id: str,
        matched_predicate: str | None,
        primary_seed: str,
    ) -> tuple[str, dict[str, Any]]:
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
                "tenant_id": tenant_id,
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
            return cypher, {"tenant_id": tenant_id, "predicate": matched_predicate}
        cypher = (
            "MATCH (s:Entity {tenant_id: $tenant_id})-[r:KG_RELATION]->(t:Entity {tenant_id: $tenant_id}) "
            "WHERE r.t_invalid = 'ACTIVE' "
            "RETURN s.canonical_name AS source, r.predicate AS predicate, t.canonical_name AS target "
            "LIMIT 25"
        )
        return cypher, {"tenant_id": tenant_id, "entity_name": primary_seed}

    async def _synthesize_grounded_llm_answer(
        self,
        query: str,
        graph_facts: list[Any],
        chunk_contexts: list[Any],
        deterministic_fallback: str,
    ) -> str:
        """
        Optional grounded LLM answer synthesis with citations (`request.synthesize_with_llm=True`).
        Uses an injected `answer_synthesizer(query, graph_facts, chunk_contexts)` callback or
        `llm_extractor.generate_text_async(...)`, falling back cleanly to structured evidence.
        """
        if self.answer_synthesizer is not None:
            res = self.answer_synthesizer(query, graph_facts, chunk_contexts)
            if hasattr(res, "__await__"):
                res = await res
            if res:
                return str(res)
        if self.llm_extractor is not None and getattr(self.llm_extractor, "use_live_llm", False):
            gen_async = getattr(self.llm_extractor, "generate_text_async", None)
            if gen_async is not None:
                prompt = (
                    f"Answer the user question strictly using ONLY the verified Knowledge Graph facts "
                    f"and Vector KB evidence below. Cite exact KB IDs and entity names inline. "
                    f"Do not invent facts not present in the evidence.\n\n"
                    f"Question: {query}\n\nEvidence:\n{deterministic_fallback}"
                )
                out = await gen_async(
                    prompt,
                    "You are a grounded enterprise GraphRAG synthesizer. Every claim must cite provided evidence.",
                )
                if out:
                    return out
        return deterministic_fallback

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
        allow_llm_calls: bool = True,
    ) -> GraphRAGQueryResponse:
        t0 = time.perf_counter()
        route = (
            AdaptiveQueryRouter.classify_query(request.query, as_of_iso=request.as_of_iso)
            if request.use_adaptive_routing
            else QueryRouteDecision.LOCAL_MULTIHOP_PPR
        )

        # Route A: Microsoft GraphRAG Global Sensemaking via Persisted Build-Time Communities
        if route == QueryRouteDecision.GLOBAL_SENSEMAKING:
            persisted = await self.vector_store.load_community_summaries(request.tenant_id)
            if persisted:
                global_res = self.graph_analytics.query_persisted_communities(
                    tenant_id=request.tenant_id,
                    query=request.query,
                    persisted_communities=persisted,
                    top_k_communities=4,
                    caller_acl_groups=request.caller_acl_groups,
                )
            else:
                all_entities = await self.graph_store.get_all_entities(request.tenant_id)
                all_edges = await self.graph_store.get_all_edges(request.tenant_id, active_only=True)
                built_comms, eng = self.graph_analytics.detect_communities_and_summarize(
                    tenant_id=request.tenant_id,
                    entities=all_entities,
                    edges=all_edges,
                    caller_acl_groups=None,
                )
                if built_comms:
                    await self.vector_store.save_community_summaries(request.tenant_id, built_comms)
                global_res = self.graph_analytics.query_persisted_communities(
                    tenant_id=request.tenant_id,
                    query=request.query,
                    persisted_communities=built_comms,
                    top_k_communities=4,
                    caller_acl_groups=request.caller_acl_groups,
                    engine_used=eng,
                )
            contributing_kbs = sorted(
                {kb for comm in global_res.communities_matched for kb in comm.contributing_kb_ids}
            )
            return GraphRAGQueryResponse(
                query=request.query,
                tenant_id=request.tenant_id,
                tbox_version=tbox.version,
                route_taken=route,
                synthesized_answer=global_res.synthesized_global_answer,
                generated_opencypher_query="-- Routed to Persisted Leiden Community Global Sensemaking",
                community_summaries_used=[
                    c.model_dump(exclude={"embedding"}) for c in global_res.communities_matched
                ],
                provenance_kbs_contributing=contributing_kbs,
                retrieval_latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        q_vec = self.embedding_provider.embed_text(request.query, dim=self.canonical_dim)

        # Path 1 & 2: Multi-KB Hybrid RRF Chunk Search (with ABAC ACL Trimming + Cross-Encoder Reranking)
        raw_chunks = await self.vector_store.hybrid_search_chunks(
            tenant_id=request.tenant_id,
            query_text=request.query,
            query_embedding=q_vec,
            top_k=request.top_k_chunks,
            kb_filter=request.kb_filter,
            domain_filter=request.domain_filter,
            caller_acl_groups=request.caller_acl_groups,
        )
        chunk_contexts = self.reranker.rerank_chunks(
            request.query, raw_chunks, top_k=request.top_k_chunks
        )

        # Route B: Fast Factual Vector-Only Route (skips graph overhead on simple lookups)
        if route == QueryRouteDecision.FACTUAL_VECTOR:
            answer_lines = ["### Direct Vector KB Evidence (Adaptive Fast Path)"]
            for chk in chunk_contexts[:4]:
                snippet = chk.text[:180].replace("\n", " ")
                answer_lines.append(f"- **[{chk.kb_id} | {chk.chunk_id}]** (score={chk.rrf_score:.4f}): {snippet}...")
            return GraphRAGQueryResponse(
                query=request.query,
                tenant_id=request.tenant_id,
                tbox_version=tbox.version,
                route_taken=route,
                synthesized_answer="\n".join(answer_lines),
                generated_opencypher_query="-- Skipped via AdaptiveQueryRouter (FACTUAL_VECTOR)",
                chunk_contexts=chunk_contexts,
                provenance_kbs_contributing=sorted({c.kb_id for c in chunk_contexts}),
                retrieval_latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        # Path 3: Indexed Seed Entity Discovery (zero full-tenant graph scan!)
        q_lower = request.query.lower()
        q_tokens = set(re.findall(r"[a-z0-9_\-\.]+", q_lower))
        seed_entity_ids: list[str] = []
        seed_entities: list[KGEntityNode] = []

        # 3a. Indexed lexical / alias lookup in graph store (Neo4j index / InMemory)
        indexed_matches = await self.graph_store.search_entities_indexed(
            tenant_id=request.tenant_id,
            query=request.query,
            limit=12,
            include_quarantined=request.include_quarantined_overflow,
            caller_acl_groups=request.caller_acl_groups,
        )
        for ent in indexed_matches:
            names_to_check = [ent.canonical_name] + list(ent.aliases)
            if any(n.lower() in q_lower or n.lower() in q_tokens for n in names_to_check if len(n) >= 3):
                if ent.entity_id not in seed_entity_ids:
                    seed_entity_ids.append(ent.entity_id)
                    seed_entities.append(ent)

        # 3b. Canonical Entity Embedding HNSW Similarity fallback in pgvector
        if not seed_entity_ids:
            sim_entities = await self.vector_store.search_similar_entities(
                tenant_id=request.tenant_id,
                query_embedding=q_vec,
                top_k=6,
            )
            for sim_ent, _ in sim_entities:
                full_ent = await self.graph_store.get_entity_by_id(request.tenant_id, sim_ent.entity_id)
                ent = full_ent or sim_ent
                if not request.include_quarantined_overflow and ent.governance_bucket != OntologyStatus.APPROVED:
                    continue
                if not _is_acl_allowed(ent.acl_groups, request.caller_acl_groups):
                    continue
                if ent.entity_id not in seed_entity_ids:
                    seed_entity_ids.append(ent.entity_id)
                    seed_entities.append(ent)
                    if len(seed_entity_ids) >= 4:
                        break

        # Expand K-Hop Subgraph in Neo4j / Neptune (with ABAC security trimming!)
        graph_facts, bridge_entities = await self.graph_store.expand_k_hop_subgraph(
            tenant_id=request.tenant_id,
            seed_entity_ids=seed_entity_ids,
            max_hops=request.max_graph_hops,
            include_quarantined=request.include_quarantined_overflow,
            as_of_iso=request.as_of_iso,
            caller_acl_groups=request.caller_acl_groups,
        )

        # HippoRAG 2 PPR Hub-Damped Re-Scoring on BOUNDED ego-subgraph (zero full-graph scan!)
        ego_entities: dict[str, KGEntityNode] = {e.entity_id: e for e in seed_entities}
        if request.use_ppr_ranking and seed_entity_ids and graph_facts:
            ego_entities, ego_edges = await self.graph_store.fetch_ego_subgraph(
                tenant_id=request.tenant_id,
                seed_entity_ids=seed_entity_ids,
                max_hops=max(2, request.max_graph_hops),
                include_quarantined=request.include_quarantined_overflow,
                as_of_iso=request.as_of_iso,
                caller_acl_groups=request.caller_acl_groups,
            )
            ppr_res = self.graph_analytics.search_ppr_subgraph(
                tenant_id=request.tenant_id,
                entities=ego_entities,
                edges=ego_edges,
                seed_entity_ids=seed_entity_ids,
                top_k_facts=max(len(graph_facts), 25),
                as_of_iso=request.as_of_iso,
                include_quarantined=request.include_quarantined_overflow,
                caller_acl_groups=request.caller_acl_groups,
            )
            ppr_by_edge = {f.edge_id: f.ppr_score for f in ppr_res.ranked_facts}
            for gf in graph_facts:
                gf.ppr_score = ppr_by_edge.get(gf.edge_id, 0.0)
            graph_facts.sort(key=lambda f: (f.hop_distance, -f.ppr_score, -f.confidence))

        # Path 4: Text-to-OpenCypher Structured Query Execution (with strict fail-closed ABAC post-filtering)
        cypher_query, cypher_params = await self.synthesize_opencypher_async(
            request.query, tbox, seed_entities, allow_llm_plan=allow_llm_calls
        )
        raw_opencypher_rows = await self.graph_store.execute_opencypher(
            tenant_id=request.tenant_id,
            cypher_query=cypher_query,
            params=cypher_params,
        )
        if request.caller_acl_groups is not None:
            allowed_entity_names = {
                ent.canonical_name
                for ent in ego_entities.values()
                if _is_acl_allowed(ent.acl_groups, request.caller_acl_groups)
            }
            for gf in graph_facts:
                allowed_entity_names.add(gf.source_name)
                allowed_entity_names.add(gf.target_name)
            allowed_fact_pairs = {
                (gf.source_name, gf.predicate, gf.target_name) for gf in graph_facts
            }
            opencypher_rows = []
            for row in raw_opencypher_rows:
                src_val = row.get("source")
                tgt_val = row.get("target")
                pred_val = row.get("predicate")
                # Fail closed: every row returned to an ABAC-scoped caller MUST carry explicit
                # verified (source, predicate, target) columns matching permitted subgraph facts!
                if not isinstance(src_val, str) or not isinstance(tgt_val, str) or not isinstance(pred_val, str):
                    continue
                if src_val not in allowed_entity_names or tgt_val not in allowed_entity_names:
                    continue
                if (src_val, pred_val, tgt_val) not in allowed_fact_pairs:
                    continue
                opencypher_rows.append(row)
        else:
            opencypher_rows = raw_opencypher_rows

        contributing_kbs: set[str] = {c.kb_id for c in chunk_contexts}
        for gf in graph_facts:
            contributing_kbs.update(gf.source_kb_ids)

        # Synthesize Grounded Answer with Cross-KB Lineage Citations & N-ary Qualifiers
        answer_lines: list[str] = []
        if graph_facts:
            answer_lines.append("### Knowledge Graph Multi-Hop Facts (Neo4j / Neptune + PPR Ranked)")
            for gf in graph_facts[:10]:
                kb_badge = ", ".join(gf.source_kb_ids)
                bridge_tag = " [CROSS-KB BRIDGE]" if gf.is_cross_kb_bridge else ""
                qual_str = f" | qualifiers={gf.qualifiers}" if gf.qualifiers else ""
                answer_lines.append(
                    f"- [Hop {gf.hop_distance} | PPR={gf.ppr_score:.3f}]{bridge_tag} **{gf.source_name}** (`{gf.source_class}`) "
                    f"--`{gf.predicate}`--> **{gf.target_name}** (`{gf.target_class}`) "
                    f"*(KBs: {kb_badge} | valid: {gf.t_valid}{qual_str})*"
                )

        if chunk_contexts:
            answer_lines.append("\n### Supporting Vector KB Evidence (`pgvector` Hybrid RRF + Reranked)")
            for chk in chunk_contexts[:4]:
                snippet = chk.text[:180].replace("\n", " ")
                answer_lines.append(
                    f"- **[{chk.kb_id} | {chk.chunk_id}]** (score={chk.rrf_score:.4f}): {snippet}..."
                )

        if not answer_lines:
            answer_lines.append("No matching knowledge graph facts or vector chunks found for the tenant / ACL scope.")

        deterministic_answer = "\n".join(answer_lines)
        final_answer = (
            await self._synthesize_grounded_llm_answer(
                request.query, graph_facts, chunk_contexts, deterministic_answer
            )
            if (request.synthesize_with_llm and allow_llm_calls and (graph_facts or chunk_contexts))
            else deterministic_answer
        )

        return GraphRAGQueryResponse(
            query=request.query,
            tenant_id=request.tenant_id,
            tbox_version=tbox.version,
            route_taken=route,
            synthesized_answer=final_answer,
            generated_opencypher_query=cypher_query,
            opencypher_results=opencypher_rows,
            seed_entities_matched=[e.canonical_name for e in seed_entities],
            cross_kb_bridge_entities=bridge_entities,
            graph_facts=graph_facts,
            chunk_contexts=chunk_contexts,
            provenance_kbs_contributing=sorted(contributing_kbs),
            retrieval_latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )
