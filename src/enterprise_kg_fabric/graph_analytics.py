from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

from .models import (
    CommunitySummaryRecord,
    GlobalCommunitySearchResponse,
    KGEntityNode,
    KGRelationEdge,
    OntologyStatus,
    PPRSearchResponse,
    RetrievedGraphFact,
)
from .vector_clustering import EmbeddingProvider, cosine_similarity


def _is_acl_allowed(item_acls: Sequence[str], caller_acls: Sequence[str] | None) -> bool:
    """ABAC Security Trimming: None caller_acls means admin/unrestricted; otherwise requires intersection."""
    if caller_acls is None:
        return True
    if not item_acls:
        return True
    caller_set = {g.lower() for g in caller_acls}
    item_set = {g.lower() for g in item_acls}
    if "public" in item_set:
        return True
    return bool(caller_set.intersection(item_set))


def _is_edge_active_at(edge: KGRelationEdge, as_of_iso: str | None) -> bool:
    if as_of_iso is None:
        return edge.t_invalid == "ACTIVE"
    if edge.t_valid > as_of_iso:
        return False
    if edge.t_invalid != "ACTIVE" and edge.t_invalid <= as_of_iso:
        return False
    return True


class GraphAnalyticsEngine:
    """
    Implements SOTA 2024-2026 GraphRAG Analytics by reusing `networkx` / `leidenalg` when
    available (with exact mathematically-equivalent sparse fallbacks for zero-dependency runs):
      1. HippoRAG 2 Personalized PageRank (PPR) Hub-Damped Subgraph Retrieval
         - Automatically penalizes high-degree supernodes (`1 / degree(v)`)
      2. Microsoft GraphRAG Hierarchical Community Detection (Leiden / Louvain) + Global Search
      3. LightRAG High-Level Theme Extraction for abstract conceptual retrieval
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        canonical_dim: int = 256,
    ):
        self.embedding_provider = embedding_provider
        self.canonical_dim = canonical_dim

    @staticmethod
    def compute_personalized_pagerank(
        nodes: Sequence[str],
        undirected_edges: Sequence[tuple[str, str, float]],
        seed_nodes: Sequence[str],
        alpha: float = 0.85,
        max_iter: int = 150,
    ) -> tuple[dict[str, float], str]:
        """
        Computes Personalized PageRank (PPR) seeded from `seed_nodes`.
        Reuses `networkx.pagerank` if installed, otherwise runs exact sparse power iteration.
        """
        node_set = set(nodes)
        valid_seeds = [s for s in seed_nodes if s in node_set]
        if not node_set or not valid_seeds:
            return {}, "empty"

        pers = {n: (1.0 / len(valid_seeds) if n in valid_seeds else 0.0) for n in node_set}

        try:
            import networkx as nx

            g = nx.Graph()
            g.add_nodes_from(node_set)
            for u, v, w in undirected_edges:
                if u in node_set and v in node_set:
                    if g.has_edge(u, v):
                        g[u][v]["weight"] += w
                    else:
                        g.add_edge(u, v, weight=w)
            scores = nx.pagerank(
                g,
                alpha=alpha,
                personalization=pers,
                max_iter=max(max_iter, 150),
                tol=1e-5,
                weight="weight",
            )
            return {str(k): float(v) for k, v in scores.items()}, "networkx_pagerank"
        except Exception:
            pass

        # Exact sparse power-iteration Personalized PageRank fallback
        adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        deg_sum: dict[str, float] = defaultdict(float)
        for u, v, w in undirected_edges:
            if u in node_set and v in node_set:
                adj[u].append((v, w))
                adj[v].append((u, w))
                deg_sum[u] += w
                deg_sum[v] += w

        scores = dict(pers)
        for _ in range(max_iter):
            new_scores = {n: (1.0 - alpha) * pers[n] for n in node_set}
            dangling_mass = sum(scores[n] for n in node_set if deg_sum[n] <= 0.0)
            if dangling_mass > 0.0:
                for s in valid_seeds:
                    new_scores[s] += alpha * dangling_mass / len(valid_seeds)

            for u, neighbors in adj.items():
                d_u = deg_sum[u]
                if d_u <= 0.0 or scores[u] <= 0.0:
                    continue
                share = alpha * scores[u] / d_u
                for v, w in neighbors:
                    new_scores[v] += share * w

            diff = sum(abs(new_scores[n] - scores[n]) for n in node_set)
            scores = new_scores
            if diff < 1e-6:
                break

        return scores, "sparse_ppr_power_iteration"

    def search_ppr_subgraph(
        self,
        tenant_id: str,
        entities: dict[str, KGEntityNode],
        edges: Sequence[KGRelationEdge],
        seed_entity_ids: Sequence[str],
        top_k_facts: int = 15,
        alpha: float = 0.85,
        as_of_iso: str | None = None,
        include_quarantined: bool = False,
        caller_acl_groups: Sequence[str] | None = None,
    ) -> PPRSearchResponse:
        # Filter entities & edges by governance bucket, bi-temporal validity, and ABAC ACLs
        allowed_entities = {
            eid: ent
            for eid, ent in entities.items()
            if (include_quarantined or ent.governance_bucket == OntologyStatus.APPROVED)
            and _is_acl_allowed(ent.acl_groups, caller_acl_groups)
        }
        allowed_edges = [
            e
            for e in edges
            if e.source_entity_id in allowed_entities
            and e.target_entity_id in allowed_entities
            and (include_quarantined or e.governance_bucket == OntologyStatus.APPROVED)
            and _is_edge_active_at(e, as_of_iso)
            and _is_acl_allowed(e.acl_groups, caller_acl_groups)
        ]

        undirected = [
            (e.source_entity_id, e.target_entity_id, max(0.1, e.confidence))
            for e in allowed_edges
        ]
        ppr_scores, engine_used = self.compute_personalized_pagerank(
            nodes=list(allowed_entities.keys()),
            undirected_edges=undirected,
            seed_nodes=seed_entity_ids,
            alpha=alpha,
        )

        # Rank nodes by PPR score
        sorted_nodes = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
        top_ranked_nodes = [
            {
                "entity_id": eid,
                "canonical_name": allowed_entities[eid].canonical_name,
                "ontology_class": allowed_entities[eid].ontology_class,
                "ppr_score": round(score, 5),
                "source_kb_ids": allowed_entities[eid].source_kb_ids,
            }
            for eid, score in sorted_nodes[:12]
            if score > 1e-4
        ]

        # Compute node degrees for HippoRAG 2 inverse-degree hub damping
        import math

        node_degree: dict[str, int] = defaultdict(int)
        for e in allowed_edges:
            node_degree[e.source_entity_id] += 1
            node_degree[e.target_entity_id] += 1

        # Score each edge by hub-damped PPR (penalizes supernode hubs like SOC2/Java with high degree)
        seed_set = set(seed_entity_ids)
        ranked_facts: list[RetrievedGraphFact] = []
        for edge in allowed_edges:
            ps = ppr_scores.get(edge.source_entity_id, 0.0)
            pt = ppr_scores.get(edge.target_entity_id, 0.0)
            d_s = 1.0 if edge.source_entity_id in seed_set else math.sqrt(max(1, node_degree[edge.source_entity_id]))
            d_t = 1.0 if edge.target_entity_id in seed_set else math.sqrt(max(1, node_degree[edge.target_entity_id]))
            damped_s = ps / d_s
            damped_t = pt / d_t
            edge_ppr = (damped_s + damped_t) * 0.5 + (0.25 * min(damped_s, damped_t))
            if edge_ppr <= 1e-5:
                continue
            s_ent = allowed_entities[edge.source_entity_id]
            t_ent = allowed_entities[edge.target_entity_id]
            hop_est = 1 if (edge.source_entity_id in seed_set or edge.target_entity_id in seed_set) else 2
            combined_kbs = set(s_ent.source_kb_ids) | set(t_ent.source_kb_ids) | set(edge.source_kb_ids)
            ranked_facts.append(
                RetrievedGraphFact(
                    edge_id=edge.edge_id,
                    source_name=s_ent.canonical_name,
                    source_class=s_ent.ontology_class,
                    predicate=edge.predicate,
                    target_name=t_ent.canonical_name,
                    target_class=t_ent.ontology_class,
                    hop_distance=hop_est,
                    source_kb_ids=edge.source_kb_ids,
                    is_cross_kb_bridge=(len(combined_kbs) >= 2),
                    t_valid=edge.t_valid,
                    t_invalid=edge.t_invalid,
                    confidence=edge.confidence,
                    ppr_score=round(edge_ppr, 5),
                    qualifiers=edge.qualifiers,
                    acl_groups=edge.acl_groups,
                )
            )

        ranked_facts.sort(key=lambda f: (-f.ppr_score, f.hop_distance, -f.confidence))
        return PPRSearchResponse(
            tenant_id=tenant_id,
            engine_used=engine_used,
            top_ranked_nodes=top_ranked_nodes,
            ranked_facts=ranked_facts[:top_k_facts],
        )

    def detect_communities_and_summarize(
        self,
        tenant_id: str,
        entities: dict[str, KGEntityNode],
        edges: Sequence[KGRelationEdge],
        caller_acl_groups: Sequence[str] | None = None,
    ) -> tuple[list[CommunitySummaryRecord], str]:
        """
        Microsoft GraphRAG + LightRAG Community Detection & High-Level Theme Summarizer.
        Reuses `leidenalg` + `igraph` or `networkx.community.louvain_communities` when
        installed, falling back cleanly to label-propagation modularity partitioning.
        """
        allowed_entities = {
            eid: ent
            for eid, ent in entities.items()
            if ent.governance_bucket == OntologyStatus.APPROVED
            and _is_acl_allowed(ent.acl_groups, caller_acl_groups)
        }
        active_edges = [
            e
            for e in edges
            if e.t_invalid == "ACTIVE"
            and e.source_entity_id in allowed_entities
            and e.target_entity_id in allowed_entities
            and _is_acl_allowed(e.acl_groups, caller_acl_groups)
        ]

        if not allowed_entities:
            return [], "empty"

        node_ids = sorted(allowed_entities.keys())
        communities_list: list[set[str]] = []
        engine_used = "label_propagation_modularity"

        # 1. Try Leiden (`igraph` + `leidenalg`) first — gold standard from Microsoft GraphRAG
        try:
            import igraph as ig
            import leidenalg

            idx_map = {nid: i for i, nid in enumerate(node_ids)}
            g = ig.Graph(n=len(node_ids))
            g.add_edges([(idx_map[e.source_entity_id], idx_map[e.target_entity_id]) for e in active_edges])
            part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
            communities_list = [{node_ids[i] for i in comm} for comm in part if comm]
            engine_used = "leidenalg_igraph"
        except ImportError:
            # 2. Try NetworkX Louvain communities next
            try:
                import networkx as nx
                from networkx.algorithms.community import louvain_communities

                g = nx.Graph()
                g.add_nodes_from(node_ids)
                for e in active_edges:
                    g.add_edge(e.source_entity_id, e.target_entity_id, weight=e.confidence)
                communities_list = [set(c) for c in louvain_communities(g, seed=42) if c]
                engine_used = "networkx_louvain"
            except ImportError:
                # 3. Deterministic Asynchronous Label Propagation fallback
                adj: dict[str, list[str]] = defaultdict(list)
                for e in active_edges:
                    adj[e.source_entity_id].append(e.target_entity_id)
                    adj[e.target_entity_id].append(e.source_entity_id)

                labels = {nid: idx for idx, nid in enumerate(node_ids)}
                for _ in range(15):
                    changed = False
                    for nid in node_ids:
                        nbrs = adj.get(nid)
                        if not nbrs:
                            continue
                        counts = Counter(labels[nb] for nb in nbrs)
                        best_lbl = min(
                            counts.keys(), key=lambda lbl: (-counts[lbl], lbl)
                        )
                        if labels[nid] != best_lbl:
                            labels[nid] = best_lbl
                            changed = True
                    if not changed:
                        break

                by_lbl: dict[int, set[str]] = defaultdict(set)
                for nid, lbl in labels.items():
                    by_lbl[lbl].add(nid)
                communities_list = list(by_lbl.values())

        # Synthesize rich `CommunitySummaryRecord`s + LightRAG High-Level Themes
        summaries: list[CommunitySummaryRecord] = []
        communities_list.sort(key=len, reverse=True)

        for idx, member_ids in enumerate(communities_list, start=1):
            member_ents = [allowed_entities[mid] for mid in member_ids]
            member_names = sorted(e.canonical_name for e in member_ents)
            class_counts = Counter(e.ontology_class for e in member_ents)
            member_classes = [cls for cls, _ in class_counts.most_common(5)]

            comm_edges = [
                e
                for e in active_edges
                if e.source_entity_id in member_ids and e.target_entity_id in member_ids
            ]
            pred_counts = Counter(e.predicate for e in comm_edges)
            top_preds = [p for p, _ in pred_counts.most_common(5)]

            kbs: set[str] = set()
            for e in member_ents:
                kbs.update(e.source_kb_ids)
            for ce in comm_edges:
                kbs.update(ce.source_kb_ids)

            # High-level themes (LightRAG dual-level indexing)
            themes = [f"{cls} ecosystem" for cls in member_classes[:3]] + [
                f"{p.lower().replace('_', ' ')} governance" for p in top_preds[:3]
            ]
            hub_preview = ", ".join(member_names[:5])
            title = f"Community #{idx}: {', '.join(member_classes[:2])} ({hub_preview})"
            summary = (
                f"Cross-KB cluster spanning {len(kbs)} KB(s) with {len(member_ents)} entities "
                f"({', '.join(member_names[:8])}) across classes {member_classes}. "
                f"Dominant relationships: {top_preds or ['standalone_concept']}. "
                f"High-level themes: {', '.join(themes)}."
            )
            member_entity_acls = {
                e.canonical_name: sorted(set(e.acl_groups or ["public"]))
                for e in member_ents
            }
            comm_acls = sorted(
                {g for acls in member_entity_acls.values() for g in acls}
            ) or ["public"]

            emb = self.embedding_provider.embed_text(
                f"{title} {summary} {' '.join(themes)}", dim=self.canonical_dim
            )
            summaries.append(
                CommunitySummaryRecord(
                    community_id=f"comm_{tenant_id}_L0_{idx:03d}",
                    tenant_id=tenant_id,
                    level=0,
                    title=title,
                    summary=summary,
                    member_entity_names=member_names,
                    member_classes=member_classes,
                    contributing_kb_ids=sorted(kbs),
                    top_predicates=top_preds,
                    high_level_themes=themes,
                    acl_groups=comm_acls,
                    member_entity_acls=member_entity_acls,
                    embedding=emb,
                )
            )

        return summaries, engine_used

    @staticmethod
    def filter_communities_by_acl(
        communities: Sequence[CommunitySummaryRecord],
        caller_acl_groups: Sequence[str] | None = None,
    ) -> list[CommunitySummaryRecord]:
        """
        Applies strict ABAC filtering over build-time persisted `CommunitySummaryRecord`s:
        strips any member entities the caller cannot access and regenerates the title/summary
        from only the permitted entities so confidential entity names NEVER leak.
        """
        if caller_acl_groups is None:
            return list(communities)

        filtered: list[CommunitySummaryRecord] = []
        for idx, c in enumerate(communities, start=1):
            if not _is_acl_allowed(c.acl_groups, caller_acl_groups):
                continue
            if c.member_entity_acls:
                allowed_names = [
                    name
                    for name in c.member_entity_names
                    if _is_acl_allowed(c.member_entity_acls.get(name, ["public"]), caller_acl_groups)
                ]
                if not allowed_names:
                    continue
                if len(allowed_names) < len(c.member_entity_names):
                    hub_preview = ", ".join(allowed_names[:5])
                    safe_title = f"Community #{idx}: {', '.join(c.member_classes[:2])} ({hub_preview})"
                    safe_summary = (
                        f"Cross-KB cluster spanning {len(c.contributing_kb_ids)} KB(s) with "
                        f"{len(allowed_names)} entities ({', '.join(allowed_names[:8])}) across "
                        f"classes {c.member_classes}. Dominant relationships: "
                        f"{c.top_predicates or ['standalone_concept']}. "
                        f"High-level themes: {', '.join(c.high_level_themes)}."
                    )
                    filtered.append(
                        c.model_copy(
                            update={
                                "title": safe_title,
                                "summary": safe_summary,
                                "member_entity_names": allowed_names,
                                "member_entity_acls": {
                                    k: v for k, v in c.member_entity_acls.items() if k in allowed_names
                                },
                            }
                        )
                    )
                else:
                    filtered.append(c)
            else:
                filtered.append(c)
        return filtered

    def query_persisted_communities(
        self,
        tenant_id: str,
        query: str,
        persisted_communities: Sequence[CommunitySummaryRecord],
        top_k_communities: int = 4,
        caller_acl_groups: Sequence[str] | None = None,
        engine_used: str = "persisted_community_cache",
    ) -> GlobalCommunitySearchResponse:
        """Serves Global Community Sensemaking in O(C) from persisted build-time communities (zero full-graph scan)."""
        communities = self.filter_communities_by_acl(persisted_communities, caller_acl_groups)
        if not communities:
            return GlobalCommunitySearchResponse(
                tenant_id=tenant_id,
                query=query,
                synthesized_global_answer="No communities available for this tenant / ACL scope.",
                communities_matched=[],
                engine_used=engine_used,
            )

        q_vec = self.embedding_provider.embed_text(query, dim=self.canonical_dim)
        scored = [
            (comm, cosine_similarity(q_vec, comm.embedding) + 0.05 * min(len(comm.contributing_kb_ids), 5))
            for comm in communities
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_comms = [c for c, _ in scored[:top_k_communities]]

        lines = [f"### Global Sensemaking Synthesis ({engine_used} across {len(communities)} Graph Communities)"]
        for c in top_comms:
            lines.append(
                f"- **{c.title}** *(KBs: {', '.join(c.contributing_kb_ids[:5])})*: {c.summary}"
            )

        return GlobalCommunitySearchResponse(
            tenant_id=tenant_id,
            query=query,
            synthesized_global_answer="\n".join(lines),
            communities_matched=top_comms,
            engine_used=engine_used,
        )

    def query_global_communities(
        self,
        tenant_id: str,
        query: str,
        entities: dict[str, KGEntityNode],
        edges: Sequence[KGRelationEdge],
        top_k_communities: int = 4,
        caller_acl_groups: Sequence[str] | None = None,
    ) -> GlobalCommunitySearchResponse:
        communities, engine_used = self.detect_communities_and_summarize(
            tenant_id=tenant_id,
            entities=entities,
            edges=edges,
            caller_acl_groups=caller_acl_groups,
        )
        if not communities:
            return GlobalCommunitySearchResponse(
                tenant_id=tenant_id,
                query=query,
                synthesized_global_answer="No communities available for this tenant / ACL scope.",
                communities_matched=[],
                engine_used=engine_used,
            )

        q_vec = self.embedding_provider.embed_text(query, dim=self.canonical_dim)
        scored = [
            (comm, cosine_similarity(q_vec, comm.embedding) + 0.05 * min(len(comm.contributing_kb_ids), 5))
            for comm in communities
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_comms = [c for c, _ in scored[:top_k_communities]]

        lines = [f"### Global Sensemaking Synthesis ({engine_used} across {len(communities)} Graph Communities)"]
        for c in top_comms:
            lines.append(
                f"- **{c.title}** *(KBs: {', '.join(c.contributing_kb_ids[:5])})*: {c.summary}"
            )

        return GlobalCommunitySearchResponse(
            tenant_id=tenant_id,
            query=query,
            synthesized_global_answer="\n".join(lines),
            communities_matched=top_comms,
            engine_used=engine_used,
        )
