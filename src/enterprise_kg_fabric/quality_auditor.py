from __future__ import annotations

import json
import os
import re
from collections import deque
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from .entity_resolution import SiblingCollisionGuard
from .models import (
    EnterpriseCompiledTBox,
    ExtractionAndERQualityMetrics,
    GraphTopologyQualityMetrics,
    KGEntityNode,
    KGQualityScorecardReport,
    KGRelationEdge,
    OntologyStatus,
    OntologyTBoxQualityMetrics,
    QualityIssueFinding,
)
from .ontology_inducer import OWLSHACLCompiler
from .vector_clustering import EmbeddingProvider, cosine_similarity

if TYPE_CHECKING:
    from .service import EnterpriseKGOntologyService


def _score_to_grade(score: float) -> str:
    if score >= 95.0:
        return "A+"
    if score >= 90.0:
        return "A"
    if score >= 85.0:
        return "A-"
    if score >= 80.0:
        return "B+"
    if score >= 75.0:
        return "B"
    if score >= 65.0:
        return "C"
    if score >= 50.0:
        return "D"
    return "F"


class KGQualityAuditor:
    """
    Automated Enterprise Ontology (TBox) & Knowledge Graph (ABox) Quality Scorecard Engine.

    Evaluates 4 orthogonal structural & semantic quality dimensions without requiring manual labels
    (plus optional LLM-as-a-Judge triple faithfulness verification on sampled edges):
      1. Ontology / TBox Quality (25 pts):
         - W3C OWL 2 / SHACL conformance (`pyshacl`)
         - Schema Utilization Rate (% of approved classes/predicates actually instantiated)
         - Near-Duplicate Class Detection (embedding cosine >= 0.86 in same UpperOntologyCategory)
         - Domain / Range Compliance Rate across active ABox edges
      2. Extraction Fidelity & Groundedness (25 pts):
         - Zero-Extraction Chunk Rate (% of ingested chunks yielding zero entities/edges)
         - DLQ Failure Rate
         - Provenance Grounded Edge Rate (% of edges carrying `source_chunk_ids` + `evidence_snippets`)
         - Sampled Triple Faithfulness (deterministic evidence entailment or optional LLM-as-a-Judge)
      3. Entity Resolution Quality (20 pts):
         - Suspected Unmerged Duplicates (same ontology class, SiblingCollisionGuard-safe, high similarity)
         - Over-Merged Entity Suspects (excessive alias count or conflicting identifiers inside aliases)
      4. Graph Topology & Cross-KB Connectivity (30 pts):
         - Orphan Entity Rate (degree == 0)
         - Connected Components & Giant Connected Component (GCC) Ratio
         - Cross-KB Linkage Coverage (% of registered KBs linked into cross-KB bridges)
         - Hub Skew & Leiden/Louvain Community Modularity (Q)
    """

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9][a-z0-9_\-\.]+", text.lower()))

    @classmethod
    def _score_edge_faithfulness_deterministic(
        cls,
        edge: KGRelationEdge,
        entities: dict[str, KGEntityNode],
    ) -> float:
        """
        Deterministic entailment check verifying whether an edge's `evidence_snippets` actually
        mention its subject & object (or their aliases). Returns 0.0, 0.5, or 1.0.
        """
        if not edge.evidence_snippets:
            return 0.0
        snippet_blob = " ".join(edge.evidence_snippets).lower()
        src = entities.get(edge.source_entity_id)
        tgt = entities.get(edge.target_entity_id)
        if not src or not tgt:
            return 0.0

        src_names = [src.canonical_name.lower()] + [a.lower() for a in src.aliases]
        tgt_names = [tgt.canonical_name.lower()] + [a.lower() for a in tgt.aliases]

        src_hit = any(n in snippet_blob for n in src_names if len(n) >= 2)
        tgt_hit = any(n in snippet_blob for n in tgt_names if len(n) >= 2)
        if src_hit and tgt_hit:
            return 1.0
        if src_hit or tgt_hit:
            return 0.5
        return 0.0

    @classmethod
    def _score_edge_faithfulness_llm(
        cls,
        sampled_edges: list[KGRelationEdge],
        entities: dict[str, KGEntityNode],
    ) -> tuple[float, str] | None:
        """
        Optional live LLM-as-a-Judge triple faithfulness check using `google-genai` when
        `use_llm_judge=True` and `GEMINI_API_KEY` is configured. Returns `(score, mode)` or `None`.
        """
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or not sampled_edges:
            return None
        try:
            from google import genai  # type: ignore
        except ImportError:
            return None

        items = []
        for idx, edge in enumerate(sampled_edges[:15]):
            src = entities.get(edge.source_entity_id)
            tgt = entities.get(edge.target_entity_id)
            items.append(
                {
                    "id": idx,
                    "triple": f"({src.canonical_name if src else edge.source_entity_id}, {edge.predicate}, {tgt.canonical_name if tgt else edge.target_entity_id})",
                    "qualifiers": edge.qualifiers,
                    "evidence": " | ".join(edge.evidence_snippets[:2])[:400],
                }
            )
        prompt = (
            "You are a strict Knowledge Graph Faithfulness Auditor. For each item below, judge "
            "whether the evidence text entails the extracted triple (1.0=fully supported, 0.5=partial, 0.0=unsupported/hallucinated).\n"
            "Return ONLY a JSON object: {\"scores\": [float, ...]}.\n\n"
            f"Items:\n{json.dumps(items)}"
        )
        try:
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=os.getenv("KG_FABRIC_LLM_MODEL", "gemini-2.5-flash"),
                contents=prompt,
            )
            text = (resp.text or "").strip()
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                return None
            parsed = json.loads(m.group(0))
            scores = [max(0.0, min(1.0, float(s))) for s in parsed.get("scores", [])]
            if not scores:
                return None
            return round(sum(scores) / len(scores), 4), "llm_judge_gemini"
        except Exception:
            return None

    @staticmethod
    def _compute_connected_components_and_modularity(
        entity_ids: list[str],
        active_edges: list[KGRelationEdge],
        community_assignments: dict[str, str],
    ) -> tuple[int, float, float]:
        """Returns `(num_components, giant_component_ratio, modularity_q)`."""
        if not entity_ids:
            return 0, 0.0, 0.0

        adj: dict[str, set[str]] = {eid: set() for eid in entity_ids}
        undirected_edges: list[tuple[str, str]] = []
        for e in active_edges:
            if e.source_entity_id in adj and e.target_entity_id in adj:
                if e.source_entity_id != e.target_entity_id:
                    adj[e.source_entity_id].add(e.target_entity_id)
                    adj[e.target_entity_id].add(e.source_entity_id)
                    undirected_edges.append((e.source_entity_id, e.target_entity_id))

        # BFS connected components (works with zero external dependencies)
        visited: set[str] = set()
        comp_sizes: list[int] = []
        for start in entity_ids:
            if start in visited:
                continue
            q = deque([start])
            visited.add(start)
            size = 0
            while q:
                cur = q.popleft()
                size += 1
                for nb in adj[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            comp_sizes.append(size)

        num_components = len(comp_sizes)
        gcc_ratio = max(comp_sizes) / len(entity_ids) if comp_sizes else 0.0

        # Newman-Girvan Modularity Q for detected communities
        m = len(undirected_edges)
        if m == 0 or not community_assignments:
            return num_components, round(gcc_ratio, 4), 0.0

        two_m = 2.0 * m
        comm_internal_2m: dict[str, float] = {}
        comm_degree_sum: dict[str, float] = {}
        for nid in entity_ids:
            cid = community_assignments.get(nid, f"singleton_{nid}")
            comm_degree_sum[cid] = comm_degree_sum.get(cid, 0.0) + len(adj[nid])
        for u, v in undirected_edges:
            cu = community_assignments.get(u, f"singleton_{u}")
            cv = community_assignments.get(v, f"singleton_{v}")
            if cu == cv:
                comm_internal_2m[cu] = comm_internal_2m.get(cu, 0.0) + 2.0

        modularity = 0.0
        for cid, deg_sum in comm_degree_sum.items():
            e_ii = comm_internal_2m.get(cid, 0.0) / two_m
            a_i = deg_sum / two_m
            modularity += e_ii - (a_i * a_i)

        return num_components, round(gcc_ratio, 4), round(max(-0.5, min(1.0, modularity)), 4)

    @classmethod
    async def audit_tenant(
        cls,
        service: EnterpriseKGOntologyService,
        tenant_id: str = "enterprise_default",
        sample_size_faithfulness: int = 25,
        use_llm_judge: bool = False,
    ) -> KGQualityScorecardReport:
        tbox: EnterpriseCompiledTBox = await service.load_or_get_tbox(tenant_id)
        entities: dict[str, KGEntityNode] = await service.graph_store.get_all_entities(tenant_id)
        active_edges: list[KGRelationEdge] = await service.graph_store.get_all_edges(
            tenant_id, active_only=True
        )
        all_edges: list[KGRelationEdge] = await service.graph_store.get_all_edges(
            tenant_id, active_only=False
        )
        registered_kbs = await service.vector_store.list_registered_kbs(tenant_id=tenant_id)
        dlq_records = await service.list_dlq_records_async(tenant_id=tenant_id)

        all_chunk_ids: set[str] = set()
        for kb in registered_kbs:
            kb_chunks = await service.vector_store.fetch_kb_chunks(kb.kb_id, tenant_id=tenant_id)
            for chk in kb_chunks:
                if chk.tenant_id == tenant_id:
                    all_chunk_ids.add(chk.chunk_id)

        findings: list[QualityIssueFinding] = []
        emb: EmbeddingProvider = service.embedding_provider

        # =====================================================================
        # 1. ONTOLOGY (TBox) QUALITY METRICS
        # =====================================================================
        approved_classes = [
            c for c in tbox.classes.values() if c.status == OntologyStatus.APPROVED
        ]
        quarantined_classes = [
            c for c in tbox.classes.values() if c.status == OntologyStatus.QUARANTINED_CANDIDATE
        ]
        approved_relations = [
            r for r in tbox.relations.values() if r.status == OntologyStatus.APPROVED
        ]
        quarantined_relations = [
            r for r in tbox.relations.values() if r.status == OntologyStatus.QUARANTINED_CANDIDATE
        ]

        # W3C OWL 2 / SHACL Conformance
        shacl_conforms = True
        shacl_violations = 0
        if entities:
            shacl_conforms, shacl_report = OWLSHACLCompiler.validate_abox_with_shacl(
                tbox=tbox,
                entities=list(entities.values()),
                edges=active_edges,
            )
            if not shacl_conforms:
                shacl_violations = max(1, shacl_report.lower().count("constraint violation"))
                findings.append(
                    QualityIssueFinding(
                        severity="HIGH",
                        category="TBOX",
                        message=f"W3C SHACL validation failed with {shacl_violations} violation(s).",
                        remediation="Inspect GET /v1/ontology/export.ttl and align missing entity/edge properties.",
                    )
                )

        # Schema Utilization Rate
        used_classes = {e.ontology_class for e in entities.values()}
        used_relations = {e.predicate for e in active_edges}
        unused_classes = sorted(
            c.class_name for c in approved_classes if c.class_name not in used_classes
        )
        unused_relations = sorted(
            r.predicate for r in approved_relations if r.predicate not in used_relations
        )
        total_approved_schema = len(approved_classes) + len(approved_relations)
        used_approved_schema = (len(approved_classes) - len(unused_classes)) + (
            len(approved_relations) - len(unused_relations)
        )
        schema_util = (
            round(used_approved_schema / total_approved_schema, 4)
            if total_approved_schema > 0
            else 1.0
        )
        if unused_classes or unused_relations:
            findings.append(
                QualityIssueFinding(
                    severity="LOW" if schema_util >= 0.70 else "MEDIUM",
                    category="TBOX",
                    message=(
                        f"{len(unused_classes)} approved class(es) and {len(unused_relations)} "
                        f"approved predicate(s) have 0 ABox instances (utilization={schema_util:.0%})."
                    ),
                    affected_ids=(unused_classes + unused_relations)[:10],
                    remediation="Prune dead schema terms or check extraction prompts for missing coverage.",
                )
            )

        # Near-Duplicate Approved Classes (within same UpperOntologyCategory)
        near_dup_classes: list[dict[str, Any]] = []
        cls_vecs = {
            c.class_name: emb.embed_text(f"{c.class_name} {c.description}", dim=service.canonical_dim)
            for c in approved_classes
        }
        for i in range(len(approved_classes)):
            for j in range(i + 1, len(approved_classes)):
                ca, cb = approved_classes[i], approved_classes[j]
                if ca.upper_category != cb.upper_category:
                    continue
                sim = cosine_similarity(cls_vecs[ca.class_name], cls_vecs[cb.class_name])
                seq_sim = SequenceMatcher(None, ca.class_name.lower(), cb.class_name.lower()).ratio()
                if sim >= 0.88 or seq_sim >= 0.85:
                    near_dup_classes.append(
                        {
                            "class_a": ca.class_name,
                            "class_b": cb.class_name,
                            "upper_category": ca.upper_category.value,
                            "similarity": round(max(sim, seq_sim), 4),
                        }
                    )
        if near_dup_classes:
            findings.append(
                QualityIssueFinding(
                    severity="MEDIUM",
                    category="TBOX",
                    message=f"Detected {len(near_dup_classes)} near-duplicate approved TBox class pair(s).",
                    affected_ids=[f"{p['class_a']}<->{p['class_b']}" for p in near_dup_classes[:5]],
                    remediation="Merge redundant classes via synonym alignment in the TBox.",
                )
            )

        # Domain / Range Compliance across active ABox edges
        dr_violations = 0
        dr_violating_ids: list[str] = []
        for edge in active_edges:
            rel_def = tbox.relations.get(edge.predicate)
            if not rel_def:
                continue
            src = entities.get(edge.source_entity_id)
            tgt = entities.get(edge.target_entity_id)
            if not src or not tgt:
                dr_violations += 1
                dr_violating_ids.append(edge.edge_id)
                continue
            dom_ok = (not rel_def.domain_classes) or (src.ontology_class in rel_def.domain_classes)
            rng_ok = (not rel_def.range_classes) or (tgt.ontology_class in rel_def.range_classes)
            if not (dom_ok and rng_ok):
                dr_violations += 1
                dr_violating_ids.append(edge.edge_id)

        dr_compliance = (
            round(1.0 - (dr_violations / len(active_edges)), 4) if active_edges else 1.0
        )
        if dr_violations > 0:
            findings.append(
                QualityIssueFinding(
                    severity="HIGH" if dr_compliance < 0.90 else "MEDIUM",
                    category="TBOX",
                    message=(
                        f"{dr_violations}/{len(active_edges)} active edge(s) violate TBox "
                        f"domain/range constraints (compliance={dr_compliance:.1%})."
                    ),
                    affected_ids=dr_violating_ids[:10],
                    remediation="Broaden TBox relation domain/range or tighten extractor entity typing.",
                )
            )

        tbox_metrics = OntologyTBoxQualityMetrics(
            tbox_version=tbox.version,
            approved_classes=len(approved_classes),
            quarantined_classes=len(quarantined_classes),
            approved_relations=len(approved_relations),
            quarantined_relations=len(quarantined_relations),
            shacl_conforms=shacl_conforms,
            shacl_violation_count=shacl_violations,
            schema_utilization_rate=schema_util,
            unused_classes=unused_classes,
            unused_relations=unused_relations,
            near_duplicate_class_pairs=near_dup_classes,
            domain_range_compliance_rate=dr_compliance,
            domain_range_violations_count=dr_violations,
        )

        # =====================================================================
        # 2. EXTRACTION FIDELITY & ENTITY RESOLUTION QUALITY METRICS
        # =====================================================================
        referenced_chunks: set[str] = set()
        for ent in entities.values():
            referenced_chunks.update(ent.source_chunk_ids)
        for edge in all_edges:
            referenced_chunks.update(edge.source_chunk_ids)

        zero_ext_chunks = sorted(all_chunk_ids - referenced_chunks)
        zero_ext_rate = (
            round(len(zero_ext_chunks) / len(all_chunk_ids), 4) if all_chunk_ids else 0.0
        )
        if zero_ext_chunks and zero_ext_rate > 0.25:
            findings.append(
                QualityIssueFinding(
                    severity="MEDIUM" if zero_ext_rate < 0.50 else "HIGH",
                    category="EXTRACTION",
                    message=(
                        f"{len(zero_ext_chunks)}/{len(all_chunk_ids)} ingested chunk(s) "
                        f"({zero_ext_rate:.0%}) yielded 0 extracted entities or edges."
                    ),
                    affected_ids=zero_ext_chunks[:10],
                    remediation="Enable KG_FABRIC_USE_LIVE_LLM=true or register a custom DomainRulePack.",
                )
            )

        if dlq_records:
            findings.append(
                QualityIssueFinding(
                    severity="MEDIUM",
                    category="EXTRACTION",
                    message=f"{len(dlq_records)} chunk(s) are currently isolated in the Dead-Letter Queue (DLQ).",
                    affected_ids=[r.chunk_id for r in dlq_records[:10]],
                    remediation="Inspect GET /v1/jobs/dlq and re-run incremental sync once extractor errors resolve.",
                )
            )

        grounded_edges = [
            e for e in active_edges if bool(e.source_chunk_ids) and bool(e.evidence_snippets)
        ]
        provenance_rate = (
            round(len(grounded_edges) / len(active_edges), 4) if active_edges else 1.0
        )
        if active_edges and provenance_rate < 0.95:
            ungrounded = [
                e.edge_id
                for e in active_edges
                if not (e.source_chunk_ids and e.evidence_snippets)
            ]
            findings.append(
                QualityIssueFinding(
                    severity="HIGH" if provenance_rate < 0.80 else "MEDIUM",
                    category="EXTRACTION",
                    message=(
                        f"{len(ungrounded)}/{len(active_edges)} active edge(s) lack chunk provenance "
                        f"or evidence snippets (provenance={provenance_rate:.1%})."
                    ),
                    affected_ids=ungrounded[:10],
                    remediation="Ensure all extractors attach `source_chunk_ids` and `evidence_snippets` to every edge.",
                )
            )

        # Sampled Triple Faithfulness Verification
        sampled_edges = active_edges[: max(1, sample_size_faithfulness)] if active_edges else []
        faithfulness_mode = "deterministic_evidence_entailment"
        faithfulness_score = 1.0
        if sampled_edges:
            llm_verdict = (
                cls._score_edge_faithfulness_llm(sampled_edges, entities)
                if use_llm_judge
                else None
            )
            if llm_verdict is not None:
                faithfulness_score, faithfulness_mode = llm_verdict
            else:
                det_scores = [
                    cls._score_edge_faithfulness_deterministic(e, entities)
                    for e in sampled_edges
                ]
                faithfulness_score = round(sum(det_scores) / len(det_scores), 4)
        if sampled_edges and faithfulness_score < 0.85:
            findings.append(
                QualityIssueFinding(
                    severity="HIGH" if faithfulness_score < 0.70 else "MEDIUM",
                    category="EXTRACTION",
                    message=(
                        f"Sampled triple faithfulness score is {faithfulness_score:.1%} "
                        f"across {len(sampled_edges)} edge(s) ({faithfulness_mode})."
                    ),
                    remediation="Review extracted triples against source chunks to eliminate hallucinated relations.",
                )
            )

        # Entity Resolution: Suspected Unmerged Duplicates (capped per class for O(n) safety)
        suspected_dups: list[dict[str, Any]] = []
        entities_by_class: dict[str, list[KGEntityNode]] = {}
        for ent in entities.values():
            entities_by_class.setdefault(ent.ontology_class, []).append(ent)

        for cls_name, cls_ents in entities_by_class.items():
            capped = cls_ents[:150]
            for i in range(len(capped)):
                for j in range(i + 1, len(capped)):
                    ea, eb = capped[i], capped[j]
                    can_merge, _ = SiblingCollisionGuard.can_merge(
                        ea.canonical_name, eb.canonical_name
                    )
                    if not can_merge:
                        continue
                    na = re.sub(r"[^a-z0-9]+", " ", ea.canonical_name.lower()).strip()
                    nb = re.sub(r"[^a-z0-9]+", " ", eb.canonical_name.lower()).strip()
                    str_sim = SequenceMatcher(None, na, nb).ratio()
                    # Also check token containment (e.g. "acme bank" vs "acme global bank")
                    ta, tb = set(na.split()), set(nb.split())
                    containment = (
                        len(ta & tb) / min(len(ta), len(tb))
                        if ta and tb and min(len(ta), len(tb)) >= 2
                        else 0.0
                    )
                    vec_sim = (
                        cosine_similarity(ea.embedding, eb.embedding)
                        if (ea.embedding and eb.embedding and len(ea.embedding) == len(eb.embedding))
                        else 0.0
                    )
                    best_sim = max(str_sim, vec_sim, 0.86 if containment >= 1.0 else 0.0)
                    if best_sim >= 0.82:
                        suspected_dups.append(
                            {
                                "entity_a": ea.canonical_name,
                                "entity_b": eb.canonical_name,
                                "ontology_class": cls_name,
                                "similarity": round(best_sim, 4),
                            }
                        )
                        if len(suspected_dups) >= 20:
                            break
                if len(suspected_dups) >= 20:
                    break

        if suspected_dups:
            findings.append(
                QualityIssueFinding(
                    severity="MEDIUM" if len(suspected_dups) <= 3 else "HIGH",
                    category="ENTITY_RESOLUTION",
                    message=f"Found {len(suspected_dups)} suspected unmerged duplicate entity pair(s).",
                    affected_ids=[f"{d['entity_a']}<->{d['entity_b']}" for d in suspected_dups[:5]],
                    remediation="Lower CrossKBEntityResolver similarity threshold or add domain aliases.",
                )
            )

        # Over-Merged Suspects (entities accumulating suspiciously many distinct aliases or conflicting numbers)
        over_merged: list[dict[str, Any]] = []
        for ent in entities.values():
            distinct_aliases = sorted({a for a in ent.aliases if a.lower() != ent.canonical_name.lower()})
            conflict_reason = None
            if len(distinct_aliases) >= 6:
                conflict_reason = f"High alias fan-in ({len(distinct_aliases)} aliases)"
            else:
                for al in distinct_aliases:
                    ok, reason = SiblingCollisionGuard.can_merge(ent.canonical_name, al)
                    if not ok:
                        conflict_reason = f"Sibling conflict with alias '{al}': {reason}"
                        break
            if conflict_reason:
                over_merged.append(
                    {
                        "entity_id": ent.entity_id,
                        "canonical_name": ent.canonical_name,
                        "alias_count": len(distinct_aliases),
                        "reason": conflict_reason,
                    }
                )
        if over_merged:
            findings.append(
                QualityIssueFinding(
                    severity="HIGH",
                    category="ENTITY_RESOLUTION",
                    message=f"Detected {len(over_merged)} potentially over-merged entity node(s).",
                    affected_ids=[o["canonical_name"] for o in over_merged[:5]],
                    remediation="Review alias lists and verify SiblingCollisionGuard rules for this domain.",
                )
            )

        ext_er_metrics = ExtractionAndERQualityMetrics(
            total_chunks_evaluated=len(all_chunk_ids),
            zero_extraction_chunk_count=len(zero_ext_chunks),
            zero_extraction_chunk_rate=zero_ext_rate,
            dlq_failed_chunks_count=len(dlq_records),
            provenance_grounded_edge_rate=provenance_rate,
            sampled_edges_checked=len(sampled_edges),
            sampled_triple_faithfulness_score=faithfulness_score,
            faithfulness_judge_mode=faithfulness_mode,
            suspected_unmerged_duplicate_pairs=suspected_dups,
            over_merged_suspects=over_merged,
        )

        # =====================================================================
        # 3. GRAPH TOPOLOGY & CROSS-KB CONNECTIVITY METRICS
        # =====================================================================
        degree: dict[str, int] = {eid: 0 for eid in entities}
        for edge in active_edges:
            if edge.source_entity_id in degree:
                degree[edge.source_entity_id] += 1
            if edge.target_entity_id in degree:
                degree[edge.target_entity_id] += 1

        orphan_ids = [eid for eid, deg in degree.items() if deg == 0]
        orphan_rate = round(len(orphan_ids) / len(entities), 4) if entities else 0.0
        if orphan_ids and orphan_rate > 0.15:
            findings.append(
                QualityIssueFinding(
                    severity="MEDIUM" if orphan_rate < 0.35 else "HIGH",
                    category="TOPOLOGY",
                    message=(
                        f"{len(orphan_ids)}/{len(entities)} canonical entities ({orphan_rate:.0%}) "
                        "are orphan nodes with degree 0."
                    ),
                    affected_ids=[entities[eid].canonical_name for eid in orphan_ids[:10]],
                    remediation="Improve relation extraction recall or prune unreferenced isolated nodes.",
                )
            )

        # Communities & Modularity via GraphAnalyticsEngine
        communities, _comm_engine = service.graph_analytics.detect_communities_and_summarize(
            tenant_id=tenant_id,
            entities=entities,
            edges=active_edges,
        )
        name_to_eid = {ent.canonical_name: eid for eid, ent in entities.items()}
        comm_map: dict[str, str] = {}
        for comm in communities:
            for ename in comm.member_entity_names:
                eid = name_to_eid.get(ename)
                if eid:
                    comm_map[eid] = comm.community_id

        num_components, gcc_ratio, modularity_q = cls._compute_connected_components_and_modularity(
            entity_ids=list(entities.keys()),
            active_edges=active_edges,
            community_assignments=comm_map,
        )

        # Cross-KB Linkage Coverage
        kb_ids_registered = {kb.kb_id for kb in registered_kbs}
        connected_kbs: set[str] = set()
        cross_kb_bridges: list[str] = []
        for ent in entities.values():
            if len(ent.source_kb_ids) >= 2:
                cross_kb_bridges.append(ent.canonical_name)
                connected_kbs.update(ent.source_kb_ids)
        for edge in active_edges:
            s_ent = entities.get(edge.source_entity_id)
            t_ent = entities.get(edge.target_entity_id)
            kbs_touched = set(edge.source_kb_ids)
            if s_ent:
                kbs_touched.update(s_ent.source_kb_ids)
            if t_ent:
                kbs_touched.update(t_ent.source_kb_ids)
            if len(kbs_touched) >= 2:
                connected_kbs.update(kbs_touched)

        if len(kb_ids_registered) >= 2:
            cross_kb_coverage = round(
                len(connected_kbs & kb_ids_registered) / len(kb_ids_registered), 4
            )
        elif len(kb_ids_registered) == 1:
            cross_kb_coverage = 1.0 if entities else 0.0
        else:
            cross_kb_coverage = 1.0 if cross_kb_bridges else 0.0

        if len(kb_ids_registered) >= 2 and cross_kb_coverage < 0.70:
            isolated_kbs = sorted(kb_ids_registered - connected_kbs)
            findings.append(
                QualityIssueFinding(
                    severity="MEDIUM",
                    category="TOPOLOGY",
                    message=(
                        f"Only {cross_kb_coverage:.0%} of registered KBs are linked across silos "
                        f"({len(isolated_kbs)} isolated KB(s))."
                    ),
                    affected_ids=isolated_kbs[:10],
                    remediation="Check cross-KB entity naming conventions or synonym alignment rules.",
                )
            )

        total_degree = sum(degree.values())
        avg_degree = round(total_degree / len(entities), 3) if entities else 0.0
        sorted_hubs = sorted(degree.items(), key=lambda kv: kv[1], reverse=True)
        top_hub_share = (
            round(sorted_hubs[0][1] / total_degree, 4) if (sorted_hubs and total_degree > 0) else 0.0
        )
        top_hubs = [
            {
                "canonical_name": entities[eid].canonical_name,
                "ontology_class": entities[eid].ontology_class,
                "degree": deg,
            }
            for eid, deg in sorted_hubs[:5]
            if deg > 0
        ]

        topo_metrics = GraphTopologyQualityMetrics(
            canonical_entities_count=len(entities),
            active_edges_count=len(active_edges),
            superseded_edges_count=max(0, len(all_edges) - len(active_edges)),
            orphan_entity_count=len(orphan_ids),
            orphan_entity_rate=orphan_rate,
            connected_components_count=num_components,
            giant_component_ratio=gcc_ratio,
            registered_kbs_count=len(kb_ids_registered),
            connected_kbs_count=len(connected_kbs & kb_ids_registered) if kb_ids_registered else len(connected_kbs),
            cross_kb_linkage_coverage=cross_kb_coverage,
            cross_kb_bridge_entity_count=len(cross_kb_bridges),
            average_degree=avg_degree,
            top_hub_degree_share=top_hub_share,
            top_hubs=top_hubs,
            community_count=len(communities),
            community_modularity=modularity_q,
        )

        # =====================================================================
        # 4. COMPOSITE WEIGHTED 0-100 QUALITY SCORECARD
        # =====================================================================
        if not entities and not tbox.classes:
            overall = 0.0
            dim_scores = {
                "ontology_tbox": 0.0,
                "extraction_fidelity": 0.0,
                "entity_resolution": 0.0,
                "graph_topology": 0.0,
            }
            findings.append(
                QualityIssueFinding(
                    severity="HIGH",
                    category="GOVERNANCE",
                    message="Tenant has no induced ontology classes or graph entities yet.",
                    remediation="Ingest KB chunks/documents, induce ontology, and build the knowledge graph.",
                )
            )
        else:
            # Ontology / TBox (25 pts)
            shacl_pts = 10.0 if shacl_conforms else max(0.0, 10.0 - 3.0 * shacl_violations)
            util_pts = 7.0 * schema_util
            dr_pts = 8.0 * dr_compliance
            dup_cls_penalty = min(5.0, 2.5 * len(near_dup_classes))
            tbox_score = max(0.0, round(shacl_pts + util_pts + dr_pts - dup_cls_penalty, 2))

            # Extraction Fidelity (25 pts)
            prov_pts = 8.0 * provenance_rate
            faith_pts = 10.0 * faithfulness_score
            yield_pts = 5.0 * (1.0 - zero_ext_rate)
            dlq_rate = (len(dlq_records) / max(1, len(all_chunk_ids))) if all_chunk_ids else 0.0
            dlq_pts = 2.0 * max(0.0, 1.0 - min(1.0, dlq_rate * 2.0))
            ext_score = max(0.0, round(prov_pts + faith_pts + yield_pts + dlq_pts, 2))

            # Entity Resolution (20 pts)
            dup_rate = len(suspected_dups) / max(1, len(entities))
            over_rate = len(over_merged) / max(1, len(entities))
            er_score = max(
                0.0,
                round(20.0 * max(0.0, 1.0 - min(1.0, dup_rate * 3.0) - min(1.0, over_rate * 4.0)), 2),
            )

            # Graph Topology & Cross-KB Connectivity (30 pts)
            orphan_pts = 10.0 * (1.0 - orphan_rate)
            gcc_pts = 8.0 * gcc_ratio
            bridge_pts = 9.0 * cross_kb_coverage
            mod_pts = 3.0 * (1.0 if modularity_q >= 0.15 or len(entities) < 6 else max(0.3, modularity_q + 0.5))
            topo_score = max(0.0, round(orphan_pts + gcc_pts + bridge_pts + mod_pts, 2))

            overall = round(min(100.0, tbox_score + ext_score + er_score + topo_score), 2)
            dim_scores = {
                "ontology_tbox": tbox_score,
                "extraction_fidelity": ext_score,
                "entity_resolution": er_score,
                "graph_topology": topo_score,
            }

        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
        findings.sort(key=lambda f: severity_order.get(f.severity, 9))

        return KGQualityScorecardReport(
            tenant_id=tenant_id,
            overall_score=overall,
            grade=_score_to_grade(overall),
            dimension_scores=dim_scores,
            tbox_metrics=tbox_metrics,
            extraction_er_metrics=ext_er_metrics,
            topology_metrics=topo_metrics,
            findings=findings,
        )
