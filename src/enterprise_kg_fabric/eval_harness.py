from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from .models import EvalSuiteReport, EvalTestCase, GraphRAGQueryRequest

if TYPE_CHECKING:
    from .service import EnterpriseKGOntologyService


class GraphRAGEvaluationHarness:
    """
    Continuous Enterprise GraphRAG Evaluation Harness measuring:
      1. Multi-Hop Fact Recall (`expected_facts` found across Graph + Vector evidence)
      2. ABAC & Temporal Zero-Leakage Rate (`forbidden_facts` must NEVER appear)
      3. Citation Groundedness Rate (every synthesized answer cites contributing KBs / chunks)
      4. Optional integration with the **`ragas`** evaluation framework when installed.
    """

    @staticmethod
    def ragas_available() -> bool:
        try:
            import ragas  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    async def evaluate_suite(
        cls,
        service: EnterpriseKGOntologyService,
        tenant_id: str,
        cases: Sequence[EvalTestCase],
    ) -> EvalSuiteReport:
        if not cases:
            return EvalSuiteReport(
                tenant_id=tenant_id,
                cases_evaluated=0,
                mean_multihop_recall=1.0,
                abac_zero_leakage_rate=1.0,
                citation_groundedness_rate=1.0,
                ragas_available=cls.ragas_available(),
            )

        recalls: list[float] = []
        leak_free_count = 0
        grounded_count = 0
        details: list[dict[str, Any]] = []

        for tc in cases:
            resp = await service.query_graph_rag(
                GraphRAGQueryRequest(
                    tenant_id=tenant_id,
                    query=tc.query,
                    caller_acl_groups=tc.caller_acl_groups,
                    as_of_iso=tc.as_of_iso,
                    max_graph_hops=3,
                    use_adaptive_routing=True,
                )
            )
            evidence_blob = " ".join(
                [resp.synthesized_answer]
                + [f"{gf.source_name} {gf.predicate} {gf.target_name}" for gf in resp.graph_facts]
                + [c.text for c in resp.chunk_contexts]
            ).lower()

            found = [f for f in tc.expected_facts if f.lower() in evidence_blob]
            recall = len(found) / len(tc.expected_facts) if tc.expected_facts else 1.0
            recalls.append(recall)

            leaked = [f for f in tc.forbidden_facts if f.lower() in evidence_blob]
            if not leaked:
                leak_free_count += 1

            has_citations = bool(resp.provenance_kbs_contributing or resp.community_summaries_used)
            if has_citations:
                grounded_count += 1

            details.append(
                {
                    "query": tc.query,
                    "route_taken": resp.route_taken.value,
                    "recall": round(recall, 4),
                    "expected_found": found,
                    "expected_missing": [f for f in tc.expected_facts if f not in found],
                    "forbidden_leaked": leaked,
                    "grounded_with_citations": has_citations,
                    "latency_ms": resp.retrieval_latency_ms,
                }
            )

        return EvalSuiteReport(
            tenant_id=tenant_id,
            cases_evaluated=len(cases),
            mean_multihop_recall=round(sum(recalls) / len(recalls), 4),
            abac_zero_leakage_rate=round(leak_free_count / len(cases), 4),
            citation_groundedness_rate=round(grounded_count / len(cases), 4),
            ragas_available=cls.ragas_available(),
            case_details=details,
        )
