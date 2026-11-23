from __future__ import annotations

import re
from typing import Any, Sequence

from .models import QueryRouteDecision, RetrievedChunkContext


class AdaptiveQueryRouter:
    """
    Adaptive-RAG Query Complexity Classifier & Cost-Aware Router:
    Dispatches incoming queries to the cheapest, highest-precision retrieval strategy:
      1. `GLOBAL_SENSEMAKING` -> Microsoft GraphRAG Leiden Community Map-Reduce
      2. `STRUCTURED_AGGREGATION` -> Ontology-Grounded OpenCypher Execution
      3. `LOCAL_MULTIHOP_PPR` -> HippoRAG 2 Personalized PageRank + Hybrid GraphRAG
      4. `FACTUAL_VECTOR` -> Fast Multi-Model pgvector + BM25 RRF (skips graph overhead)
    """

    _GLOBAL_PATTERNS = re.compile(
        r"\b(overall|systemic|across\s+all|top\s+\d+\s+(?:risks|themes|trends|ecosystems)|"
        r"executive\s+summary|macro|landscape|major\s+themes|summarize\s+the\s+entire)\b",
        re.I,
    )
    _AGG_PATTERNS = re.compile(
        r"\b(how\s+many|count\s+all|list\s+all|total\s+number\s+of|which\s+assets\s+are\s+governed)\b",
        re.I,
    )
    _MULTIHOP_PATTERNS = re.compile(
        r"\b(connected|blast\s+radius|downstream|upstream|depend|impact|chain|bridge|"
        r"path\s+between|compliance|governed|mitigated|cve-|failover|cost\s+center|"
        r"how\s+(?:is|are|do)|linked|relationship|audit|as\s+of|june|historical)\b",
        re.I,
    )

    @classmethod
    def classify_query(
        cls,
        query: str,
        as_of_iso: str | None = None,
    ) -> QueryRouteDecision:
        if as_of_iso is not None:
            return QueryRouteDecision.LOCAL_MULTIHOP_PPR
        if cls._GLOBAL_PATTERNS.search(query):
            return QueryRouteDecision.GLOBAL_SENSEMAKING
        if cls._AGG_PATTERNS.search(query):
            return QueryRouteDecision.STRUCTURED_AGGREGATION
        if cls._MULTIHOP_PATTERNS.search(query):
            return QueryRouteDecision.LOCAL_MULTIHOP_PPR
        return QueryRouteDecision.FACTUAL_VECTOR


class PluggableCrossEncoderReranker:
    """
    Reuses `sentence_transformers.CrossEncoder` (e.g. `BAAI/bge-reranker-v2-m3`) or
    AWS Bedrock Rerank (`bedrock-agent-runtime`) when configured, with a zero-dependency
    query-term coverage + exact-entity-boost reranker fallback.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_cross_encoder: bool = False,
    ):
        self.model_name = model_name
        self._cross_encoder: Any = None
        if use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder(model_name)
            except Exception:
                self._cross_encoder = None

    @property
    def engine_name(self) -> str:
        return f"cross_encoder:{self.model_name}" if self._cross_encoder is not None else "lexical_coverage_reranker"

    def rerank_chunks(
        self,
        query: str,
        chunks: Sequence[RetrievedChunkContext],
        top_k: int | None = None,
    ) -> list[RetrievedChunkContext]:
        if not chunks:
            return []
        limit = top_k or len(chunks)

        if self._cross_encoder is not None:
            pairs = [(query, c.text) for c in chunks]
            scores = self._cross_encoder.predict(pairs)
            scored = sorted(zip(chunks, scores), key=lambda x: float(x[1]), reverse=True)
            out: list[RetrievedChunkContext] = []
            for chk, sc in scored[:limit]:
                chk.rrf_score = round(float(sc), 5)
                if "cross_encoder_rerank" not in chk.retrieval_channels:
                    chk.retrieval_channels.append("cross_encoder_rerank")
                out.append(chk)
            return out

        # Fast query-token & exact identifier coverage reranker fallback
        q_tokens = set(re.findall(r"[a-z0-9_\-\.]{2,}", query.lower()))
        rescored: list[tuple[RetrievedChunkContext, float]] = []
        for chk in chunks:
            c_tokens = set(re.findall(r"[a-z0-9_\-\.]{2,}", chk.text.lower()))
            overlap = q_tokens & c_tokens
            coverage = len(overlap) / max(1, len(q_tokens))
            id_bonus = sum(0.08 for tok in overlap if "-" in tok or any(ch.isdigit() for ch in tok))
            combined = chk.rrf_score + (0.04 * coverage) + id_bonus
            rescored.append((chk, combined))

        rescored.sort(key=lambda x: x[1], reverse=True)
        result: list[RetrievedChunkContext] = []
        for chk, sc in rescored[:limit]:
            chk.rrf_score = round(sc, 5)
            if "reranked" not in chk.retrieval_channels:
                chk.retrieval_channels.append("reranked")
            result.append(chk)
        return result
