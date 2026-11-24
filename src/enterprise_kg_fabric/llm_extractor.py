from __future__ import annotations

import os
import re
from typing import Any

from pydantic import BaseModel, Field

from .llama_index_bridge import LlamaIndexSchemaAndDynamicExtractor
from .models import (
    ChunkExtractionResult,
    EnterpriseCompiledTBox,
    ExtractedEntityMention,
    ExtractedRelationTriple,
    VectorChunkRecord,
)

# Reuse AsyncCircuitBreaker from the user's published `agentic-composite-memory` SDK when installed
try:
    from agentic_memory_sdk.governance import AsyncCircuitBreaker
except ImportError:
    import time

    class AsyncCircuitBreaker:  # type: ignore[no-redef]
        def __init__(self, failure_threshold: int = 5, recovery_timeout_sec: float = 30.0):
            self.failure_threshold = failure_threshold
            self.recovery_timeout_sec = recovery_timeout_sec
            self.failures = 0
            self.opened_at: float | None = None

        @property
        def state(self) -> str:
            if self.opened_at is None:
                return "CLOSED"
            if (time.monotonic() - self.opened_at) >= self.recovery_timeout_sec:
                return "HALF_OPEN"
            return "OPEN"

        def record_success(self) -> None:
            self.failures = 0
            self.opened_at = None

        def record_failure(self) -> None:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = time.monotonic()


class _LLMStructuredTriple(BaseModel):
    subject: str
    subject_class: str
    predicate: str
    object: str
    object_class: str
    confidence: float = 0.90
    qualifiers: dict[str, Any] = Field(default_factory=dict)


class _LLMStructuredExtractionPayload(BaseModel):
    triples: list[_LLMStructuredTriple] = Field(default_factory=list)


class StructuredLLMExtractorAdapter:
    """
    Reuses best-in-class Structured-Output LLM SDKs (`google-genai`, `boto3` Bedrock Converse,
    `openai`) with constrained Pydantic JSON schemas + `AsyncCircuitBreaker` protection,
    and automatically enriches extracted triples with N-ary reified claim qualifiers
    (`condition`, `approved_by`, `negated`).

    Falls back gracefully to `LlamaIndexSchemaAndDynamicExtractor` when no live LLM API key
    is configured or if the circuit breaker opens.
    """

    _COND_QUALIFIER_RE = re.compile(
        r"\b(?:only\s+if|when|if|provided\s+that)\s+([^,.;\n]{4,70})", re.I
    )
    _APPROVER_QUALIFIER_RE = re.compile(
        r"\b(?:approved\s+by|authorized\s+by|signed\s+off\s+by)\s+([A-Za-z0-9_\-\s]{3,40})",
        re.I,
    )

    def __init__(
        self,
        provider: str = "auto",  # "gemini" | "bedrock" | "openai" | "offline" | "auto"
        model_name: str | None = None,
        use_live_llm: bool = False,
        fallback_extractor: LlamaIndexSchemaAndDynamicExtractor | None = None,
    ):
        self.provider = provider
        self.model_name = model_name or os.environ.get("KG_FABRIC_LLM_MODEL", "gemini-2.5-flash")
        self.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.use_live_llm = use_live_llm
        self.fallback = fallback_extractor or LlamaIndexSchemaAndDynamicExtractor()
        self.circuit_breaker = AsyncCircuitBreaker(failure_threshold=4, recovery_timeout_sec=30.0)

    @property
    def active_backend(self) -> str:
        if not self.use_live_llm:
            return "deterministic_llamaindex_extractor"
        if self.provider in ("gemini", "auto") and self.api_key:
            return f"google_genai:{self.model_name}"
        if self.provider in ("openai", "auto") and self.openai_key:
            return f"openai:{self.model_name}"
        if self.provider == "bedrock":
            return f"aws_bedrock:{self.model_name}"
        return "deterministic_llamaindex_extractor"

    @classmethod
    def extract_nary_qualifiers(cls, text: str) -> dict[str, Any]:
        """Extracts Zep/Graphiti / HyperGraphRAG style N-ary qualifiers from claim prose."""
        qualifiers: dict[str, Any] = {}
        m_cond = cls._COND_QUALIFIER_RE.search(text)
        if m_cond:
            qualifiers["condition"] = m_cond.group(1).strip()
        m_app = cls._APPROVER_QUALIFIER_RE.search(text)
        if m_app:
            qualifiers["approved_by"] = m_app.group(1).strip().rstrip(".,;:")
        if re.search(r"\b(?:must\s+not|cannot|never|prohibited\s+from|exempt\s+from)\b", text, re.I):
            qualifiers["negated_or_prohibited"] = True
        return qualifiers

    def _build_extraction_prompt(
        self, chunk: VectorChunkRecord, tbox: EnterpriseCompiledTBox
    ) -> str:
        allowed_classes = sorted(tbox.classes.keys())
        allowed_preds = sorted(tbox.relations.keys())
        return (
            "Extract factual knowledge graph triples with optional N-ary qualifiers "
            "(condition, approved_by) strictly using these ontology classes and predicates.\n"
            'Return ONLY valid JSON matching: {"triples": [{"subject": str, "subject_class": str, '
            '"predicate": str, "object": str, "object_class": str, "confidence": float, "qualifiers": {}}]}.\n'
            f"Allowed Classes: {allowed_classes}\n"
            f"Allowed Predicates: {allowed_preds}\n\n"
            f"Text:\n{chunk.text}"
        )

    def _call_gemini_structured(
        self, chunk: VectorChunkRecord, tbox: EnterpriseCompiledTBox
    ) -> _LLMStructuredExtractionPayload:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        resp = client.models.generate_content(
            model=self.model_name,
            contents=self._build_extraction_prompt(chunk, tbox),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_LLMStructuredExtractionPayload,
                temperature=0.0,
            ),
        )
        return _LLMStructuredExtractionPayload.model_validate_json(resp.text or "{}")

    def _call_openai_structured(
        self, chunk: VectorChunkRecord, tbox: EnterpriseCompiledTBox
    ) -> _LLMStructuredExtractionPayload:
        from openai import OpenAI

        client = OpenAI(api_key=self.openai_key)
        resp = client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a strict Knowledge Graph triple extractor."},
                {"role": "user", "content": self._build_extraction_prompt(chunk, tbox)},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        return _LLMStructuredExtractionPayload.model_validate_json(raw)

    def _call_bedrock_structured(
        self, chunk: VectorChunkRecord, tbox: EnterpriseCompiledTBox
    ) -> _LLMStructuredExtractionPayload:
        import boto3

        client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        resp = client.converse(
            modelId=self.model_name,
            inferenceConfig={"temperature": 0.0, "maxTokens": 2048},
            messages=[
                {
                    "role": "user",
                    "content": [{"text": self._build_extraction_prompt(chunk, tbox)}],
                }
            ],
        )
        content_blocks = resp.get("output", {}).get("message", {}).get("content", [])
        text = "".join(b.get("text", "") for b in content_blocks if isinstance(b, dict)).strip()
        m = re.search(r"\{.*\}", text, re.S)
        return _LLMStructuredExtractionPayload.model_validate_json(m.group(0) if m else "{}")

    def extract_from_chunk(
        self,
        chunk: VectorChunkRecord,
        tbox: EnterpriseCompiledTBox,
        allow_dynamic_emergent: bool = True,
    ) -> ChunkExtractionResult:
        nary_qualifiers = self.extract_nary_qualifiers(chunk.text)
        base_result: ChunkExtractionResult | None = None

        live_caller = None
        if self.use_live_llm and self.circuit_breaker.state != "OPEN":
            if self.provider in ("gemini", "auto") and self.api_key:
                live_caller = self._call_gemini_structured
            elif self.provider in ("openai", "auto") and self.openai_key:
                live_caller = self._call_openai_structured
            elif self.provider == "bedrock":
                live_caller = self._call_bedrock_structured

        if live_caller is not None:
            try:
                payload = live_caller(chunk, tbox)
                self.circuit_breaker.record_success()
                entities: list[ExtractedEntityMention] = []
                triples: list[ExtractedRelationTriple] = []
                for t in payload.triples:
                    s_cls = tbox.resolve_class(t.subject_class)
                    o_cls = tbox.resolve_class(t.object_class)
                    pred = tbox.resolve_predicate(t.predicate)
                    entities.append(
                        ExtractedEntityMention(
                            surface_name=t.subject,
                            proposed_class=s_cls,
                            confidence=t.confidence,
                            is_novel_class=(s_cls not in tbox.classes),
                        )
                    )
                    entities.append(
                        ExtractedEntityMention(
                            surface_name=t.object,
                            proposed_class=o_cls,
                            confidence=t.confidence,
                            is_novel_class=(o_cls not in tbox.classes),
                        )
                    )
                    merged_q = {**nary_qualifiers, **t.qualifiers}
                    triples.append(
                        ExtractedRelationTriple(
                            subject_name=t.subject,
                            subject_class=s_cls,
                            predicate=pred,
                            object_name=t.object,
                            object_class=o_cls,
                            evidence_snippet=chunk.text[:220],
                            confidence=t.confidence,
                            qualifiers=merged_q,
                            is_novel_predicate=(pred not in tbox.relations),
                        )
                    )
                base_result = ChunkExtractionResult(
                    chunk_id=chunk.chunk_id,
                    kb_id=chunk.kb_id,
                    acl_groups=list(chunk.acl_groups),
                    entities=entities,
                    triples=triples,
                )
            except Exception:
                self.circuit_breaker.record_failure()

        if base_result is None:
            base_result = self.fallback.extract_from_chunk(
                chunk, tbox=tbox, allow_dynamic_emergent=allow_dynamic_emergent
            )
            base_result.acl_groups = list(chunk.acl_groups)
            if nary_qualifiers:
                for tr in base_result.triples:
                    tr.qualifiers = {**nary_qualifiers, **tr.qualifiers}

        return base_result

    def generate_text_sync(self, prompt: str, system_prompt: str = "") -> str | None:
        """
        Shared live LLM text-generation primitive (Gemini / OpenAI / AWS Bedrock)
        used for optional grounded answer synthesis, LLM-assisted ontology induction,
        and LLM text-to-OpenCypher. Returns None when offline or circuit is open.
        """
        if not self.use_live_llm or self.circuit_breaker.state == "OPEN":
            return None
        try:
            if self.provider in ("gemini", "auto") and self.api_key:
                from google import genai

                client = genai.Client(api_key=self.api_key)
                full_prompt = f"{system_prompt}\n\n{prompt}".strip() if system_prompt else prompt
                resp = client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config={"temperature": 0.0},
                )
                self.circuit_breaker.record_success()
                return (getattr(resp, "text", None) or "").strip() or None
            if self.provider in ("openai", "auto") and self.openai_key:
                from openai import OpenAI

                client = OpenAI(api_key=self.openai_key)
                msgs = []
                if system_prompt:
                    msgs.append({"role": "system", "content": system_prompt})
                msgs.append({"role": "user", "content": prompt})
                resp = client.chat.completions.create(
                    model=self.model_name,
                    temperature=0.0,
                    messages=msgs,
                )
                self.circuit_breaker.record_success()
                return (resp.choices[0].message.content or "").strip() or None
            if self.provider == "bedrock":
                import boto3

                client = boto3.client(
                    "bedrock-runtime",
                    region_name=os.environ.get("AWS_REGION", "us-east-1"),
                )
                full_prompt = f"{system_prompt}\n\n{prompt}".strip() if system_prompt else prompt
                resp = client.converse(
                    modelId=self.model_name,
                    inferenceConfig={"temperature": 0.0, "maxTokens": 1024},
                    messages=[{"role": "user", "content": [{"text": full_prompt}]}],
                )
                blocks = resp.get("output", {}).get("message", {}).get("content", [])
                self.circuit_breaker.record_success()
                return "".join(b.get("text", "") for b in blocks if isinstance(b, dict)).strip() or None
        except Exception:
            self.circuit_breaker.record_failure()
        return None

    async def generate_text_async(self, prompt: str, system_prompt: str = "") -> str | None:
        """Non-blocking wrapper around `generate_text_sync` so FastAPI/Uvicorn never stalls."""
        import asyncio

        if not self.use_live_llm:
            return None
        return await asyncio.to_thread(self.generate_text_sync, prompt, system_prompt)

    async def extract_from_chunk_async(
        self,
        chunk: VectorChunkRecord,
        tbox: EnterpriseCompiledTBox,
        allow_dynamic_emergent: bool = True,
    ) -> ChunkExtractionResult:
        import asyncio

        if self.use_live_llm:
            return await asyncio.to_thread(
                self.extract_from_chunk, chunk, tbox, allow_dynamic_emergent
            )
        return self.extract_from_chunk(
            chunk, tbox=tbox, allow_dynamic_emergent=allow_dynamic_emergent
        )

    async def extract_batch_async(
        self,
        chunks: list[VectorChunkRecord],
        tbox: EnterpriseCompiledTBox,
        max_concurrency: int = 8,
        allow_dynamic_emergent: bool = True,
    ) -> list[tuple[VectorChunkRecord, ChunkExtractionResult | Exception]]:
        """
        Bounded-concurrency non-blocking chunk extraction (`asyncio.Semaphore` + `asyncio.to_thread`).
        Returns per-chunk `(chunk, result_or_exception)` preserving input order so DLQ fault isolation
        remains exact even when concurrent live LLM calls fail individually.
        """
        import asyncio

        if not chunks:
            return []
        if not self.use_live_llm:
            results: list[tuple[VectorChunkRecord, ChunkExtractionResult | Exception]] = []
            for chk in chunks:
                try:
                    res = self.extract_from_chunk(
                        chk, tbox=tbox, allow_dynamic_emergent=allow_dynamic_emergent
                    )
                    results.append((chk, res))
                except Exception as exc:
                    results.append((chk, exc))
            return results

        sem = asyncio.Semaphore(max(1, max_concurrency))

        async def _run_one(
            chk: VectorChunkRecord,
        ) -> tuple[VectorChunkRecord, ChunkExtractionResult | Exception]:
            async with sem:
                try:
                    res = await self.extract_from_chunk_async(
                        chk, tbox=tbox, allow_dynamic_emergent=allow_dynamic_emergent
                    )
                    return chk, res
                except Exception as exc:
                    return chk, exc

        return list(await asyncio.gather(*[_run_one(c) for c in chunks]))
