from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from .models import (
    ChunkExtractionResult,
    EnterpriseCompiledTBox,
    ExtractedEntityMention,
    ExtractedRelationTriple,
    OntologyStatus,
    RawDocumentInput,
    UpperOntologyCategory,
    VectorChunkRecord,
    compute_sha256,
    utc_now_iso,
)
from .pii_redaction import (
    PIIRedactionMode,
    PIIRedactionResult,
    PIIRedactor,
    PresidioPIIRedactor,
)
from .vector_clustering import EmbeddingProvider


@dataclass
class LlamaTextNode:
    """
    LlamaIndex-compatible `TextNode` representation (`llama_index.core.schema.TextNode`).
    If `llama-index-core` is installed in the environment, `.to_native_llama_node()` converts
    directly to a `llama_index.core.schema.TextNode`.
    """

    id_: str
    text: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_node_id: str | None = None
    prev_node_id: str | None = None
    next_node_id: str | None = None

    def to_native_llama_node(self) -> Any:
        try:
            from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

            relationships = {}
            if self.parent_node_id:
                relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=self.parent_node_id)
            if self.prev_node_id:
                relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=self.prev_node_id)
            if self.next_node_id:
                relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=self.next_node_id)
            return TextNode(
                id_=self.id_,
                text=self.text,
                embedding=self.embedding if self.embedding else None,
                metadata=self.metadata,
                relationships=relationships,
            )
        except ImportError:
            return self


class LlamaIndexHierarchicalNodeParser:
    """
    LlamaIndex-compatible Hierarchical & Markdown Section Node Parser.
    Splits raw enterprise documents into parent section nodes and child semantic chunks,
    scrubs PII/credentials, computes native KB embeddings + canonical embeddings,
    and emits `VectorChunkRecord`s ready for `PGVectorStore` and `PropertyGraphIndex`.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        max_chunk_chars: int = 650,
        chunk_overlap_chars: int = 80,
        canonical_embedding_dim: int = 256,
    ):
        self.embedding_provider = embedding_provider
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.canonical_embedding_dim = canonical_embedding_dim

    def parse_document(
        self,
        doc: RawDocumentInput,
        native_kb_dim: int = 768,
    ) -> list[VectorChunkRecord]:
        scrubbed_content = PIIRedactor.scrub(doc.content.strip())
        if not scrubbed_content:
            return []

        # Split by markdown headers or double-newline semantic paragraphs
        raw_sections = re.split(r"\n(?=#{1,3}\s+)", scrubbed_content)
        records: list[VectorChunkRecord] = []
        parent_id = f"{doc.doc_id}::root"

        for sec_idx, section in enumerate(raw_sections):
            sec_clean = section.strip()
            if not sec_clean:
                continue
            lines = sec_clean.splitlines()
            header_line = lines[0].lstrip("#").strip() if lines[0].startswith("#") else doc.title
            section_path = f"{doc.title} > {header_line}" if header_line != doc.title else doc.title

            # Split large sections into overlapping semantic chunks
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", sec_clean) if p.strip()]
            buffer = ""
            sub_chunks: list[str] = []
            for para in paragraphs:
                if not buffer:
                    buffer = para
                elif len(buffer) + len(para) + 2 <= self.max_chunk_chars:
                    buffer = f"{buffer}\n\n{para}"
                else:
                    sub_chunks.append(buffer)
                    overlap = buffer[-self.chunk_overlap_chars :] if len(buffer) > self.chunk_overlap_chars else ""
                    buffer = f"{overlap}\n{para}".strip()
            if buffer:
                sub_chunks.append(buffer)

            for sub_idx, chunk_text in enumerate(sub_chunks):
                chunk_id = f"{doc.kb_id}::{doc.doc_id}::s{sec_idx}_c{sub_idx}"
                contextual_text = (
                    f"[{section_path}] {chunk_text}"
                    if not chunk_text.startswith("#")
                    else chunk_text
                )
                native_vec = self.embedding_provider.embed_text(contextual_text, dim=native_kb_dim)
                canon_vec = self.embedding_provider.embed_text(
                    contextual_text, dim=self.canonical_embedding_dim
                )
                rec = VectorChunkRecord(
                    chunk_id=chunk_id,
                    kb_id=doc.kb_id,
                    tenant_id=doc.tenant_id,
                    domain_tag=doc.domain_tag,
                    doc_id=doc.doc_id,
                    text=contextual_text,
                    content_sha256=compute_sha256(contextual_text),
                    native_embedding=native_vec,
                    native_embedding_dim=native_kb_dim,
                    canonical_embedding=canon_vec,
                    parent_node_id=parent_id,
                    section_path=section_path,
                    metadata={
                        **doc.metadata,
                        "title": doc.title,
                        "source_uri": doc.source_uri,
                        "section_path": section_path,
                    },
                )
                records.append(rec)

        # Link prev_node_id and next_node_id across chunks in the document
        for idx, rec in enumerate(records):
            if idx > 0:
                rec.prev_node_id = records[idx - 1].chunk_id
            if idx + 1 < len(records):
                rec.next_node_id = records[idx + 1].chunk_id

        return records


class LlamaIndexSchemaAndDynamicExtractor:
    """
    Combines LlamaIndex's two core PropertyGraphIndex extractors:
      1. `SchemaLLMPathExtractor`: Extracts entities & triples strictly aligned with the
         induced `EnterpriseCompiledTBox` (resolving KB-local class and predicate synonyms).
      2. `DynamicLLMPathExtractor`: Discovers open-world emergent entities & predicates
         not yet in the approved TBox, routing them for Two-Bucket governance validation.
    """

    # Recognizes structured natural-language patterns across technical, FinOps, SecOps, Compliance & Customer KBs
    _NL_RELATION_PATTERNS: list[tuple[re.Pattern[str], str, str, str]] = [
        # (regex, default_subject_class, raw_predicate, default_object_class)
        (
            re.compile(
                r"\b([A-Za-z0-9][A-Za-z0-9_\-\.]{2,35})\s+(?:depends\s+on|calls\s+upstream|requires\s+service|is\s+backed\s+by|connects\s+to)\s+([A-Za-z0-9][A-Za-z0-9_\-\.]{2,35})\b",
                re.I,
            ),
            "CloudService",
            "DEPENDS_ON",
            "DatabaseCluster",
        ),
        (
            re.compile(
                r"\b([A-Za-z0-9][A-Za-z0-9_\-\.]{2,35})\s+(?:is\s+deployed\s+in\s+primary|is\s+hosted\s+in\s+region|runs\s+in\s+primary\s+region|failed\s+over\s+to)\s+([a-z]{2}\-[a-z]+\-\d)\b",
                re.I,
            ),
            "CloudService",
            "DEPLOYED_IN_PRIMARY",
            "CloudRegion",
        ),
        (
            re.compile(
                r"\b([A-Za-z0-9][A-Za-z0-9_\-\.]{2,35})\s+(?:is\s+allocated\s+to\s+cost\s+center|is\s+owned\s+by\s+primary|is\s+funded\s+by\s+cost\s+center)\s+(CC\-[0-9A-Za-z\-]+)\b",
                re.I,
            ),
            "CloudService",
            "OWNED_BY_PRIMARY",
            "CostCenter",
        ),
        (
            re.compile(
                r"\b([A-Za-z0-9][A-Za-z0-9_\-\.]{2,35})\s+(?:is\s+maintained\s+by|is\s+operated\s+by\s+squad|is\s+stewarded\s+by)\s+([A-Za-z0-9][A-Za-z0-9_\-]{2,35})\b",
                re.I,
            ),
            "CloudService",
            "MAINTAINED_BY_TEAM",
            "EngineeringTeam",
        ),
        (
            re.compile(
                r"\b([A-Za-z0-9][A-Za-z0-9_\-\.]{2,35})\s+(?:is\s+governed\s+by|complies\s+with|is\s+regulated\s+by|is\s+audited\s+under)\s+((?:PCI\-DSS|GDPR|SOC2|HIPAA|ISO\-27001)[A-Za-z0-9_\-\.]*)\b",
                re.I,
            ),
            "DatabaseCluster",
            "GOVERNED_BY",
            "ComplianceStandard",
        ),
        (
            re.compile(
                r"\b((?:CVE\-\d{4}\-\d{3,6}|INC\-\d{3,6}))\s+(?:affects|impacts|exposes|degrades)\s+([A-Za-z0-9][A-Za-z0-9_\-\.]{2,35})\b",
                re.I,
            ),
            "SecurityVulnerability",
            "AFFECTS_ASSET",
            "CloudService",
        ),
        (
            re.compile(
                r"\b((?:CVE\-\d{4}\-\d{3,6}|INC\-\d{3,6}))\s+(?:is\s+mitigated\s+by|is\s+remediated\s+by|is\s+blocked\s+by|is\s+shielded\s+by)\s+([A-Za-z0-9][A-Za-z0-9_\-\.]{2,35})\b",
                re.I,
            ),
            "SecurityVulnerability",
            "MITIGATED_BY_CONTROL",
            "SecurityControl",
        ),
        (
            re.compile(
                r"\b([A-Za-z0-9][A-Za-z0-9_\-]{2,35})\s+(?:subscribes\s+to\s+sla|is\s+contracted\s+under|uses\s+platform\s+service)\s+([A-Za-z0-9][A-Za-z0-9_\-\.]{2,35})\b",
                re.I,
            ),
            "EnterpriseCustomer",
            "SUBSCRIBES_TO_SLA",
            "CloudService",
        ),
    ]

    @staticmethod
    def _infer_entity_class(name: str, default_cls: str, tbox: EnterpriseCompiledTBox) -> str:
        upper = name.upper()
        lower = name.lower()
        if upper.startswith("CVE-"):
            return tbox.resolve_class("SecurityVulnerability")
        if upper.startswith("INC-"):
            return tbox.resolve_class("OperationalIncident")
        if upper.startswith("CC-"):
            return tbox.resolve_class("CostCenter")
        if any(upper.startswith(p) for p in ("PCI-DSS", "GDPR", "SOC2", "HIPAA", "ISO-27001")):
            return tbox.resolve_class("ComplianceStandard")
        if re.match(r"^(us|eu|ap)\-[a-z]+\-\d$", lower):
            return tbox.resolve_class("CloudRegion")
        if any(k in lower for k in ("aurora", "postgres", "pgvector", "neptune", "neo4j", "ledger", "db", "warehouse", "cluster")):
            return tbox.resolve_class("DatabaseCluster")
        if any(k in lower for k in ("router", "gateway", "-gw", "api", "-svc", "service", "worker")):
            return tbox.resolve_class("CloudService")
        if any(k in lower for k in ("waf", "shield", "guard", "rule", "control")):
            return tbox.resolve_class("SecurityControl")
        if any(k in lower for k in ("squad", "team", "desk", "steward")):
            return tbox.resolve_class("EngineeringTeam")
        if any(k in lower for k in ("sla-", "tier-")):
            return tbox.resolve_class("SLAContract")
        if any(k in lower for k in ("acme", "globex", "initech", "umbrella", "wayne", "stark")):
            return tbox.resolve_class("EnterpriseCustomer")
        return tbox.resolve_class(default_cls)

    def extract_from_chunk(
        self,
        chunk: VectorChunkRecord,
        tbox: EnterpriseCompiledTBox,
        allow_dynamic_emergent: bool = True,
    ) -> ChunkExtractionResult:
        text = PIIRedactor.scrub(chunk.text)
        entities_by_key: dict[str, ExtractedEntityMention] = {}
        triples: list[ExtractedRelationTriple] = []

        t_valid_override = chunk.metadata.get("t_valid") or chunk.updated_at or utc_now_iso()

        def _register_entity(
            surface: str,
            raw_cls: str,
            conf: float = 0.92,
            summary: str = "",
        ) -> ExtractedEntityMention:
            resolved_cls = self._infer_entity_class(surface, raw_cls, tbox)
            is_novel = resolved_cls not in tbox.classes
            cls_def = tbox.classes.get(resolved_cls)
            upper_cat = (
                cls_def.upper_category
                if cls_def is not None
                else UpperOntologyCategory.SYSTEM_OR_ASSET
            )
            key = f"{resolved_cls}::{surface.lower()}"
            if key not in entities_by_key:
                entities_by_key[key] = ExtractedEntityMention(
                    surface_name=surface,
                    proposed_class=resolved_cls,
                    upper_category=upper_cat,
                    summary=summary or f"{resolved_cls} '{surface}' mentioned in KB {chunk.kb_id}",
                    confidence=conf,
                    is_novel_class=is_novel,
                )
            return entities_by_key[key]

        # 1. Extract structured triples from metadata if provided by upstream vector KB ETL
        meta_triples = chunk.metadata.get("structured_triples", [])
        if isinstance(meta_triples, list):
            for mt in meta_triples:
                if not isinstance(mt, dict):
                    continue
                subj = str(mt.get("subject", "")).strip()
                subj_cls = tbox.resolve_class(str(mt.get("subject_class", "CloudService")))
                pred = tbox.resolve_predicate(str(mt.get("predicate", "DEPENDS_ON")))
                obj = str(mt.get("object", "")).strip()
                obj_cls = tbox.resolve_class(str(mt.get("object_class", "DatabaseCluster")))
                conf = float(mt.get("confidence", 0.94))
                if not subj or not obj:
                    continue
                s_ent = _register_entity(subj, subj_cls, conf=conf)
                o_ent = _register_entity(obj, obj_cls, conf=conf)
                is_novel_pred = pred not in tbox.relations
                if is_novel_pred and not allow_dynamic_emergent:
                    continue
                triples.append(
                    ExtractedRelationTriple(
                        subject_name=s_ent.surface_name,
                        subject_class=s_ent.proposed_class,
                        predicate=pred,
                        object_name=o_ent.surface_name,
                        object_class=o_ent.proposed_class,
                        evidence_snippet=text[:240],
                        confidence=conf,
                        t_valid=str(mt.get("t_valid", t_valid_override)),
                        is_novel_predicate=is_novel_pred,
                    )
                )

        # 2. Extract inline structured triples: [Triple: subj (ClassA) --PRED--> obj (ClassB)]
        for m in re.finditer(
            r"\[Triple:\s*([A-Za-z0-9_\-\.\s]+?)\s*\(([A-Za-z0-9_]+)\)\s*\-\-([A-Za-z0-9_]+)\-\->\s*([A-Za-z0-9_\-\.\s]+?)\s*\(([A-Za-z0-9_]+)\)\]",
            text,
        ):
            subj, raw_s_cls, raw_pred, obj, raw_o_cls = (
                m.group(1).strip(),
                m.group(2).strip(),
                m.group(3).strip(),
                m.group(4).strip(),
                m.group(5).strip(),
            )
            s_cls = tbox.resolve_class(raw_s_cls)
            o_cls = tbox.resolve_class(raw_o_cls)
            pred = tbox.resolve_predicate(raw_pred)
            s_ent = _register_entity(subj, s_cls, conf=0.93)
            o_ent = _register_entity(obj, o_cls, conf=0.93)
            is_novel_pred = pred not in tbox.relations
            if is_novel_pred and not allow_dynamic_emergent:
                continue
            triples.append(
                ExtractedRelationTriple(
                    subject_name=s_ent.surface_name,
                    subject_class=s_ent.proposed_class,
                    predicate=pred,
                    object_name=o_ent.surface_name,
                    object_class=o_ent.proposed_class,
                    evidence_snippet=m.group(0),
                    confidence=0.93,
                    t_valid=str(t_valid_override),
                    is_novel_predicate=is_novel_pred,
                )
            )

        # 3. Conjunction Subject Carry-Over for messy compound prose
        # e.g., "payment-router-01 calls upstream aurora-ledger-01 and is deployed in primary us-east-1"
        expanded_sentences: list[str] = []
        _compound_pred_re = re.compile(
            r"\b([A-Za-z0-9][A-Za-z0-9_\-\.]{2,35})\s+"
            r"(?:depends\s+on|calls\s+upstream|requires\s+service|is\s+backed\s+by|connects\s+to|"
            r"is\s+deployed\s+in\s+primary|is\s+governed\s+by|complies\s+with|uses\s+platform\s+service|"
            r"is\s+allocated\s+to\s+cost\s+center|is\s+owned\s+by\s+primary|affects|impacts)\s+"
            r"[A-Za-z0-9][A-Za-z0-9_\-\.]{2,35}\s+and\s+"
            r"((?:is\s+deployed\s+in\s+primary|is\s+hosted\s+in\s+region|is\s+maintained\s+by|"
            r"is\s+governed\s+by|complies\s+with|is\s+allocated\s+to\s+cost\s+center|"
            r"is\s+owned\s+by\s+primary|is\s+mitigated\s+by|depends\s+on|subscribes\s+to\s+sla)\s+[^\.;\n]+)",
            re.I,
        )
        for sentence in re.split(r"(?<=[\.\!\?])\s+|\n+", text):
            expanded_sentences.append(sentence)
            for cm in _compound_pred_re.finditer(sentence):
                carried_subj = cm.group(1).strip()
                second_clause = cm.group(2).strip()
                if carried_subj.lower() not in {"the", "this", "that", "microservice", "database", "service"}:
                    expanded_sentences.append(f"{carried_subj} {second_clause}")

        nl_text = "\n".join(expanded_sentences)

        # Extract Natural-Language Relational Triples via SchemaLLMPathExtractor patterns
        seen_triple_sigs: set[tuple[str, str, str]] = set()
        for pattern, def_s_cls, raw_pred, def_o_cls in self._NL_RELATION_PATTERNS:
            for match in pattern.finditer(nl_text):
                subj_raw = match.group(1).strip().rstrip(".,;:")
                obj_raw = match.group(2).strip().rstrip(".,;:")
                # Filter out generic stopwords
                stopwords = {"the", "this", "that", "which", "service", "system", "every", "and", "or", "for", "with", "from", "into"}
                if subj_raw.lower() in stopwords or obj_raw.lower() in stopwords:
                    continue
                pred = tbox.resolve_predicate(raw_pred)
                sig = (subj_raw.lower(), pred, obj_raw.lower())
                if sig in seen_triple_sigs:
                    continue
                seen_triple_sigs.add(sig)
                s_ent = _register_entity(subj_raw, def_s_cls, conf=0.91)
                o_ent = _register_entity(obj_raw, def_o_cls, conf=0.91)
                triples.append(
                    ExtractedRelationTriple(
                        subject_name=s_ent.surface_name,
                        subject_class=s_ent.proposed_class,
                        predicate=pred,
                        object_name=o_ent.surface_name,
                        object_class=o_ent.proposed_class,
                        evidence_snippet=match.group(0),
                        confidence=0.91,
                        t_valid=str(t_valid_override),
                        is_novel_predicate=(pred not in tbox.relations),
                    )
                )

        return ChunkExtractionResult(
            chunk_id=chunk.chunk_id,
            kb_id=chunk.kb_id,
            entities=list(entities_by_key.values()),
            triples=triples,
        )
