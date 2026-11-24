from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import (
    ChunkExtractionResult,
    EnterpriseCompiledTBox,
    ExtractedEntityMention,
    ExtractedRelationTriple,
    RawDocumentInput,
    UpperOntologyCategory,
    VectorChunkRecord,
    compute_sha256,
    utc_now_iso,
)
from .pii_redaction import (
    PIIRedactor,
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
        pii_redactor: Any | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.canonical_embedding_dim = canonical_embedding_dim
        self.pii_redactor = pii_redactor

    def parse_document(
        self,
        doc: RawDocumentInput,
        native_kb_dim: int = 768,
        audit_counts: dict[str, int] | None = None,
    ) -> list[VectorChunkRecord]:
        raw_text = doc.content.strip()
        if not raw_text:
            return []
        if self.pii_redactor is not None:
            redaction_res = self.pii_redactor.analyze_and_redact(raw_text)
            scrubbed_content = redaction_res.scrubbed_text
            if audit_counts is not None:
                for k, v in redaction_res.redaction_counts.items():
                    audit_counts[k] = audit_counts.get(k, 0) + v
        else:
            scrubbed_content = PIIRedactor.scrub(raw_text)
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
                chunk_id = f"{doc.tenant_id}::{doc.kb_id}::{doc.doc_id}::s{sec_idx}_c{sub_idx}"
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
                    acl_groups=list(doc.acl_groups),
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
        if default_cls in {
            "OrganizationOrFoundation",
            "ProgrammingLanguage",
            "QueryLanguageOrProtocol",
            "AIConcept",
            "SoftwarePlatform",
        }:
            return tbox.resolve_class(default_cls)
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

        # 4. Open-Domain Internet Prose Extractor with Document-Title Coreference Resolution
        # Handles real unstructured Wikipedia / web documentation articles cleanly.
        if chunk.metadata.get("open_domain_prose") or chunk.metadata.get("source_url"):
            raw_title = str(chunk.metadata.get("title") or "").strip()
            clean_title = re.sub(r"\s*\([^)]*\)\s*$", "", raw_title).strip()
            clean_title = re.sub(r"^The\s+", "", clean_title).strip()
            self._extract_open_domain_triples(
                text=text,
                doc_subject=clean_title,
                tbox=tbox,
                t_valid=str(t_valid_override),
                register_entity=_register_entity,
                triples=triples,
                seen_triple_sigs=seen_triple_sigs,
            )

        return ChunkExtractionResult(
            chunk_id=chunk.chunk_id,
            kb_id=chunk.kb_id,
            acl_groups=list(chunk.acl_groups),
            entities=list(entities_by_key.values()),
            triples=triples,
        )

    @staticmethod
    def _extract_open_domain_triples(
        text: str,
        doc_subject: str,
        tbox: EnterpriseCompiledTBox,
        t_valid: str,
        register_entity: Any,
        triples: list[ExtractedRelationTriple],
        seen_triple_sigs: set[tuple[str, str, str]],
    ) -> None:
        """
        Extracts real-world entities and relationships from open-domain internet prose
        (e.g., Wikipedia articles, technical docs, CVE advisories, regulatory texts)
        using Document-Title Coreference Resolution + lexico-syntactic relation patterns.
        """
        # Canonical domain entity gazetteer & class typing for open-domain tech/cloud/security/AI/legal prose
        known_entities: list[tuple[str, str, re.Pattern[str]]] = [
            # Organizations & Foundations
            ("Google", "OrganizationOrFoundation", re.compile(r"\bGoogle\b")),
            ("Amazon Web Services", "OrganizationOrFoundation", re.compile(r"\b(?:Amazon Web Services|AWS)\b")),
            ("Amazon", "OrganizationOrFoundation", re.compile(r"\bAmazon(?:\.com)?\b")),
            ("Cloud Native Computing Foundation", "OrganizationOrFoundation", re.compile(r"\b(?:Cloud Native Computing Foundation|CNCF)\b")),
            ("Apache Software Foundation", "OrganizationOrFoundation", re.compile(r"\bApache Software Foundation\b")),
            ("Linux Foundation", "OrganizationOrFoundation", re.compile(r"\bLinux Foundation\b")),
            ("HashiCorp", "OrganizationOrFoundation", re.compile(r"\bHashiCorp\b")),
            ("LinkedIn", "OrganizationOrFoundation", re.compile(r"\bLinkedIn\b")),
            ("SoundCloud", "OrganizationOrFoundation", re.compile(r"\bSoundCloud\b")),
            ("Neo4j, Inc.", "OrganizationOrFoundation", re.compile(r"\bNeo4j,?\s*Inc\.?\b")),
            ("PostgreSQL Global Development Group", "OrganizationOrFoundation", re.compile(r"\bPostgreSQL Global Development Group\b")),
            ("OWASP", "OrganizationOrFoundation", re.compile(r"\b(?:OWASP|Open Worldwide Application Security Project)\b")),
            ("European Union", "OrganizationOrFoundation", re.compile(r"\b(?:European Union|European Parliament)\b")),
            ("Payment Card Industry Security Standards Council", "OrganizationOrFoundation", re.compile(r"\bPayment Card Industry Security Standards Council\b")),
            ("United States Congress", "OrganizationOrFoundation", re.compile(r"\bUnited States Congress\b")),
            ("Docker, Inc.", "OrganizationOrFoundation", re.compile(r"\bDocker,?\s*Inc\.?\b")),
            ("Grafana Labs", "OrganizationOrFoundation", re.compile(r"\bGrafana Labs\b")),
            # Databases & Streaming Stores
            ("PostgreSQL", "DatabaseCluster", re.compile(r"\bPostgreSQL\b")),
            ("Neo4j", "DatabaseCluster", re.compile(r"\bNeo4j\b")),
            ("Amazon Neptune", "DatabaseCluster", re.compile(r"\bAmazon Neptune\b")),
            ("Redis", "DatabaseCluster", re.compile(r"\bRedis\b")),
            ("Apache Kafka", "DatabaseCluster", re.compile(r"\b(?:Apache )?Kafka\b")),
            ("Vector database", "DatabaseCluster", re.compile(r"\bvector databases?\b", re.I)),
            # Cloud & Software Platforms
            ("Kubernetes", "SoftwarePlatform", re.compile(r"\bKubernetes\b")),
            ("Docker", "SoftwarePlatform", re.compile(r"\bDocker\b")),
            ("Prometheus", "SoftwarePlatform", re.compile(r"\bPrometheus\b")),
            ("Grafana", "SoftwarePlatform", re.compile(r"\bGrafana\b")),
            ("Terraform", "SoftwarePlatform", re.compile(r"\bTerraform\b")),
            ("Apache Log4j", "SoftwarePlatform", re.compile(r"\b(?:Apache )?Log4j\b")),
            ("OpenSSL", "SoftwarePlatform", re.compile(r"\bOpenSSL\b")),
            ("Linux", "SoftwarePlatform", re.compile(r"\bLinux\b")),
            ("containerd", "SoftwarePlatform", re.compile(r"\bcontainerd\b")),
            # Security Vulnerabilities
            ("Log4Shell", "SecurityVulnerability", re.compile(r"\bLog4Shell\b")),
            ("Heartbleed", "SecurityVulnerability", re.compile(r"\bHeartbleed\b")),
            # Compliance Standards
            ("General Data Protection Regulation", "ComplianceStandard", re.compile(r"\b(?:General Data Protection Regulation|GDPR)\b")),
            ("PCI DSS", "ComplianceStandard", re.compile(r"\b(?:Payment Card Industry Data Security Standard|PCI[\-\s]DSS)\b")),
            ("HIPAA", "ComplianceStandard", re.compile(r"\b(?:Health Insurance Portability and Accountability Act|HIPAA)\b")),
            # AI & Semantic Concepts
            ("Retrieval-augmented generation", "AIConcept", re.compile(r"\b(?:Retrieval-augmented generation|RAG)\b", re.I)),
            ("Knowledge graph", "AIConcept", re.compile(r"\bknowledge graphs?\b", re.I)),
            ("Ontology", "AIConcept", re.compile(r"\bontolog(?:y|ies)\b", re.I)),
            ("Large language model", "AIConcept", re.compile(r"\b(?:large language models?|LLMs?)\b", re.I)),
            # Programming Languages
            ("Go", "ProgrammingLanguage", re.compile(r"\b(?:Go\b|Golang\b)")),
            ("Java", "ProgrammingLanguage", re.compile(r"\bJava\b")),
            ("Python", "ProgrammingLanguage", re.compile(r"\bPython\b")),
            ("Scala", "ProgrammingLanguage", re.compile(r"\bScala\b")),
            ("C", "ProgrammingLanguage", re.compile(r"\bwritten in C\b|\bC programming language\b")),
            # Query Languages & Protocols
            ("openCypher", "QueryLanguageOrProtocol", re.compile(r"\bopenCypher\b", re.I)),
            ("Cypher", "QueryLanguageOrProtocol", re.compile(r"\bCypher\b")),
            ("Gremlin", "QueryLanguageOrProtocol", re.compile(r"\bGremlin\b")),
            ("SPARQL", "QueryLanguageOrProtocol", re.compile(r"\bSPARQL\b")),
            ("SQL", "QueryLanguageOrProtocol", re.compile(r"\bSQL\b")),
            ("TLS", "QueryLanguageOrProtocol", re.compile(r"\b(?:Transport Layer Security|TLS)\b")),
            ("RDF", "QueryLanguageOrProtocol", re.compile(r"\b(?:Resource Description Framework|RDF)\b")),
        ]

        # Resolve doc_subject canonical name and class
        subj_name = doc_subject
        subj_cls = "SoftwarePlatform"
        for k_name, k_cls, k_pat in known_entities:
            if doc_subject and (doc_subject.lower() == k_name.lower() or k_pat.fullmatch(doc_subject)):
                subj_name = k_name
                subj_cls = k_cls
                break
        if not subj_name:
            return

        def _add_triple(s_name: str, s_cls: str, pred: str, o_name: str, o_cls: str, snippet: str) -> None:
            if s_name.lower() == o_name.lower():
                return
            c_pred = tbox.resolve_predicate(pred)
            sig = (s_name.lower(), c_pred, o_name.lower())
            if sig in seen_triple_sigs:
                return
            seen_triple_sigs.add(sig)
            s_e = register_entity(s_name, s_cls, conf=0.92)
            o_e = register_entity(o_name, o_cls, conf=0.92)
            triples.append(
                ExtractedRelationTriple(
                    subject_name=s_e.surface_name,
                    subject_class=s_e.proposed_class,
                    predicate=c_pred,
                    object_name=o_e.surface_name,
                    object_class=o_e.proposed_class,
                    evidence_snippet=snippet[:220],
                    confidence=0.92,
                    t_valid=t_valid,
                    is_novel_predicate=(c_pred not in tbox.relations),
                )
            )

        # Register the document's primary subject entity
        register_entity(subj_name, subj_cls, conf=0.95)

        sentences = [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+|\n+", text) if len(s.strip()) > 15]
        for sent in sentences:
            # Resolve leading coreference ("It ", "The software ", etc.) to doc_subject
            resolved_sent = re.sub(
                r"^(?:It|The\s+(?:software|system|project|database|vulnerability|bug|flaw|library|regulation|platform|framework|tool|standard))\b",
                subj_name,
                sent,
            )
            sent_lower = resolved_sent.lower()

            # Discover any explicit CVE identifiers in sentence
            for cve_m in re.finditer(r"\bCVE-\d{4}-\d{4,7}\b", resolved_sent):
                cve_id = cve_m.group(0)
                if subj_cls == "SecurityVulnerability":
                    _add_triple(subj_name, subj_cls, "INTEGRATES_WITH", cve_id, "SecurityVulnerability", sent)
                else:
                    _add_triple(cve_id, "SecurityVulnerability", "AFFECTS_ASSET", subj_name, subj_cls, sent)

            # Match known entities mentioned in this sentence
            found_in_sent: list[tuple[str, str]] = []
            for k_name, k_cls, k_pat in known_entities:
                if k_name.lower() == subj_name.lower():
                    continue
                if k_pat.search(resolved_sent):
                    found_in_sent.append((k_name, k_cls))

            for target_name, target_cls in found_in_sent:
                # 1. Organization / Foundation relationships
                if target_cls == "OrganizationOrFoundation":
                    if subj_cls == "ComplianceStandard" and any(
                        w in sent_lower for w in ("regulation", "law", "enacted", "adopted", "council", "parliament", "congress", "standard")
                    ):
                        _add_triple(subj_name, subj_cls, "ENACTED_BY", target_name, target_cls, sent)
                    elif any(w in sent_lower for w in ("maintained by", "hosted by", "graduated", "foundation", "donated to", "cncf", "apache software foundation")):
                        if "foundation" in target_name.lower() or target_name in {"OWASP", "Cloud Native Computing Foundation", "Apache Software Foundation", "Linux Foundation"}:
                            _add_triple(subj_name, subj_cls, "HOSTED_BY_FOUNDATION", target_name, target_cls, sent)
                        else:
                            _add_triple(subj_name, subj_cls, "DEVELOPED_BY", target_name, target_cls, sent)
                    elif any(w in sent_lower for w in ("designed", "developed", "created", "built", "founded", "introduced", "offered", "managed", "released", "originally", "product")):
                        _add_triple(subj_name, subj_cls, "DEVELOPED_BY", target_name, target_cls, sent)

                # 2. Programming Language relationships
                elif target_cls == "ProgrammingLanguage":
                    if any(w in sent_lower for w in ("written in", "implemented in", "programming language", "java logging", "written primarily in")):
                        _add_triple(subj_name, subj_cls, "WRITTEN_IN", target_name, target_cls, sent)

                # 3. Query Language & Wire Protocol relationships
                elif target_cls == "QueryLanguageOrProtocol":
                    if subj_cls == "SecurityVulnerability":
                        _add_triple(subj_name, subj_cls, "AFFECTS_ASSET", target_name, target_cls, sent)
                    elif any(w in sent_lower for w in ("support", "query", "language", "implement", "protocol", "uses", "compatible", "relational", "graph", "encryption")):
                        _add_triple(subj_name, subj_cls, "SUPPORTS_PROTOCOL", target_name, target_cls, sent)

                # 4. Vulnerability -> Affected Software Platform / Database
                elif subj_cls == "SecurityVulnerability" and target_cls in {"SoftwarePlatform", "DatabaseCluster"}:
                    if any(w in sent_lower for w in ("vulnerability", "flaw", "bug", "affect", "in ", "exploit", "library", "zero-day", "arbitrary code")):
                        _add_triple(subj_name, subj_cls, "AFFECTS_ASSET", target_name, target_cls, sent)

                # 5. Software Platform / Database / AI Concept Ecosystem Integrations
                elif target_cls in {"SoftwarePlatform", "DatabaseCluster", "AIConcept"} and subj_cls in {
                    "SoftwarePlatform",
                    "DatabaseCluster",
                    "AIConcept",
                }:
                    if any(
                        w in sent_lower
                        for w in (
                            "integrat",
                            "work",
                            "with",
                            "support",
                            "container",
                            "visualiz",
                            "monitor",
                            "retriev",
                            "augment",
                            "vector",
                            "graph",
                            "ontology",
                            "knowledge",
                            "language model",
                            "database",
                            "data source",
                            "runtime",
                        )
                    ):
                        _add_triple(subj_name, subj_cls, "INTEGRATES_WITH", target_name, target_cls, sent)
