from __future__ import annotations

import re
from typing import Sequence

from .models import (
    ChunkExtractionResult,
    EnterpriseCompiledTBox,
    ExtractedEntityMention,
    KGEntityNode,
    KGRelationEdge,
    OntologyStatus,
    UpperOntologyCategory,
    utc_now_iso,
)
from .vector_clustering import EmbeddingProvider, cosine_similarity


class SiblingCollisionGuard:
    """
    Prevents catastrophic false entity merges across 100s of KBs when two distinct
    entities have very high embedding cosine similarity (>0.95) but differ in:
      - Numeric identifiers / ordinals (e.g. `payment-router-01` vs `payment-router-02`, `CC-104` vs `CC-209`)
      - Cloud regions / zones (e.g. `us-east-1` vs `us-east-2` vs `eu-central-1`)
      - Environments (e.g. `prod` vs `staging` vs `canary` vs `dev`)
      - Version numbers (e.g. `PCI-DSS-3.2` vs `PCI-DSS-4.0`)
    """

    _REGION_RE = re.compile(r"\b(us|eu|ap|sa|ca|me|af)\-(east|west|central|north|south|northeast|southeast)\-\d\b", re.I)
    _ENV_TOKENS = {"prod", "production", "staging", "stage", "dev", "canary", "sandbox", "dr"}
    _NUMBER_SEQ_RE = re.compile(r"\d+(?:\.\d+)?")

    @classmethod
    def extract_numbers(cls, text: str) -> list[str]:
        # Normalize leading zeros on pure integers ("01" -> "1") so "Router 1" and "router-01" CAN merge,
        # while "router-01" ("1") and "router-02" ("2") CANNOT merge.
        raw_nums = cls._NUMBER_SEQ_RE.findall(text)
        normalized: list[str] = []
        for n in raw_nums:
            if "." in n:
                normalized.append(n)
            else:
                normalized.append(str(int(n)))
        return normalized

    @classmethod
    def extract_regions(cls, text: str) -> set[str]:
        return {m.group(0).lower() for m in cls._REGION_RE.finditer(text)}

    @classmethod
    def extract_envs(cls, text: str) -> set[str]:
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        envs = tokens.intersection(cls._ENV_TOKENS)
        if "production" in envs:
            envs.discard("production")
            envs.add("prod")
        if "stage" in envs:
            envs.discard("stage")
            envs.add("staging")
        return envs

    @classmethod
    def can_merge(cls, name_a: str, name_b: str) -> tuple[bool, str]:
        nums_a = cls.extract_numbers(name_a)
        nums_b = cls.extract_numbers(name_b)
        if nums_a != nums_b and (nums_a or nums_b):
            return False, f"Numeric identifier mismatch: {nums_a} vs {nums_b}"

        regs_a = cls.extract_regions(name_a)
        regs_b = cls.extract_regions(name_b)
        if regs_a and regs_b and regs_a != regs_b:
            return False, f"Region mismatch: {regs_a} vs {regs_b}"

        envs_a = cls.extract_envs(name_a)
        envs_b = cls.extract_envs(name_b)
        if envs_a and envs_b and envs_a != envs_b:
            return False, f"Environment mismatch: {envs_a} vs {envs_b}"

        return True, "OK"


class CrossKBEntityResolver:
    """
    Resolves entity mentions and relationship triples extracted across 100s of KBs
    into canonical `KGEntityNode`s and bi-temporal `KGRelationEdge`s.
    """

    _NOISE_SUFFIX_WORDS = {"service", "microservice", "cluster", "database", "db", "system", "workload", "standard"}

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        entity_similarity_threshold: float = 0.86,
        canonical_embedding_dim: int = 256,
    ):
        self.embedding_provider = embedding_provider
        self.similarity_threshold = entity_similarity_threshold
        self.canonical_embedding_dim = canonical_embedding_dim

    @classmethod
    def normalize_entity_slug(cls, surface_name: str) -> str:
        # Convert camelCase and separators into normalized tokens
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", surface_name.strip())
        tokens = re.findall(r"[a-zA-Z0-9]+", s.lower())
        # Normalize integer tokens ("01" -> "01" kept consistent, strip trailing generic noise words if core tokens exist)
        core_tokens = [t for t in tokens if t not in cls._NOISE_SUFFIX_WORDS]
        if not core_tokens:
            core_tokens = tokens
        return "_".join(core_tokens)

    @classmethod
    def token_set(cls, surface_name: str) -> set[str]:
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", surface_name.strip())
        tokens = set(re.findall(r"[a-z0-9]+", s.lower()))
        core = tokens - cls._NOISE_SUFFIX_WORDS
        return core if core else tokens

    def resolve_extractions(
        self,
        extractions: Sequence[ChunkExtractionResult],
        tbox: EnterpriseCompiledTBox,
        existing_entities: dict[str, KGEntityNode] | None = None,
    ) -> tuple[dict[str, KGEntityNode], list[KGRelationEdge], int, int]:
        """
        Returns:
          - canonical_entities_by_id
          - resolved_edges
          - cross_kb_merged_entity_count
          - sibling_collisions_prevented_count
        """
        canonical_entities: dict[str, KGEntityNode] = (
            dict(existing_entities) if existing_entities else {}
        )
        surface_to_entity_id: dict[str, str] = {}
        slug_to_entity_id: dict[str, str] = {}

        for eid, ent in canonical_entities.items():
            slug_key = f"{ent.upper_category.value}::{self.normalize_entity_slug(ent.canonical_name)}"
            slug_to_entity_id[slug_key] = eid
            surface_to_entity_id[f"{ent.upper_category.value}::{ent.canonical_name.lower()}"] = eid
            for alias in ent.aliases:
                surface_to_entity_id[f"{ent.upper_category.value}::{alias.lower()}"] = eid

        sibling_collisions_prevented = 0

        def _resolve_or_create_entity(
            mention: ExtractedEntityMention,
            kb_id: str,
            chunk_id: str,
        ) -> KGEntityNode:
            nonlocal sibling_collisions_prevented

            canon_cls = tbox.resolve_class(mention.proposed_class)
            cls_def = tbox.classes.get(canon_cls)
            upper_cat = (
                cls_def.upper_category
                if cls_def is not None
                else (mention.upper_category or UpperOntologyCategory.SYSTEM_OR_ASSET)
            )
            gov_bucket = (
                cls_def.status
                if cls_def is not None
                else OntologyStatus.QUARANTINED_CANDIDATE
            )

            exact_key = f"{upper_cat.value}::{mention.surface_name.lower()}"
            if exact_key in surface_to_entity_id:
                target_ent = canonical_entities[surface_to_entity_id[exact_key]]
                self._merge_provenance_into_entity(target_ent, mention, kb_id, chunk_id)
                return target_ent

            slug = self.normalize_entity_slug(mention.surface_name)
            slug_key = f"{upper_cat.value}::{slug}"
            if slug_key in slug_to_entity_id:
                candidate_ent = canonical_entities[slug_to_entity_id[slug_key]]
                can_merge, _ = SiblingCollisionGuard.can_merge(
                    candidate_ent.canonical_name, mention.surface_name
                )
                if can_merge:
                    surface_to_entity_id[exact_key] = candidate_ent.entity_id
                    self._merge_provenance_into_entity(candidate_ent, mention, kb_id, chunk_id)
                    return candidate_ent
                else:
                    sibling_collisions_prevented += 1

            # Check vector + token Jaccard similarity against existing entities of the same class
            mention_vec = self.embedding_provider.embed_text(
                f"{canon_cls} {mention.surface_name}", dim=self.canonical_embedding_dim
            )
            mention_tokens = self.token_set(mention.surface_name)

            best_match: KGEntityNode | None = None
            best_score = 0.0

            for existing_ent in canonical_entities.values():
                if existing_ent.ontology_class != canon_cls:
                    continue
                sim = cosine_similarity(mention_vec, existing_ent.embedding)
                ex_tokens = self.token_set(existing_ent.canonical_name)
                union_len = len(mention_tokens | ex_tokens)
                jaccard = len(mention_tokens & ex_tokens) / union_len if union_len > 0 else 0.0
                combined_score = 0.55 * sim + 0.45 * jaccard

                if combined_score >= 0.72 or sim >= self.similarity_threshold:
                    can_merge, _ = SiblingCollisionGuard.can_merge(
                        existing_ent.canonical_name, mention.surface_name
                    )
                    if not can_merge:
                        sibling_collisions_prevented += 1
                        continue
                    if combined_score > best_score and jaccard >= 0.5:
                        best_score = combined_score
                        best_match = existing_ent

            if best_match is not None:
                surface_to_entity_id[exact_key] = best_match.entity_id
                slug_to_entity_id[slug_key] = best_match.entity_id
                self._merge_provenance_into_entity(best_match, mention, kb_id, chunk_id)
                return best_match

            # Create new canonical entity node
            entity_id = f"ent::{canon_cls.lower()}::{slug}"
            # Prefer clean hyphenated or canonical display form
            new_node = KGEntityNode(
                entity_id=entity_id,
                tenant_id=tbox.tenant_id,
                canonical_name=mention.surface_name,
                ontology_class=canon_cls,
                upper_category=upper_cat,
                aliases=[mention.surface_name],
                summary=mention.summary,
                properties=dict(mention.attributes),
                source_kb_ids=[kb_id],
                source_chunk_ids=[chunk_id],
                mention_count=1,
                confidence=mention.confidence,
                governance_bucket=gov_bucket,
                embedding=mention_vec,
            )
            canonical_entities[entity_id] = new_node
            surface_to_entity_id[exact_key] = entity_id
            slug_to_entity_id[slug_key] = entity_id
            return new_node

        # Pass 1: Resolve all entity mentions across all extracted chunks
        for chunk_res in extractions:
            for ent_mention in chunk_res.entities:
                _resolve_or_create_entity(ent_mention, chunk_res.kb_id, chunk_res.chunk_id)

        # Pass 2: Resolve all relationship triples using canonical entity IDs
        edges_by_signature: dict[str, KGRelationEdge] = {}
        for chunk_res in extractions:
            for triple in chunk_res.triples:
                subj_node = _resolve_or_create_entity(
                    ExtractedEntityMention(
                        surface_name=triple.subject_name,
                        proposed_class=triple.subject_class,
                        confidence=triple.confidence,
                    ),
                    chunk_res.kb_id,
                    chunk_res.chunk_id,
                )
                obj_node = _resolve_or_create_entity(
                    ExtractedEntityMention(
                        surface_name=triple.object_name,
                        proposed_class=triple.object_class,
                        confidence=triple.confidence,
                    ),
                    chunk_res.kb_id,
                    chunk_res.chunk_id,
                )
                canon_pred = tbox.resolve_predicate(triple.predicate)
                rel_def = tbox.relations.get(canon_pred)
                is_func = rel_def.is_functional if rel_def is not None else False
                gov_bucket = (
                    rel_def.status
                    if (
                        rel_def is not None
                        and subj_node.governance_bucket == OntologyStatus.APPROVED
                        and obj_node.governance_bucket == OntologyStatus.APPROVED
                    )
                    else OntologyStatus.QUARANTINED_CANDIDATE
                )

                edge_sig = f"{subj_node.entity_id}--{canon_pred}-->{obj_node.entity_id}"
                if edge_sig not in edges_by_signature:
                    edges_by_signature[edge_sig] = KGRelationEdge(
                        edge_id=f"edge::{subj_node.entity_id}::{canon_pred}::{obj_node.entity_id}",
                        tenant_id=tbox.tenant_id,
                        source_entity_id=subj_node.entity_id,
                        target_entity_id=obj_node.entity_id,
                        predicate=canon_pred,
                        is_functional=is_func,
                        confidence=triple.confidence,
                        t_valid=triple.t_valid,
                        t_invalid="ACTIVE",
                        source_kb_ids=[chunk_res.kb_id],
                        source_chunk_ids=[chunk_res.chunk_id],
                        evidence_snippets=[triple.evidence_snippet] if triple.evidence_snippet else [],
                        governance_bucket=gov_bucket,
                    )
                else:
                    existing_edge = edges_by_signature[edge_sig]
                    if chunk_res.kb_id not in existing_edge.source_kb_ids:
                        existing_edge.source_kb_ids.append(chunk_res.kb_id)
                    if chunk_res.chunk_id not in existing_edge.source_chunk_ids:
                        existing_edge.source_chunk_ids.append(chunk_res.chunk_id)
                    if (
                        triple.evidence_snippet
                        and triple.evidence_snippet not in existing_edge.evidence_snippets
                    ):
                        existing_edge.evidence_snippets.append(triple.evidence_snippet)
                    existing_edge.confidence = max(existing_edge.confidence, triple.confidence)
                    if triple.t_valid > existing_edge.t_valid:
                        existing_edge.t_valid = triple.t_valid

        cross_kb_merged_count = sum(
            1 for e in canonical_entities.values() if len(e.source_kb_ids) >= 2
        )
        return (
            canonical_entities,
            list(edges_by_signature.values()),
            cross_kb_merged_count,
            sibling_collisions_prevented,
        )

    @staticmethod
    def _merge_provenance_into_entity(
        entity: KGEntityNode,
        mention: ExtractedEntityMention,
        kb_id: str,
        chunk_id: str,
    ) -> None:
        if mention.surface_name not in entity.aliases:
            entity.aliases.append(mention.surface_name)
        # Prefer compact hyphenated technical identifier as canonical_name when merging spaced surface variants
        if " " in entity.canonical_name and " " not in mention.surface_name and "-" in mention.surface_name:
            entity.canonical_name = mention.surface_name
        if kb_id not in entity.source_kb_ids:
            entity.source_kb_ids.append(kb_id)
        if chunk_id not in entity.source_chunk_ids:
            entity.source_chunk_ids.append(chunk_id)
        entity.mention_count += 1
        entity.confidence = max(entity.confidence, mention.confidence)
        entity.updated_at = utc_now_iso()
