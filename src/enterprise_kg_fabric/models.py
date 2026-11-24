from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class UpperOntologyCategory(str, Enum):
    """
    Layer-0 Upper Ontology Categories anchoring all induced domain classes across 100s of KBs.
    Prevents semantic unmooring and drives cross-KB ontology alignment & entity resolution blocking.
    """

    ACTOR_OR_ORG = "p:ActorOrOrg"
    SYSTEM_OR_ASSET = "p:SystemOrAsset"
    PROCESS_OR_EVENT = "p:ProcessOrEvent"
    POLICY_OR_STANDARD = "p:PolicyOrStandard"
    DATA_OR_ARTIFACT = "p:DataOrArtifact"
    LOCATION_OR_REGION = "p:LocationOrRegion"
    METRIC_OR_FINANCIAL = "p:MetricOrFinancial"


class OntologyStatus(str, Enum):
    APPROVED = "APPROVED"
    QUARANTINED_CANDIDATE = "QUARANTINED_CANDIDATE"
    DEPRECATED = "DEPRECATED"


class KBSourceType(str, Enum):
    EXISTING_PGVECTOR_KB = "EXISTING_PGVECTOR_KB"
    MANAGED_RAW_DOC_KB = "MANAGED_RAW_DOC_KB"


class VectorKBColumnMapping(BaseModel):
    """Maps heterogeneous legacy pgvector table columns into the unified reader contract."""

    id_column: str = "chunk_id"
    text_column: str = "content"
    embedding_column: str = "embedding"
    metadata_column: str = "metadata"
    doc_id_column: str = "doc_id"
    updated_at_column: str = "updated_at"


class VectorKBRegistration(BaseModel):
    """
    Metadata registration for one of potentially 100s of enterprise vector KBs.
    Each KB can live in its own Postgres schema/table and have its own native embedding dimension.
    """

    kb_id: str = Field(..., description="Unique KB identifier, e.g. 'kb_014_sre_incidents'")
    tenant_id: str = Field(default="enterprise_default")
    domain_tag: str = Field(..., description="Business domain tag, e.g. 'sre_cloud', 'finops', 'secops', 'compliance'")
    source_type: KBSourceType = Field(default=KBSourceType.EXISTING_PGVECTOR_KB)
    postgres_dsn_alias: str = Field(default="primary_pgvector_cluster")
    schema_name: str = Field(default="public")
    table_name: str = Field(..., description="PostgreSQL pgvector table backing this KB")
    embedding_dim: int = Field(default=768, ge=16, le=4096)
    embedding_model_id: str = Field(default="text-embedding-004")
    column_mapping: VectorKBColumnMapping = Field(default_factory=VectorKBColumnMapping)
    owner_team: str = Field(default="platform-engineering")
    security_classification: str = Field(default="INTERNAL")
    registered_at: str = Field(default_factory=utc_now_iso)


class RawDocumentInput(BaseModel):
    """Raw unstructured or semi-structured document ingested via LlamaIndex node parsing."""

    doc_id: str
    kb_id: str
    tenant_id: str = "enterprise_default"
    domain_tag: str = "general"
    title: str
    source_uri: str = ""
    mime_type: str = "text/markdown"
    content: str
    acl_groups: list[str] = Field(default_factory=lambda: ["public"])
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorChunkRecord(BaseModel):
    """
    Canonical representation of a chunk either reverse-hydrated from an existing pgvector KB
    or produced by the LlamaIndex RawDocumentIngestor.
    """

    chunk_id: str
    kb_id: str
    tenant_id: str = "enterprise_default"
    domain_tag: str = "general"
    doc_id: str
    text: str
    content_sha256: str = ""
    native_embedding: list[float] = Field(default_factory=list)
    native_embedding_dim: int = 0
    canonical_embedding: list[float] = Field(default_factory=list)
    parent_node_id: str | None = None
    prev_node_id: str | None = None
    next_node_id: str | None = None
    section_path: str = ""
    acl_groups: list[str] = Field(default_factory=lambda: ["public"])
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=utc_now_iso)

    def model_post_init(self, __context: Any) -> None:
        if not self.content_sha256:
            self.content_sha256 = compute_sha256(self.text)
        if self.native_embedding and not self.native_embedding_dim:
            self.native_embedding_dim = len(self.native_embedding)


class OntologyClassDef(BaseModel):
    """Terminological Box (TBox) Class Definition with cross-KB alignment provenance."""

    class_name: str
    upper_category: UpperOntologyCategory
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    required_properties: list[str] = Field(default_factory=list)
    source_kb_ids: list[str] = Field(default_factory=list)
    support_chunk_count: int = 1
    confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    status: OntologyStatus = OntologyStatus.APPROVED


class OntologyRelationDef(BaseModel):
    """Terminological Box (TBox) Object Property / Predicate Rule."""

    predicate: str
    domain_classes: list[str] = Field(default_factory=list)
    range_classes: list[str] = Field(default_factory=list)
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    is_functional: bool = False
    is_transitive: bool = False
    inverse_predicate: str | None = None
    source_kb_ids: list[str] = Field(default_factory=list)
    support_chunk_count: int = 1
    confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    status: OntologyStatus = OntologyStatus.APPROVED


class EnterpriseCompiledTBox(BaseModel):
    """
    Compiled & aligned Enterprise Ontology across 100s of KBs.
    Includes Approved classes/predicates, Quarantined candidates, and alias lookup maps.
    """

    tenant_id: str = "enterprise_default"
    version: str = "1.0.0"
    induced_from_kb_count: int = 0
    sampled_centroid_chunks: int = 0
    total_corpus_chunks: int = 0
    classes: dict[str, OntologyClassDef] = Field(default_factory=dict)
    relations: dict[str, OntologyRelationDef] = Field(default_factory=dict)
    class_alias_to_canonical: dict[str, str] = Field(default_factory=dict)
    relation_alias_to_canonical: dict[str, str] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=utc_now_iso)

    def resolve_class(self, raw_class_name: str) -> str:
        norm = raw_class_name.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        return self.class_alias_to_canonical.get(norm, raw_class_name.strip())

    def resolve_predicate(self, raw_predicate: str) -> str:
        norm = raw_predicate.strip().upper().replace(" ", "_").replace("-", "_")
        return self.relation_alias_to_canonical.get(norm, norm)


class ExtractedEntityMention(BaseModel):
    """Raw entity extracted from a chunk by LlamaIndex Schema/Dynamic Path Extractor."""

    surface_name: str
    proposed_class: str
    upper_category: UpperOntologyCategory | None = None
    summary: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    is_novel_class: bool = False


class ExtractedRelationTriple(BaseModel):
    """Raw relation triple extracted from a chunk by LlamaIndex Schema/Dynamic Path Extractor."""

    subject_name: str
    subject_class: str
    predicate: str
    object_name: str
    object_class: str
    evidence_snippet: str = ""
    confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    t_valid: str = Field(default_factory=utc_now_iso)
    is_novel_predicate: bool = False
    qualifiers: dict[str, Any] = Field(
        default_factory=dict,
        description="N-ary reified claim qualifiers e.g. {'condition': 'latency > 200ms', 'approved_by': 'vp_sre'}",
    )


class ChunkExtractionResult(BaseModel):
    chunk_id: str
    kb_id: str
    acl_groups: list[str] = Field(default_factory=lambda: ["public"])
    entities: list[ExtractedEntityMention] = Field(default_factory=list)
    triples: list[ExtractedRelationTriple] = Field(default_factory=list)


class KGEntityNode(BaseModel):
    """
    Canonical Knowledge Graph Entity Node (ABox) stored in Neo4j/Neptune
    and indexed in PostgreSQL pgvector `canonical_entity_embeddings`.
    """

    entity_id: str
    tenant_id: str = "enterprise_default"
    canonical_name: str
    ontology_class: str
    upper_category: UpperOntologyCategory
    aliases: list[str] = Field(default_factory=list)
    summary: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)
    source_kb_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    acl_groups: list[str] = Field(default_factory=lambda: ["public"])
    mention_count: int = 1
    confidence: float = 0.92
    governance_bucket: OntologyStatus = OntologyStatus.APPROVED
    embedding: list[float] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utc_now_iso)


class KGRelationEdge(BaseModel):
    """
    Bi-Temporal Knowledge Graph Relationship Edge (ABox) in Neo4j/Neptune
    linking canonical entities across KBs, with N-ary reified qualifiers and ABAC ACLs.
    """

    edge_id: str
    tenant_id: str = "enterprise_default"
    source_entity_id: str
    target_entity_id: str
    predicate: str
    is_functional: bool = False
    confidence: float = 0.90
    t_valid: str = Field(default_factory=utc_now_iso)
    t_invalid: str = "ACTIVE"
    source_kb_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    acl_groups: list[str] = Field(default_factory=lambda: ["public"])
    governance_bucket: OntologyStatus = OntologyStatus.APPROVED


class KBWatermarkState(BaseModel):
    """Incremental Change-Data-Capture (CDC) watermark per (kb_id, chunk_id)."""

    kb_id: str
    chunk_id: str
    content_sha256: str
    tbox_version: str
    last_indexed_at: str = Field(default_factory=utc_now_iso)


class OntologyInductionReport(BaseModel):
    tenant_id: str
    tbox_version: str
    kbs_scanned: int
    total_chunks_across_kbs: int
    sampled_centroid_chunks: int
    sampling_compression_ratio: float
    raw_candidate_classes_discovered: int
    aligned_approved_classes: int
    quarantined_candidate_classes: int
    raw_candidate_relations_discovered: int
    aligned_approved_relations: int
    quarantined_candidate_relations: int
    merged_synonym_mappings: dict[str, list[str]] = Field(default_factory=dict)


class KGGraphBuildReport(BaseModel):
    tenant_id: str
    tbox_version: str
    kbs_processed: int
    total_chunks_inspected: int
    delta_chunks_extracted: int
    skipped_unchanged_chunks_via_watermark: int
    raw_entity_mentions: int
    canonical_entities_after_er: int
    cross_kb_merged_entities: int
    sibling_collisions_prevented: int
    active_edges_upserted: int
    superseded_bitemporal_edges_closed: int
    shacl_conformance_passed: bool


class QueryRouteDecision(str, Enum):
    """Adaptive-RAG query routing strategies."""

    FACTUAL_VECTOR = "FACTUAL_VECTOR"
    LOCAL_MULTIHOP_PPR = "LOCAL_MULTIHOP_PPR"
    STRUCTURED_AGGREGATION = "STRUCTURED_AGGREGATION"
    GLOBAL_SENSEMAKING = "GLOBAL_SENSEMAKING"


class GraphRAGQueryRequest(BaseModel):
    tenant_id: str = "enterprise_default"
    query: str
    kb_filter: list[str] | None = None
    domain_filter: list[str] | None = None
    caller_acl_groups: list[str] | None = Field(
        default=None,
        description="Optional ABAC caller groups (e.g. ['public', 'sre']). None disables ACL filtering for admin queries.",
    )
    top_k_chunks: int = Field(default=6, ge=1, le=50)
    max_graph_hops: int = Field(default=2, ge=1, le=4)
    include_quarantined_overflow: bool = False
    as_of_iso: str | None = None
    use_adaptive_routing: bool = False
    use_ppr_ranking: bool = True


class RetrievedChunkContext(BaseModel):
    chunk_id: str
    kb_id: str
    domain_tag: str
    doc_id: str
    text: str
    rrf_score: float
    retrieval_channels: list[str] = Field(default_factory=list)
    acl_groups: list[str] = Field(default_factory=lambda: ["public"])


class RetrievedGraphFact(BaseModel):
    edge_id: str
    source_name: str
    source_class: str
    predicate: str
    target_name: str
    target_class: str
    hop_distance: int
    source_kb_ids: list[str]
    is_cross_kb_bridge: bool
    t_valid: str
    t_invalid: str
    confidence: float
    ppr_score: float = 0.0
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    acl_groups: list[str] = Field(default_factory=lambda: ["public"])


class GraphRAGQueryResponse(BaseModel):
    query: str
    tenant_id: str
    tbox_version: str
    route_taken: QueryRouteDecision = QueryRouteDecision.LOCAL_MULTIHOP_PPR
    synthesized_answer: str
    generated_opencypher_query: str
    opencypher_results: list[dict[str, Any]] = Field(default_factory=list)
    seed_entities_matched: list[str] = Field(default_factory=list)
    cross_kb_bridge_entities: list[str] = Field(default_factory=list)
    graph_facts: list[RetrievedGraphFact] = Field(default_factory=list)
    chunk_contexts: list[RetrievedChunkContext] = Field(default_factory=list)
    community_summaries_used: list[dict[str, Any]] = Field(default_factory=list)
    provenance_kbs_contributing: list[str] = Field(default_factory=list)
    retrieval_latency_ms: float = 0.0


class ChunkSearchRequest(BaseModel):
    """Granular multi-KB vector + BM25 search primitive for autonomous agents."""

    tenant_id: str = "enterprise_default"
    query: str
    top_k: int = Field(default=6, ge=1, le=50)
    kb_filter: list[str] | None = None
    domain_filter: list[str] | None = None
    caller_acl_groups: list[str] | None = None


class EntitySearchRequest(BaseModel):
    """Granular entity lookup & resolution primitive across all KBs."""

    tenant_id: str = "enterprise_default"
    query: str
    ontology_class_filter: str | None = None
    top_k: int = Field(default=8, ge=1, le=50)
    include_quarantined: bool = False
    caller_acl_groups: list[str] | None = None


class EntitySearchMatch(BaseModel):
    entity_id: str
    canonical_name: str
    ontology_class: str
    upper_category: UpperOntologyCategory
    aliases: list[str] = Field(default_factory=list)
    source_kb_ids: list[str] = Field(default_factory=list)
    acl_groups: list[str] = Field(default_factory=lambda: ["public"])
    is_cross_kb_bridge: bool = False
    match_score: float = 0.0


class SubgraphSearchRequest(BaseModel):
    """Granular K-hop bi-temporal subgraph traversal primitive around specific entities."""

    tenant_id: str = "enterprise_default"
    seed_entities: list[str] = Field(
        ..., description="List of entity_ids or canonical entity names (e.g. ['payment-router-01'])"
    )
    max_hops: int = Field(default=2, ge=1, le=4)
    predicate_filter: list[str] | None = None
    as_of_iso: str | None = None
    include_quarantined: bool = False
    caller_acl_groups: list[str] | None = None


class SubgraphSearchResponse(BaseModel):
    tenant_id: str
    resolved_seed_entity_ids: list[str] = Field(default_factory=list)
    cross_kb_bridge_entities: list[str] = Field(default_factory=list)
    facts: list[RetrievedGraphFact] = Field(default_factory=list)


class ReadOnlyCypherSearchRequest(BaseModel):
    """Guarded read-only OpenCypher query primitive for ontology-aware agents."""

    tenant_id: str = "enterprise_default"
    cypher_query: str
    params: dict[str, Any] = Field(default_factory=dict)


class ReadOnlyCypherSearchResponse(BaseModel):
    tenant_id: str
    cypher_query: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0


class CommunitySummaryRecord(BaseModel):
    """Hierarchical Leiden/Louvain community summary (Microsoft GraphRAG Global Search)."""

    community_id: str
    tenant_id: str
    level: int = 0
    title: str
    summary: str
    member_entity_names: list[str] = Field(default_factory=list)
    member_classes: list[str] = Field(default_factory=list)
    contributing_kb_ids: list[str] = Field(default_factory=list)
    top_predicates: list[str] = Field(default_factory=list)
    high_level_themes: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)


class GlobalCommunitySearchRequest(BaseModel):
    tenant_id: str = "enterprise_default"
    query: str
    top_k_communities: int = Field(default=4, ge=1, le=20)
    caller_acl_groups: list[str] | None = None


class GlobalCommunitySearchResponse(BaseModel):
    tenant_id: str
    query: str
    synthesized_global_answer: str
    communities_matched: list[CommunitySummaryRecord] = Field(default_factory=list)
    engine_used: str = "leiden_or_louvain"


class PPRSearchRequest(BaseModel):
    """HippoRAG 2 Personalized PageRank hub-damped subgraph retrieval request."""

    tenant_id: str = "enterprise_default"
    seed_entities: list[str]
    top_k_facts: int = Field(default=15, ge=1, le=100)
    alpha: float = Field(default=0.85, ge=0.1, le=0.99)
    as_of_iso: str | None = None
    caller_acl_groups: list[str] | None = None


class PPRSearchResponse(BaseModel):
    tenant_id: str
    engine_used: str
    top_ranked_nodes: list[dict[str, Any]] = Field(default_factory=list)
    ranked_facts: list[RetrievedGraphFact] = Field(default_factory=list)


class SyncJobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DLQFailedChunkRecord(BaseModel):
    dlq_id: str
    tenant_id: str
    kb_id: str
    chunk_id: str
    error_message: str
    retry_count: int = 0
    failed_at: str = Field(default_factory=utc_now_iso)


class SyncJobRecord(BaseModel):
    job_id: str
    tenant_id: str
    status: SyncJobStatus = SyncJobStatus.QUEUED
    incremental: bool = True
    report: KGGraphBuildReport | None = None
    dlq_count: int = 0
    error: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    completed_at: str | None = None


class EvalTestCase(BaseModel):
    query: str
    expected_facts: list[str] = Field(default_factory=list)
    forbidden_facts: list[str] = Field(default_factory=list)
    caller_acl_groups: list[str] | None = None
    as_of_iso: str | None = None


class EvalSuiteReport(BaseModel):
    tenant_id: str
    cases_evaluated: int
    mean_multihop_recall: float
    vector_only_baseline_recall: float = 0.0
    graphrag_recall_lift: float = 0.0
    abac_zero_leakage_rate: float
    citation_groundedness_rate: float
    ragas_available: bool = False
    case_details: list[dict[str, Any]] = Field(default_factory=list)


class QualityIssueFinding(BaseModel):
    severity: str  # "HIGH" | "MEDIUM" | "LOW" | "INFO"
    category: str  # "TBOX" | "EXTRACTION" | "ENTITY_RESOLUTION" | "TOPOLOGY" | "GOVERNANCE"
    message: str
    affected_ids: list[str] = Field(default_factory=list)
    remediation: str = ""


class OntologyTBoxQualityMetrics(BaseModel):
    tbox_version: str
    approved_classes: int = 0
    quarantined_classes: int = 0
    approved_relations: int = 0
    quarantined_relations: int = 0
    shacl_conforms: bool = True
    shacl_violation_count: int = 0
    schema_utilization_rate: float = 1.0
    unused_classes: list[str] = Field(default_factory=list)
    unused_relations: list[str] = Field(default_factory=list)
    near_duplicate_class_pairs: list[dict[str, Any]] = Field(default_factory=list)
    domain_range_compliance_rate: float = 1.0
    domain_range_violations_count: int = 0


class ExtractionAndERQualityMetrics(BaseModel):
    total_chunks_evaluated: int = 0
    zero_extraction_chunk_count: int = 0
    zero_extraction_chunk_rate: float = 0.0
    dlq_failed_chunks_count: int = 0
    provenance_grounded_edge_rate: float = 1.0
    sampled_edges_checked: int = 0
    sampled_triple_faithfulness_score: float = 1.0
    faithfulness_judge_mode: str = "deterministic_evidence_entailment"
    suspected_unmerged_duplicate_pairs: list[dict[str, Any]] = Field(default_factory=list)
    over_merged_suspects: list[dict[str, Any]] = Field(default_factory=list)


class GraphTopologyQualityMetrics(BaseModel):
    canonical_entities_count: int = 0
    active_edges_count: int = 0
    superseded_edges_count: int = 0
    orphan_entity_count: int = 0
    orphan_entity_rate: float = 0.0
    connected_components_count: int = 0
    giant_component_ratio: float = 0.0
    registered_kbs_count: int = 0
    connected_kbs_count: int = 0
    cross_kb_linkage_coverage: float = 0.0
    cross_kb_bridge_entity_count: int = 0
    average_degree: float = 0.0
    top_hub_degree_share: float = 0.0
    top_hubs: list[dict[str, Any]] = Field(default_factory=list)
    community_count: int = 0
    community_modularity: float = 0.0


class KGQualityScorecardReport(BaseModel):
    tenant_id: str
    generated_at: str = Field(default_factory=utc_now_iso)
    overall_score: float  # 0.0 - 100.0
    grade: str  # "A+" | "A" | "A-" | "B+" | "B" | "C" | "D" | "F"
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    tbox_metrics: OntologyTBoxQualityMetrics
    extraction_er_metrics: ExtractionAndERQualityMetrics
    topology_metrics: GraphTopologyQualityMetrics
    findings: list[QualityIssueFinding] = Field(default_factory=list)


class AnswerabilityCheckRequest(BaseModel):
    query: str
    tenant_id: str = "enterprise_default"
    caller_acl_groups: list[str] | None = None


class AnswerabilityCheckResponse(BaseModel):
    query: str
    tenant_id: str
    answerable: bool
    status: str  # "ANSWERABLE" | "PARTIAL" | "UNANSWERABLE"
    coverage_score: float
    matched_tbox_classes: list[str] = Field(default_factory=list)
    matched_entities: list[str] = Field(default_factory=list)
    unmatched_terms: list[str] = Field(default_factory=list)
    evidence_counts: dict[str, int] = Field(default_factory=dict)
    clarification_hint: str | None = None
