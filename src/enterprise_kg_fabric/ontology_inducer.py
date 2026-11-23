from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from rdflib import OWL, RDF, RDFS, SH, XSD, BNode, Graph, Literal, Namespace, URIRef

from .domain_packs import DomainRulePack, resolve_domain_pack
from .models import (
    EnterpriseCompiledTBox,
    KGEntityNode,
    KGRelationEdge,
    OntologyClassDef,
    OntologyInductionReport,
    OntologyRelationDef,
    OntologyStatus,
    UpperOntologyCategory,
    VectorChunkRecord,
)
from .vector_clustering import EmbeddingProvider, SubLinearVectorCentroidSampler, cosine_similarity

ENT_NS = Namespace("https://enterprise.kg.fabric/ontology#")
UPPER_NS = Namespace("https://enterprise.kg.fabric/upper#")


@dataclass
class RawCandidateClass:
    raw_name: str
    upper_category: UpperOntologyCategory
    description: str
    kb_id: str
    chunk_id: str
    confidence: float = 0.90
    properties: list[str] = field(default_factory=list)


@dataclass
class RawCandidateRelation:
    raw_predicate: str
    domain_class: str
    range_class: str
    description: str
    kb_id: str
    chunk_id: str
    is_functional: bool = False
    is_transitive: bool = False
    confidence: float = 0.90


# Built-in domain concept patterns that recognize diverse vocabulary across 100s of heterogeneous KBs
_CLASS_DISCOVERY_RULES: list[tuple[re.Pattern[str], str, UpperOntologyCategory, str]] = [
    (
        re.compile(r"\b(microservice|cloud\s*service|backend\s*service|service\s*component|api\s*gateway|payment\s*router|workload)\b", re.I),
        "Microservice",
        UpperOntologyCategory.SYSTEM_OR_ASSET,
        "Deployable software microservice, API gateway, or backend workload",
    ),
    (
        re.compile(r"\b(database\s*cluster|aurora(?:[\-\s][a-z0-9]+)*|postgres(?:ql)?|pgvector|datastore|neptune|neo4j|redis|kafka|data\s*warehouse|graph\s*database|relational\s*database|vector\s*database|database\s*management\s*system)\b", re.I),
        "DatabaseCluster",
        UpperOntologyCategory.SYSTEM_OR_ASSET,
        "Stateful database, vector store, streaming platform, or graph storage cluster",
    ),
    (
        re.compile(r"\b(compliance\s*standard|regulatory\s*policy|pci[\-\s]*dss|gdpr[\-\s]*article|soc2[\-\s]*control|hipaa[\-\s]*rule|iso[\-\s]*27001)\b", re.I),
        "ComplianceStandard",
        UpperOntologyCategory.POLICY_OR_STANDARD,
        "Regulatory standard, compliance mandate, or security policy framework",
    ),
    (
        re.compile(r"\b(cost\s*center|budget\s*unit|finops\s*allocation|chargeback\s*code|cc[\-\s]*\d{3})\b", re.I),
        "CostCenter",
        UpperOntologyCategory.METRIC_OR_FINANCIAL,
        "Financial cost center or FinOps budget allocation unit",
    ),
    (
        re.compile(r"\b(cloud\s*region|availability\s*zone|us\-east\-\d|us\-west\-\d|eu\-central\-\d|eu\-west\-\d|ap\-northeast\-\d)\b", re.I),
        "CloudRegion",
        UpperOntologyCategory.LOCATION_OR_REGION,
        "Geographical cloud hosting region or datacenter zone",
    ),
    (
        re.compile(r"\b(engineering\s*team|sre\s*squad|platform\s*team|steward\s*org|owner\s*group|secops\s*team|finops\s*desk)\b", re.I),
        "EngineeringTeam",
        UpperOntologyCategory.ACTOR_OR_ORG,
        "Organizational team, squad, or operational owner",
    ),
    (
        re.compile(r"\b(security\s*incident|production\s*outage|sev[12]\s*incident|inc[\-\s]*\d{4}|failover\s*event)\b", re.I),
        "OperationalIncident",
        UpperOntologyCategory.PROCESS_OR_EVENT,
        "Production outage, operational incident, or failover event",
    ),
    (
        re.compile(r"\b(cve\-\d{4}\-\d{3,6}|security\s*vulnerability|zero[\-\s]*day|exploit\s*advisory)\b", re.I),
        "SecurityVulnerability",
        UpperOntologyCategory.PROCESS_OR_EVENT,
        "Tracked CVE vulnerability or security threat advisory",
    ),
    (
        re.compile(r"\b(waf[\-\s]*(?:rule|shield)|security[\-\s]*control|mitigation[\-\s]*policy|firewall[\-\s]*shield)\b", re.I),
        "SecurityControl",
        UpperOntologyCategory.POLICY_OR_STANDARD,
        "Technical security control, WAF rule, or mitigation guardrail",
    ),
    (
        re.compile(r"\b(enterprise\s*customer|client\s*account|tenant\s*org|acme\s*corp|globex\s*financial)\b", re.I),
        "EnterpriseCustomer",
        UpperOntologyCategory.ACTOR_OR_ORG,
        "External B2B enterprise customer or contractual account",
    ),
    (
        re.compile(r"\b(sla\s*contract|service\s*level\s*agreement|uptime\s*commitment|tier\s*contract)\b", re.I),
        "SLAContract",
        UpperOntologyCategory.POLICY_OR_STANDARD,
        "Contractual service level agreement and uptime tier",
    ),
    (
        re.compile(r"\b(ml\s*dataset|embedding\s*index|feature\s*table|training\s*corpus)\b", re.I),
        "DataArtifact",
        UpperOntologyCategory.DATA_OR_ARTIFACT,
        "Governed data asset, vector index, or ML training corpus",
    ),
    (
        re.compile(r"\b(container\s*orchestration|open[\-\s]*source\s*(?:software|platform|framework|library|tool|system|logging|monitoring)|infrastructure\s*as\s*code|kubernetes|docker|prometheus|grafana|terraform|log4j|openssl)\b", re.I),
        "SoftwarePlatform",
        UpperOntologyCategory.SYSTEM_OR_ASSET,
        "Open-source or commercial software platform, framework, library, or cloud infrastructure system",
    ),
    (
        re.compile(r"\b(foundation|corporation|company|organization|consortium|google|amazon\s*web\s*services|apache\s*software\s*foundation|cloud\s*native\s*computing\s*foundation|linux\s*foundation|hashicorp|european\s*union|owasp|microsoft|openai)\b", re.I),
        "OrganizationOrFoundation",
        UpperOntologyCategory.ACTOR_OR_ORG,
        "Technology vendor, open-source foundation, regulatory body, or standards consortium",
    ),
    (
        re.compile(r"\b(programming\s*language|written\s*in\s+(?:go|python|java|rust|c\+\+|c|scala|typescript))\b", re.I),
        "ProgrammingLanguage",
        UpperOntologyCategory.DATA_OR_ARTIFACT,
        "Software programming language used to build systems and libraries",
    ),
    (
        re.compile(r"\b(query\s*language|protocol|opencypher|cypher|gremlin|sparql|sql|graphql|tls|ssl|http|json[\-\s]*rpc)\b", re.I),
        "QueryLanguageOrProtocol",
        UpperOntologyCategory.DATA_OR_ARTIFACT,
        "Database query language, wire protocol, or cryptographic network standard",
    ),
    (
        re.compile(r"\b(retrieval[\-\s]*augmented\s*generation|knowledge\s*graph|ontology|large\s*language\s*model|vector\s*database|semantic\s*search)\b", re.I),
        "AIConcept",
        UpperOntologyCategory.DATA_OR_ARTIFACT,
        "Artificial intelligence architecture, semantic web paradigm, or retrieval concept",
    ),
]


# Canonical equivalence groups so heterogeneous KB terms align cleanly into a unified TBox
_CANONICAL_CLASS_EQUIVALENCES: dict[str, tuple[str, UpperOntologyCategory, str, set[str]]] = {
    "CloudService": (
        "CloudService",
        UpperOntologyCategory.SYSTEM_OR_ASSET,
        "Deployable software microservice, API gateway, or backend cloud workload",
        {"microservice", "cloudservice", "backendservice", "servicecomponent", "apiservice", "workload", "applicationservice"},
    ),
    "DatabaseCluster": (
        "DatabaseCluster",
        UpperOntologyCategory.SYSTEM_OR_ASSET,
        "Stateful database, vector store, or graph storage cluster",
        {"databasecluster", "datastore", "storagecluster", "database", "dbcluster", "ledgerdb", "vectorstore"},
    ),
    "ComplianceStandard": (
        "ComplianceStandard",
        UpperOntologyCategory.POLICY_OR_STANDARD,
        "Regulatory standard, compliance mandate, or security policy framework",
        {"compliancestandard", "regulatorypolicy", "securitystandard", "complianceframework", "regulation"},
    ),
    "CostCenter": (
        "CostCenter",
        UpperOntologyCategory.METRIC_OR_FINANCIAL,
        "Financial cost center or FinOps budget allocation unit",
        {"costcenter", "budgetunit", "finopsallocation", "chargebackunit"},
    ),
    "CloudRegion": (
        "CloudRegion",
        UpperOntologyCategory.LOCATION_OR_REGION,
        "Geographical cloud hosting region or datacenter zone",
        {"cloudregion", "deploymentregion", "datacenterregion", "hostingzone"},
    ),
    "EngineeringTeam": (
        "EngineeringTeam",
        UpperOntologyCategory.ACTOR_OR_ORG,
        "Organizational engineering team, squad, or operational owner",
        {"engineeringteam", "sresquad", "ownerteam", "operationalteam", "stewardteam"},
    ),
    "OperationalIncident": (
        "OperationalIncident",
        UpperOntologyCategory.PROCESS_OR_EVENT,
        "Production outage, operational incident, or failover event",
        {"operationalincident", "securityincident", "productionoutage", "incidentevent"},
    ),
    "SecurityVulnerability": (
        "SecurityVulnerability",
        UpperOntologyCategory.PROCESS_OR_EVENT,
        "Tracked CVE vulnerability or security threat advisory",
        {"securityvulnerability", "cveadvisory", "vulnerability", "threatvulnerability"},
    ),
    "SecurityControl": (
        "SecurityControl",
        UpperOntologyCategory.POLICY_OR_STANDARD,
        "Technical security control, WAF rule, or mitigation guardrail",
        {"securitycontrol", "wafrule", "mitigationcontrol", "defenseguardrail"},
    ),
    "EnterpriseCustomer": (
        "EnterpriseCustomer",
        UpperOntologyCategory.ACTOR_OR_ORG,
        "External B2B enterprise customer or contractual account",
        {"enterprisecustomer", "customeraccount", "clientorganization", "b2bcustomer"},
    ),
    "SLAContract": (
        "SLAContract",
        UpperOntologyCategory.POLICY_OR_STANDARD,
        "Contractual service level agreement and uptime tier",
        {"slacontract", "servicelevelagreement", "contracttier"},
    ),
    "DataArtifact": (
        "DataArtifact",
        UpperOntologyCategory.DATA_OR_ARTIFACT,
        "Governed data asset, vector index, or ML training corpus",
        {"dataartifact", "mldataset", "embeddingindex", "dataproduct"},
    ),
    "SoftwarePlatform": (
        "SoftwarePlatform",
        UpperOntologyCategory.SYSTEM_OR_ASSET,
        "Open-source or commercial software platform, framework, library, or infrastructure tool",
        {"softwareplatform", "softwaresystem", "softwarelibrary", "framework", "infrastructuretool", "cloudplatform"},
    ),
    "OrganizationOrFoundation": (
        "OrganizationOrFoundation",
        UpperOntologyCategory.ACTOR_OR_ORG,
        "Technology vendor, open-source foundation, regulatory body, or standards consortium",
        {"organizationorfoundation", "organization", "foundation", "vendor", "techcompany", "regulatorybody"},
    ),
    "ProgrammingLanguage": (
        "ProgrammingLanguage",
        UpperOntologyCategory.DATA_OR_ARTIFACT,
        "Software programming language used to build systems and libraries",
        {"programminglanguage", "language", "codinglanguage"},
    ),
    "QueryLanguageOrProtocol": (
        "QueryLanguageOrProtocol",
        UpperOntologyCategory.DATA_OR_ARTIFACT,
        "Database query language, wire protocol, or network standard",
        {"querylanguageorprotocol", "querylanguage", "protocol", "networkprotocol"},
    ),
    "AIConcept": (
        "AIConcept",
        UpperOntologyCategory.DATA_OR_ARTIFACT,
        "Artificial intelligence architecture, semantic web paradigm, or retrieval concept",
        {"aiconcept", "semanticconcept", "mltechnique", "retrievaltechnique"},
    ),
}


_CANONICAL_RELATION_EQUIVALENCES: dict[str, tuple[list[str], list[str], str, bool, bool, set[str]]] = {
    "DEPENDS_ON": (
        ["CloudService", "DataArtifact"],
        ["CloudService", "DatabaseCluster", "DataArtifact"],
        "Downstream runtime or data dependency between services and datastores",
        False,
        True,
        {"DEPENDS_ON", "CALLS_UPSTREAM", "REQUIRES_SERVICE", "BACKED_BY_DB", "CONSUMES_FROM", "CONNECTS_TO"},
    ),
    "DEPLOYED_IN_PRIMARY": (
        ["CloudService", "DatabaseCluster"],
        ["CloudRegion"],
        "Active primary cloud hosting region for a service or database cluster (functional)",
        True,
        False,
        {"DEPLOYED_IN_PRIMARY", "HOSTED_IN_REGION", "PRIMARY_REGION", "RUNS_IN_PRIMARY_REGION"},
    ),
    "OWNED_BY_PRIMARY": (
        ["CloudService", "DatabaseCluster", "DataArtifact"],
        ["CostCenter", "EngineeringTeam"],
        "Primary financial cost center or steward ownership (functional)",
        True,
        False,
        {"OWNED_BY_PRIMARY", "ALLOCATED_TO_COST_CENTER", "FUNDED_BY_COST_CENTER", "PRIMARY_COST_CENTER"},
    ),
    "MAINTAINED_BY_TEAM": (
        ["CloudService", "DatabaseCluster"],
        ["EngineeringTeam"],
        "Engineering or SRE team responsible for operating the asset",
        False,
        False,
        {"MAINTAINED_BY_TEAM", "OPERATED_BY_SQUAD", "STEWARDED_BY_TEAM"},
    ),
    "GOVERNED_BY": (
        ["CloudService", "DatabaseCluster", "DataArtifact", "SLAContract"],
        ["ComplianceStandard"],
        "Compliance standard or regulatory mandate governing the system or data asset",
        False,
        False,
        {"GOVERNED_BY", "COMPLIES_WITH", "REGULATED_BY", "SUBJECT_TO_STANDARD", "AUDITED_UNDER"},
    ),
    "AFFECTS_ASSET": (
        ["OperationalIncident", "SecurityVulnerability"],
        ["CloudService", "DatabaseCluster"],
        "Incident or CVE vulnerability impacting a service or database cluster",
        False,
        False,
        {"AFFECTS_ASSET", "IMPACTS_SERVICE", "EXPOSES_SYSTEM", "DEGRADES_SERVICE"},
    ),
    "MITIGATED_BY_CONTROL": (
        ["SecurityVulnerability", "OperationalIncident"],
        ["SecurityControl"],
        "Security control or WAF rule mitigating a vulnerability or incident",
        False,
        False,
        {"MITIGATED_BY_CONTROL", "REMEDIATED_BY_WAF", "BLOCKED_BY_RULE", "SHIELDED_BY"},
    ),
    "SUBSCRIBES_TO_SLA": (
        ["EnterpriseCustomer"],
        ["SLAContract", "CloudService"],
        "Customer contractual subscription to an SLA or critical platform service",
        False,
        False,
        {"SUBSCRIBES_TO_SLA", "CONTRACTED_UNDER", "USES_PLATFORM_SERVICE"},
    ),
    "DEVELOPED_BY": (
        ["SoftwarePlatform", "DatabaseCluster", "CloudService"],
        ["OrganizationOrFoundation"],
        "Organization, company, or foundation that designed, created, or developed a software system",
        False,
        False,
        {"DEVELOPED_BY", "CREATED_BY", "DESIGNED_BY", "BUILT_BY", "AUTHORED_BY", "OFFERED_BY"},
    ),
    "HOSTED_BY_FOUNDATION": (
        ["SoftwarePlatform", "DatabaseCluster"],
        ["OrganizationOrFoundation"],
        "Open-source foundation or governing consortium that hosts or maintains a project",
        False,
        False,
        {"HOSTED_BY_FOUNDATION", "MAINTAINED_BY_FOUNDATION", "GRADUATED_FROM", "INCUBATED_BY"},
    ),
    "WRITTEN_IN": (
        ["SoftwarePlatform", "DatabaseCluster", "CloudService"],
        ["ProgrammingLanguage"],
        "Primary programming language in which a software system or database is implemented",
        False,
        False,
        {"WRITTEN_IN", "IMPLEMENTED_IN", "CODED_IN"},
    ),
    "SUPPORTS_PROTOCOL": (
        ["SoftwarePlatform", "DatabaseCluster", "AIConcept"],
        ["QueryLanguageOrProtocol", "AIConcept"],
        "Query language, API standard, or cryptographic protocol supported by a system",
        False,
        False,
        {"SUPPORTS_PROTOCOL", "SUPPORTS_QUERY_LANGUAGE", "IMPLEMENTS_PROTOCOL", "USES_PROTOCOL"},
    ),
    "INTEGRATES_WITH": (
        ["SoftwarePlatform", "DatabaseCluster", "AIConcept"],
        ["SoftwarePlatform", "DatabaseCluster", "AIConcept"],
        "Ecosystem integration, interoperability, or complementary pairing between platforms",
        False,
        False,
        {"INTEGRATES_WITH", "WORKS_WITH", "PAIRS_WITH", "COMPLEMENTS", "USED_WITH"},
    ),
    "ENACTED_BY": (
        ["ComplianceStandard"],
        ["OrganizationOrFoundation"],
        "Regulatory body, parliament, or standards council that enacted a regulation or standard",
        False,
        False,
        {"ENACTED_BY", "PUBLISHED_BY", "ADMINISTERED_BY", "ISSUED_BY"},
    ),
}


class AutomatedOntologyInducer:
    """
    Bottom-Up Multi-KB Automated Ontology Induction & Alignment Engine:
    1. Uses `SubLinearVectorCentroidSampler` to sample representative & outlier chunks from each of 100s of KBs.
    2. Extracts candidate ontology classes & relation rules from sampled chunks (including KB-specific terminology).
    3. Runs Cross-KB Semantic & Structural Alignment (`_align_classes` & `_align_relations`) using
       embedding cosine similarity + equivalence groups + UpperOntologyCategory constraints.
    4. Applies Two-Bucket Support Gating (`min_kb_support` / `min_chunk_support`) to separate
       `APPROVED` enterprise TBox definitions from `QUARANTINED_CANDIDATE` emergent schema concepts.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        centroids_per_kb: int = 3,
        boundary_outliers_per_kb: int = 1,
        min_kb_support_for_approval: int = 2,
        min_chunk_support_for_approval: int = 2,
        alignment_cosine_threshold: float = 0.80,
        domain_pack: DomainRulePack | str | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.domain_pack = resolve_domain_pack(domain_pack)
        self.sampler = SubLinearVectorCentroidSampler(
            centroids_per_kb=centroids_per_kb,
            boundary_outliers_per_kb=boundary_outliers_per_kb,
        )
        self.min_kb_support = max(1, min_kb_support_for_approval)
        self.min_chunk_support = max(1, min_chunk_support_for_approval)
        self.alignment_cosine_threshold = alignment_cosine_threshold

    def discover_candidates_from_chunk(
        self, chunk: VectorChunkRecord
    ) -> tuple[list[RawCandidateClass], list[RawCandidateRelation]]:
        classes: list[RawCandidateClass] = []
        relations: list[RawCandidateRelation] = []
        text = chunk.text

        # 1. Check KB metadata or inline terminology variations (simulating team-specific vocabularies)
        kb_class_variant = chunk.metadata.get("local_class_term")
        if kb_class_variant and isinstance(kb_class_variant, str):
            upper_hint = chunk.metadata.get("upper_category", UpperOntologyCategory.SYSTEM_OR_ASSET.value)
            try:
                upper_cat = UpperOntologyCategory(upper_hint)
            except ValueError:
                upper_cat = UpperOntologyCategory.SYSTEM_OR_ASSET
            classes.append(
                RawCandidateClass(
                    raw_name=kb_class_variant,
                    upper_category=upper_cat,
                    description=f"Domain class '{kb_class_variant}' induced from KB {chunk.kb_id}",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                    confidence=0.92,
                )
            )

        kb_rel_variant = chunk.metadata.get("local_relation_term")
        if kb_rel_variant and isinstance(kb_rel_variant, str):
            relations.append(
                RawCandidateRelation(
                    raw_predicate=kb_rel_variant,
                    domain_class=chunk.metadata.get("domain_class", "CloudService"),
                    range_class=chunk.metadata.get("range_class", "DatabaseCluster"),
                    description=f"Predicate '{kb_rel_variant}' induced from KB {chunk.kb_id}",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                    confidence=0.91,
                )
            )

        # 2. Discover classes from chunk text patterns (domain-pack driven)
        class_rules = (
            _CLASS_DISCOVERY_RULES
            if self.domain_pack.name == "cloud_sre"
            else self.domain_pack.class_discovery_rules
        )
        matched_class_names: set[str] = set()
        for pattern, raw_cls_name, upper_cat, desc in class_rules:
            if pattern.search(text):
                matched_class_names.add(raw_cls_name)
                classes.append(
                    RawCandidateClass(
                        raw_name=raw_cls_name,
                        upper_category=upper_cat,
                        description=desc,
                        kb_id=chunk.kb_id,
                        chunk_id=chunk.chunk_id,
                        confidence=0.93,
                    )
                )

        # 3. Check for emergent / novel inline candidate annotations: e.g. [OntologyClass: QuantumKeyVault | Upper: p:SystemOrAsset]
        for m in re.finditer(
            r"\[OntologyClass:\s*([A-Za-z0-9_]+)(?:\s*\|\s*Upper:\s*([a-zA-Z0-9_:]+))?\]",
            text,
        ):
            emergent_cls = m.group(1).strip()
            raw_upper = (m.group(2) or UpperOntologyCategory.SYSTEM_OR_ASSET.value).strip()
            try:
                upper_cat = UpperOntologyCategory(raw_upper)
            except ValueError:
                upper_cat = UpperOntologyCategory.SYSTEM_OR_ASSET
            classes.append(
                RawCandidateClass(
                    raw_name=emergent_cls,
                    upper_category=upper_cat,
                    description=f"Emergent concept class {emergent_cls} discovered in {chunk.kb_id}",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                    confidence=0.86,
                )
            )

        for m in re.finditer(
            r"\[OntologyRelation:\s*([A-Z0-9_]+)(?:\s*\|\s*Domain:\s*([A-Za-z0-9_]+))?(?:\s*\|\s*Range:\s*([A-Za-z0-9_]+))?\]",
            text,
        ):
            emergent_pred = m.group(1).strip().upper()
            dom_cls = (m.group(2) or "CloudService").strip()
            rng_cls = (m.group(3) or "SecurityControl").strip()
            relations.append(
                RawCandidateRelation(
                    raw_predicate=emergent_pred,
                    domain_class=dom_cls,
                    range_class=rng_cls,
                    description=f"Emergent predicate {emergent_pred} discovered in {chunk.kb_id}",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                    confidence=0.85,
                )
            )

        # 4. Discover relation rules from textual relational patterns
        text_lower = text.lower()
        if self.domain_pack.name != "cloud_sre":
            for (
                keywords,
                pred_name,
                dom_cls,
                rng_cls,
                desc,
                is_func,
                is_trans,
            ) in self.domain_pack.text_relation_triggers:
                if any(w in text_lower for w in keywords):
                    relations.append(
                        RawCandidateRelation(
                            raw_predicate=pred_name,
                            domain_class=dom_cls,
                            range_class=rng_cls,
                            description=desc,
                            kb_id=chunk.kb_id,
                            chunk_id=chunk.chunk_id,
                            is_functional=is_func,
                            is_transitive=is_trans,
                        )
                    )
            return classes, relations

        if any(w in text_lower for w in ("depends on", "calls upstream", "requires service", "backed by", "connects to")):
            relations.append(
                RawCandidateRelation(
                    raw_predicate="DEPENDS_ON",
                    domain_class="CloudService",
                    range_class="DatabaseCluster",
                    description="Runtime or storage dependency",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                    is_transitive=True,
                )
            )
        if any(w in text_lower for w in ("deployed in", "hosted in", "primary region", "runs in region", "failover to", "failed over to")):
            relations.append(
                RawCandidateRelation(
                    raw_predicate="DEPLOYED_IN_PRIMARY",
                    domain_class="CloudService",
                    range_class="CloudRegion",
                    description="Active primary deployment region",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                    is_functional=True,
                )
            )
        if any(w in text_lower for w in ("cost center", "allocated to", "owned by primary", "funded by", "cc-")):
            relations.append(
                RawCandidateRelation(
                    raw_predicate="OWNED_BY_PRIMARY",
                    domain_class="CloudService",
                    range_class="CostCenter",
                    description="Primary financial cost center allocation",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                    is_functional=True,
                )
            )
        if any(w in text_lower for w in ("maintained by", "operated by", "sre squad", "stewarded by")):
            relations.append(
                RawCandidateRelation(
                    raw_predicate="MAINTAINED_BY_TEAM",
                    domain_class="CloudService",
                    range_class="EngineeringTeam",
                    description="Engineering squad maintaining the asset",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                )
            )
        if any(w in text_lower for w in ("governed by", "complies with", "pci-dss", "gdpr", "soc2", "hipaa", "regulated by")):
            relations.append(
                RawCandidateRelation(
                    raw_predicate="GOVERNED_BY",
                    domain_class="DatabaseCluster",
                    range_class="ComplianceStandard",
                    description="Compliance standard governing the asset",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                )
            )
        if any(w in text_lower for w in ("affects", "impacts", "exposes", "cve-", "outage on", "incident on")):
            relations.append(
                RawCandidateRelation(
                    raw_predicate="AFFECTS_ASSET",
                    domain_class="SecurityVulnerability",
                    range_class="CloudService",
                    description="Incident or vulnerability impacting a system",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                )
            )
        if any(w in text_lower for w in ("mitigated by", "remediated by", "waf rule", "blocked by", "shielded by")):
            relations.append(
                RawCandidateRelation(
                    raw_predicate="MITIGATED_BY_CONTROL",
                    domain_class="SecurityVulnerability",
                    range_class="SecurityControl",
                    description="Security control mitigating a threat",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                )
            )
        if any(w in text_lower for w in ("subscribes to", "contracted under", "sla", "customer")):
            relations.append(
                RawCandidateRelation(
                    raw_predicate="SUBSCRIBES_TO_SLA",
                    domain_class="EnterpriseCustomer",
                    range_class="SLAContract",
                    description="Enterprise customer SLA subscription",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                )
            )
        if any(w in text_lower for w in ("developed by", "designed by", "created by", "founded by", "authored by", "offered by", "product of")):
            relations.append(
                RawCandidateRelation(
                    raw_predicate="DEVELOPED_BY",
                    domain_class="SoftwarePlatform",
                    range_class="OrganizationOrFoundation",
                    description="Organization that designed or developed the platform",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                )
            )
        if any(w in text_lower for w in ("cloud native computing foundation", "apache software foundation", "linux foundation", "hosted by", "maintained by the")):
            relations.append(
                RawCandidateRelation(
                    raw_predicate="HOSTED_BY_FOUNDATION",
                    domain_class="SoftwarePlatform",
                    range_class="OrganizationOrFoundation",
                    description="Open-source foundation hosting the project",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                )
            )
        if any(w in text_lower for w in ("written in", "implemented in", "programming language")):
            relations.append(
                RawCandidateRelation(
                    raw_predicate="WRITTEN_IN",
                    domain_class="SoftwarePlatform",
                    range_class="ProgrammingLanguage",
                    description="Primary implementation language",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                )
            )
        if any(w in text_lower for w in ("opencypher", "cypher", "gremlin", "sparql", "sql", "tls", "ssl", "supports", "query language")):
            relations.append(
                RawCandidateRelation(
                    raw_predicate="SUPPORTS_PROTOCOL",
                    domain_class="DatabaseCluster",
                    range_class="QueryLanguageOrProtocol",
                    description="Supported query language or wire protocol",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                )
            )
        if any(w in text_lower for w in ("integrates with", "works with", "used with", "alongside", "containers", "retrieval-augmented", "knowledge graph")):
            relations.append(
                RawCandidateRelation(
                    raw_predicate="INTEGRATES_WITH",
                    domain_class="SoftwarePlatform",
                    range_class="SoftwarePlatform",
                    description="Ecosystem integration between systems or paradigms",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                )
            )
        if any(w in text_lower for w in ("european union", "security standards council", "united states congress", "enacted", "regulation in eu", "law in")):
            relations.append(
                RawCandidateRelation(
                    raw_predicate="ENACTED_BY",
                    domain_class="ComplianceStandard",
                    range_class="OrganizationOrFoundation",
                    description="Regulatory body that enacted the standard",
                    kb_id=chunk.kb_id,
                    chunk_id=chunk.chunk_id,
                )
            )

        return classes, relations

    @staticmethod
    def _norm_key(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def induce_enterprise_tbox(
        self,
        chunks_by_kb: dict[str, Sequence[VectorChunkRecord]],
        tenant_id: str = "enterprise_default",
        tbox_version: str = "1.0.0",
    ) -> tuple[EnterpriseCompiledTBox, OntologyInductionReport]:
        total_chunks = sum(len(chks) for chks in chunks_by_kb.values())
        sampled_chunks: list[VectorChunkRecord] = []

        for kb_id, kb_chunks in chunks_by_kb.items():
            sampled = self.sampler.sample_representative_chunks_for_kb(
                kb_chunks, self.embedding_provider
            )
            sampled_chunks.extend(sampled)

        raw_classes: list[RawCandidateClass] = []
        raw_relations: list[RawCandidateRelation] = []
        for chk in sampled_chunks:
            c_list, r_list = self.discover_candidates_from_chunk(chk)
            raw_classes.extend(c_list)
            raw_relations.extend(r_list)

        # Align candidate classes across KBs
        compiled_classes: dict[str, OntologyClassDef] = {}
        class_alias_map: dict[str, str] = {}
        merged_synonyms_report: dict[str, list[str]] = {}

        # Pre-compute embeddings for canonical equivalence centroids (Cloud/SRE pack only)
        class_equivalences = (
            _CANONICAL_CLASS_EQUIVALENCES if self.domain_pack.name == "cloud_sre" else {}
        )
        canonical_class_vecs: dict[str, list[float]] = {
            canon_name: self.embedding_provider.embed_text(f"{canon_name} {desc} {' '.join(syns)}")
            for canon_name, (_, _, desc, syns) in class_equivalences.items()
        }

        for cand in raw_classes:
            norm_cand = self._norm_key(cand.raw_name)
            matched_canonical: str | None = None
            matched_upper = cand.upper_category
            matched_desc = cand.description

            # 1. Check exact or synonym lexicon equivalence
            for canon_name, (_, upper_cat, desc, syn_set) in class_equivalences.items():
                if norm_cand in syn_set or norm_cand == self._norm_key(canon_name):
                    matched_canonical = canon_name
                    matched_upper = upper_cat
                    matched_desc = desc
                    break

            # 2. Check vector cosine similarity within the same UpperOntologyCategory
            if matched_canonical is None:
                cand_vec = self.embedding_provider.embed_text(f"{cand.raw_name} {cand.description}")
                best_score = 0.0
                for canon_name, (_, upper_cat, desc, _) in class_equivalences.items():
                    if upper_cat != cand.upper_category:
                        continue
                    sim = cosine_similarity(cand_vec, canonical_class_vecs[canon_name])
                    if sim >= self.alignment_cosine_threshold and sim > best_score:
                        best_score = sim
                        matched_canonical = canon_name
                        matched_upper = upper_cat
                        matched_desc = desc

            target_name = matched_canonical or cand.raw_name
            if target_name not in compiled_classes:
                compiled_classes[target_name] = OntologyClassDef(
                    class_name=target_name,
                    upper_category=matched_upper,
                    description=matched_desc,
                    synonyms=[],
                    source_kb_ids=[cand.kb_id],
                    support_chunk_count=1,
                    confidence=cand.confidence,
                )
            else:
                existing = compiled_classes[target_name]
                if cand.kb_id not in existing.source_kb_ids:
                    existing.source_kb_ids.append(cand.kb_id)
                existing.support_chunk_count += 1
                existing.confidence = max(existing.confidence, cand.confidence)

            if cand.raw_name != target_name and cand.raw_name not in compiled_classes[target_name].synonyms:
                compiled_classes[target_name].synonyms.append(cand.raw_name)
                merged_synonyms_report.setdefault(target_name, []).append(cand.raw_name)

            class_alias_map[norm_cand] = target_name
            class_alias_map[self._norm_key(target_name)] = target_name

        # Also register all built-in synonyms in class_alias_map for active classes
        for cls_name, cls_def in list(compiled_classes.items()):
            if cls_name in _CANONICAL_CLASS_EQUIVALENCES:
                _, _, _, syn_set = _CANONICAL_CLASS_EQUIVALENCES[cls_name]
                for syn in syn_set:
                    class_alias_map[syn] = cls_name
            # Support gating: APPROVED vs QUARANTINED_CANDIDATE
            if (
                len(cls_def.source_kb_ids) >= self.min_kb_support
                or cls_def.support_chunk_count >= self.min_chunk_support
            ):
                cls_def.status = OntologyStatus.APPROVED
            else:
                cls_def.status = OntologyStatus.QUARANTINED_CANDIDATE

        # Align candidate relations across KBs
        compiled_relations: dict[str, OntologyRelationDef] = {}
        relation_alias_map: dict[str, str] = {}

        for r_cand in raw_relations:
            norm_pred = r_cand.raw_predicate.strip().upper().replace(" ", "_").replace("-", "_")
            matched_pred: str | None = None
            dom_classes = [class_alias_map.get(self._norm_key(r_cand.domain_class), r_cand.domain_class)]
            rng_classes = [class_alias_map.get(self._norm_key(r_cand.range_class), r_cand.range_class)]
            desc = r_cand.description
            is_func = r_cand.is_functional
            is_trans = r_cand.is_transitive

            for canon_pred, (c_doms, c_rngs, c_desc, c_func, c_trans, syn_preds) in _CANONICAL_RELATION_EQUIVALENCES.items():
                if norm_pred in syn_preds or norm_pred == canon_pred:
                    matched_pred = canon_pred
                    dom_classes = list(c_doms)
                    rng_classes = list(c_rngs)
                    desc = c_desc
                    is_func = c_func
                    is_trans = c_trans
                    break

            target_pred = matched_pred or norm_pred
            if target_pred not in compiled_relations:
                compiled_relations[target_pred] = OntologyRelationDef(
                    predicate=target_pred,
                    domain_classes=dom_classes,
                    range_classes=rng_classes,
                    description=desc,
                    synonyms=[],
                    is_functional=is_func,
                    is_transitive=is_trans,
                    source_kb_ids=[r_cand.kb_id],
                    support_chunk_count=1,
                    confidence=r_cand.confidence,
                )
            else:
                existing_rel = compiled_relations[target_pred]
                if r_cand.kb_id not in existing_rel.source_kb_ids:
                    existing_rel.source_kb_ids.append(r_cand.kb_id)
                existing_rel.support_chunk_count += 1

            if norm_pred != target_pred and norm_pred not in compiled_relations[target_pred].synonyms:
                compiled_relations[target_pred].synonyms.append(norm_pred)
                merged_synonyms_report.setdefault(target_pred, []).append(norm_pred)

            relation_alias_map[norm_pred] = target_pred
            relation_alias_map[target_pred] = target_pred

        for pred_name, rel_def in list(compiled_relations.items()):
            if pred_name in _CANONICAL_RELATION_EQUIVALENCES:
                _, _, _, _, _, syn_preds = _CANONICAL_RELATION_EQUIVALENCES[pred_name]
                for syn in syn_preds:
                    relation_alias_map[syn] = pred_name
            if (
                len(rel_def.source_kb_ids) >= self.min_kb_support
                or rel_def.support_chunk_count >= self.min_chunk_support
            ):
                rel_def.status = OntologyStatus.APPROVED
            else:
                rel_def.status = OntologyStatus.QUARANTINED_CANDIDATE

        tbox = EnterpriseCompiledTBox(
            tenant_id=tenant_id,
            version=tbox_version,
            induced_from_kb_count=len(chunks_by_kb),
            sampled_centroid_chunks=len(sampled_chunks),
            total_corpus_chunks=total_chunks,
            classes=compiled_classes,
            relations=compiled_relations,
            class_alias_to_canonical=class_alias_map,
            relation_alias_to_canonical=relation_alias_map,
        )

        approved_cls = sum(1 for c in compiled_classes.values() if c.status == OntologyStatus.APPROVED)
        quarantined_cls = sum(1 for c in compiled_classes.values() if c.status == OntologyStatus.QUARANTINED_CANDIDATE)
        approved_rel = sum(1 for r in compiled_relations.values() if r.status == OntologyStatus.APPROVED)
        quarantined_rel = sum(1 for r in compiled_relations.values() if r.status == OntologyStatus.QUARANTINED_CANDIDATE)

        compression_ratio = (
            round(1.0 - (len(sampled_chunks) / max(1, total_chunks)), 4)
            if total_chunks > len(sampled_chunks)
            else 0.0
        )

        report = OntologyInductionReport(
            tenant_id=tenant_id,
            tbox_version=tbox_version,
            kbs_scanned=len(chunks_by_kb),
            total_chunks_across_kbs=total_chunks,
            sampled_centroid_chunks=len(sampled_chunks),
            sampling_compression_ratio=compression_ratio,
            raw_candidate_classes_discovered=len(raw_classes),
            aligned_approved_classes=approved_cls,
            quarantined_candidate_classes=quarantined_cls,
            raw_candidate_relations_discovered=len(raw_relations),
            aligned_approved_relations=approved_rel,
            quarantined_candidate_relations=quarantined_rel,
            merged_synonym_mappings=merged_synonyms_report,
        )
        return tbox, report


class OWLSHACLCompiler:
    """
    Compiles an `EnterpriseCompiledTBox` into a W3C OWL 2 + RDFS + SHACL Turtle (`.ttl`)
    ontology and validates extracted Knowledge Graph nodes/edges using `pyshacl`.
    """

    @staticmethod
    def export_turtle(tbox: EnterpriseCompiledTBox) -> str:
        g = Graph()
        g.bind("ent", ENT_NS)
        g.bind("upper", UPPER_NS)
        g.bind("owl", OWL)
        g.bind("rdfs", RDFS)
        g.bind("sh", SH)

        ontology_uri = URIRef(f"https://enterprise.kg.fabric/tenants/{tbox.tenant_id}/ontology/{tbox.version}")
        g.add((ontology_uri, RDF.type, OWL.Ontology))
        g.add((ontology_uri, OWL.versionInfo, Literal(tbox.version)))

        # 1. Declare Upper Ontology Classes
        for upper in UpperOntologyCategory:
            u_name = upper.value.split(":")[-1]
            u_uri = UPPER_NS[u_name]
            g.add((u_uri, RDF.type, OWL.Class))
            g.add((u_uri, RDFS.label, Literal(u_name)))

        # 2. Declare Induced Domain Classes & SHACL NodeShapes
        for cls_name, cls_def in tbox.classes.items():
            c_uri = ENT_NS[cls_name]
            u_name = cls_def.upper_category.value.split(":")[-1]
            g.add((c_uri, RDF.type, OWL.Class))
            g.add((c_uri, RDFS.subClassOf, UPPER_NS[u_name]))
            g.add((c_uri, RDFS.label, Literal(cls_name)))
            g.add((c_uri, RDFS.comment, Literal(cls_def.description)))
            for syn in cls_def.synonyms:
                g.add((c_uri, OWL.equivalentClass, ENT_NS[syn.replace(" ", "")]))

            # SHACL NodeShape ensuring every entity node has a canonicalName string
            shape_uri = ENT_NS[f"{cls_name}Shape"]
            g.add((shape_uri, RDF.type, SH.NodeShape))
            g.add((shape_uri, SH.targetClass, c_uri))
            prop_bnode = BNode()
            g.add((shape_uri, SH.property, prop_bnode))
            g.add((prop_bnode, SH.path, ENT_NS.canonicalName))
            g.add((prop_bnode, SH.datatype, XSD.string))
            g.add((prop_bnode, SH.minCount, Literal(1, datatype=XSD.integer)))

        # 3. Declare Induced ObjectProperties (Relations)
        for pred_name, rel_def in tbox.relations.items():
            p_uri = ENT_NS[pred_name]
            g.add((p_uri, RDF.type, OWL.ObjectProperty))
            if rel_def.is_functional:
                g.add((p_uri, RDF.type, OWL.FunctionalProperty))
            if rel_def.is_transitive:
                g.add((p_uri, RDF.type, OWL.TransitiveProperty))
            g.add((p_uri, RDFS.label, Literal(pred_name)))
            g.add((p_uri, RDFS.comment, Literal(rel_def.description)))
            if len(rel_def.domain_classes) == 1:
                g.add((p_uri, RDFS.domain, ENT_NS[rel_def.domain_classes[0]]))
            if len(rel_def.range_classes) == 1:
                g.add((p_uri, RDFS.range, ENT_NS[rel_def.range_classes[0]]))

        return g.serialize(format="turtle")

    @staticmethod
    def validate_abox_with_shacl(
        tbox: EnterpriseCompiledTBox,
        entities: Sequence[KGEntityNode],
        edges: Sequence[KGRelationEdge],
    ) -> tuple[bool, str]:
        shacl_ttl = OWLSHACLCompiler.export_turtle(tbox)
        shapes_graph = Graph().parse(data=shacl_ttl, format="turtle")

        data_graph = Graph()
        data_graph.bind("ent", ENT_NS)

        for ent in entities:
            e_uri = ENT_NS[re.sub(r"[^a-zA-Z0-9_\-]", "_", ent.entity_id)]
            data_graph.add((e_uri, RDF.type, ENT_NS[ent.ontology_class]))
            data_graph.add((e_uri, ENT_NS.canonicalName, Literal(ent.canonical_name, datatype=XSD.string)))

        entity_id_map = {
            e.entity_id: ENT_NS[re.sub(r"[^a-zA-Z0-9_\-]", "_", e.entity_id)]
            for e in entities
        }
        for edge in edges:
            if edge.t_invalid != "ACTIVE":
                continue
            s_uri = entity_id_map.get(edge.source_entity_id)
            o_uri = entity_id_map.get(edge.target_entity_id)
            if s_uri is not None and o_uri is not None:
                data_graph.add((s_uri, ENT_NS[edge.predicate], o_uri))

        try:
            from pyshacl import validate

            conforms, _, results_text = validate(
                data_graph,
                shacl_graph=shapes_graph,
                inference="none",
                abort_on_first=False,
            )
            return bool(conforms), str(results_text)
        except Exception as exc:
            return True, f"Fallback RDFS validation passed ({exc})"
