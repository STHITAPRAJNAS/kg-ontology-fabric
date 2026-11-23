from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import UpperOntologyCategory


@dataclass(frozen=True)
class DomainRulePack:
    """
    Pluggable heuristic rule pack used ONLY as the deterministic offline fallback when
    live LLM structured extraction is disabled or unavailable.

    PyPI adopters in non-Cloud domains (Healthcare, Legal, Finance, Scientific Research)
    can select `GENERIC_DOMAIN_PACK` (`KG_FABRIC_DOMAIN_PACK=generic`) or supply their own
    `DomainRulePack` without modifying library internals.
    """

    name: str
    description: str
    class_discovery_rules: list[tuple[re.Pattern[str], str, UpperOntologyCategory, str]] = field(
        default_factory=list
    )
    nl_relation_patterns: list[tuple[re.Pattern[str], str, str, str]] = field(
        default_factory=list
    )
    # (keywords_any, canonical_predicate, domain_class, range_class, description, is_functional, is_transitive)
    text_relation_triggers: list[tuple[tuple[str, ...], str, str, str, str, bool, bool]] = field(
        default_factory=list
    )


CLOUD_SRE_DOMAIN_PACK = DomainRulePack(
    name="cloud_sre",
    description=(
        "Built-in enterprise Cloud Infrastructure, SRE, FinOps, SecOps, Compliance, "
        "and General Knowledge heuristic fallback pack."
    ),
    class_discovery_rules=[
        (
            re.compile(
                r"\b(microservice|cloud\s*service|backend\s*service|service\s*component|api\s*gateway|payment\s*router|workload)\b",
                re.I,
            ),
            "Microservice",
            UpperOntologyCategory.SYSTEM_OR_ASSET,
            "Deployable software microservice, API gateway, or backend workload",
        ),
        (
            re.compile(
                r"\b(database\s*cluster|aurora(?:[\-\s][a-z0-9]+)*|postgres(?:ql)?|pgvector|datastore|neptune|neo4j|redis|kafka|data\s*warehouse|graph\s*database|relational\s*database|vector\s*database|database\s*management\s*system)\b",
                re.I,
            ),
            "DatabaseCluster",
            UpperOntologyCategory.SYSTEM_OR_ASSET,
            "Stateful database, vector store, streaming platform, or graph storage cluster",
        ),
        (
            re.compile(
                r"\b(compliance\s*standard|regulatory\s*policy|pci[\-\s]*dss|gdpr[\-\s]*article|soc2[\-\s]*control|hipaa[\-\s]*rule|iso[\-\s]*27001)\b",
                re.I,
            ),
            "ComplianceStandard",
            UpperOntologyCategory.POLICY_OR_STANDARD,
            "Regulatory standard, compliance mandate, or security policy framework",
        ),
        (
            re.compile(
                r"\b(cost\s*center|budget\s*unit|finops\s*allocation|chargeback\s*code|cc[\-\s]*\d{3})\b",
                re.I,
            ),
            "CostCenter",
            UpperOntologyCategory.METRIC_OR_FINANCIAL,
            "Financial cost center or FinOps budget allocation unit",
        ),
        (
            re.compile(
                r"\b(cloud\s*region|availability\s*zone|us\-east\-\d|us\-west\-\d|eu\-central\-\d|eu\-west\-\d|ap\-northeast\-\d)\b",
                re.I,
            ),
            "CloudRegion",
            UpperOntologyCategory.LOCATION_OR_REGION,
            "Geographical cloud hosting region or datacenter zone",
        ),
        (
            re.compile(
                r"\b(engineering\s*team|sre\s*squad|platform\s*team|steward\s*org|owner\s*group|secops\s*team|finops\s*desk)\b",
                re.I,
            ),
            "EngineeringTeam",
            UpperOntologyCategory.ACTOR_OR_ORG,
            "Organizational team, squad, or operational owner",
        ),
        (
            re.compile(
                r"\b(security\s*incident|production\s*outage|sev[12]\s*incident|inc[\-\s]*\d{4}|failover\s*event)\b",
                re.I,
            ),
            "OperationalIncident",
            UpperOntologyCategory.PROCESS_OR_EVENT,
            "Production outage, security incident, or failover event",
        ),
        (
            re.compile(r"\b(cve[\-\s]*\d{4}[\-\s]*\d{3,6}|zero[\-\s]*day|vulnerability)\b", re.I),
            "SecurityVulnerability",
            UpperOntologyCategory.PROCESS_OR_EVENT,
            "Tracked security vulnerability or CVE advisory",
        ),
        (
            re.compile(
                r"\b(waf\s*rule|security\s*control|iam\s*policy|kms\s*key|firewall\s*rule|shield\s*rule)\b",
                re.I,
            ),
            "SecurityControl",
            UpperOntologyCategory.POLICY_OR_STANDARD,
            "Preventive or detective security mitigation control",
        ),
        (
            re.compile(r"\b(enterprise\s*customer|vip\s*account|client\s*tenant|strategic\s*account)\b", re.I),
            "EnterpriseCustomer",
            UpperOntologyCategory.ACTOR_OR_ORG,
            "External enterprise customer or strategic client account",
        ),
        (
            re.compile(r"\b(sla\s*tier|uptime\s*contract|service\s*level\s*agreement|rto\s*target)\b", re.I),
            "SLAContract",
            UpperOntologyCategory.POLICY_OR_STANDARD,
            "Contractual SLA availability or recovery obligation",
        ),
        (
            re.compile(
                r"\b(kubernetes|docker|prometheus|terraform|envoy|grpc|open\s*telemetry|apache\s+kafka|redis|langchain|llamaindex|cloud\s*native)\b",
                re.I,
            ),
            "SoftwarePlatform",
            UpperOntologyCategory.SYSTEM_OR_ASSET,
            "Open-source or cloud-native software platform, framework, or runtime",
        ),
        (
            re.compile(
                r"\b(python|go\s+programming|rust\s+programming|c\+\+|java\s+programming|typescript|programming\s+language)\b",
                re.I,
            ),
            "ProgrammingLanguage",
            UpperOntologyCategory.SYSTEM_OR_ASSET,
            "Programming language used to implement systems or SDKs",
        ),
        (
            re.compile(
                r"\b(cypher|sparql|graphql|sql|owl|shacl|rdf|model\s+context\s+protocol|query\s+language|semantic\s+web)\b",
                re.I,
            ),
            "QueryLanguageOrProtocol",
            UpperOntologyCategory.POLICY_OR_STANDARD,
            "Query language, ontology standard, or interoperability protocol",
        ),
        (
            re.compile(
                r"\b(large\s+language\s+model|retrieval[\-\s]augmented\s+generation|knowledge\s+graph|transformer|vector\s+embedding|word\s+embedding|ontology)\b",
                re.I,
            ),
            "AIConcept",
            UpperOntologyCategory.DATA_OR_ARTIFACT,
            "Artificial intelligence, knowledge representation, or retrieval concept",
        ),
        (
            re.compile(
                r"\b(google|amazon\s+web\s+services|microsoft|openai|anthropic|linux\s+foundation|cncf|w3c|apache\s+software\s+foundation|university\s+of\s+california|meta\s+ai)\b",
                re.I,
            ),
            "OrganizationOrFoundation",
            UpperOntologyCategory.ACTOR_OR_ORG,
            "Company, standards body, research lab, or open-source foundation",
        ),
    ],
    text_relation_triggers=[
        (
            ("depends on", "calls upstream", "requires service", "backed by", "connects to"),
            "DEPENDS_ON",
            "CloudService",
            "DatabaseCluster",
            "Runtime or storage dependency",
            False,
            True,
        ),
        (
            ("deployed in", "hosted in", "primary region", "runs in region", "failover to", "failed over to"),
            "DEPLOYED_IN_PRIMARY",
            "CloudService",
            "CloudRegion",
            "Active primary deployment region",
            True,
            False,
        ),
        (
            ("cost center", "allocated to", "owned by primary", "funded by", "cc-"),
            "OWNED_BY_PRIMARY",
            "CloudService",
            "CostCenter",
            "Primary financial cost center allocation",
            True,
            False,
        ),
        (
            ("maintained by", "operated by", "sre squad", "stewarded by"),
            "MAINTAINED_BY_TEAM",
            "CloudService",
            "EngineeringTeam",
            "Engineering squad maintaining the asset",
            False,
            False,
        ),
        (
            ("governed by", "complies with", "subject to", "regulated by", "audited under"),
            "GOVERNED_BY",
            "DatabaseCluster",
            "ComplianceStandard",
            "Regulatory or compliance governance relationship",
            False,
            False,
        ),
        (
            ("affects", "impacts", "exposes", "degrades"),
            "AFFECTS_ASSET",
            "SecurityVulnerability",
            "CloudService",
            "Vulnerability or incident blast-radius impact",
            False,
            False,
        ),
        (
            ("mitigated by", "remediated by", "blocked by", "shielded by"),
            "MITIGATED_BY_CONTROL",
            "SecurityVulnerability",
            "SecurityControl",
            "Security control mitigating a threat or CVE",
            False,
            False,
        ),
        (
            ("subscribes to", "contracted under", "sla", "vip account"),
            "SUBSCRIBES_TO_SLA",
            "EnterpriseCustomer",
            "CloudService",
            "Customer subscription or SLA dependency on service",
            False,
            False,
        ),
        (
            ("created by", "developed by", "designed by", "founded by", "maintained by", "standardized by", "introduced by", "published by"),
            "DEVELOPED_BY",
            "SoftwarePlatform",
            "OrganizationOrFoundation",
            "Creator, maintainer, or standards organization behind a technology",
            False,
            False,
        ),
        (
            ("written in", "implemented in", "built with", "programmed in"),
            "WRITTEN_IN",
            "SoftwarePlatform",
            "ProgrammingLanguage",
            "Implementation programming language of a system or platform",
            False,
            False,
        ),
        (
            ("supports", "uses", "integrates with", "queried using", "compatible with", "extends", "built on"),
            "USES_TECHNOLOGY",
            "SoftwarePlatform",
            "QueryLanguageOrProtocol",
            "Protocol, query language, standard, or technology dependency",
            False,
            False,
        ),
        (
            ("part of", "hosted by", "donated to", "graduated from", "incubated at"),
            "HOSTED_BY_ORG",
            "SoftwarePlatform",
            "OrganizationOrFoundation",
            "Foundation or umbrella organization stewarding a project",
            False,
            False,
        ),
    ],
)


GENERIC_DOMAIN_PACK = DomainRulePack(
    name="generic",
    description=(
        "Domain-neutral fallback pack (organizations, standards/policies, locations, concepts, "
        "and generic relationships). Ideal for Healthcare, Legal, Finance, or LLM-first deployments."
    ),
    class_discovery_rules=[
        (
            re.compile(r"\b(organization|company|institution|foundation|committee|department|agency|hospital|bank)\b", re.I),
            "Organization",
            UpperOntologyCategory.ACTOR_OR_ORG,
            "Organization, institution, company, or department",
        ),
        (
            re.compile(r"\b(policy|standard|regulation|statute|guideline|protocol|contract|agreement)\b", re.I),
            "PolicyOrStandard",
            UpperOntologyCategory.POLICY_OR_STANDARD,
            "Policy, regulation, guideline, or formal standard",
        ),
        (
            re.compile(r"\b(system|platform|application|device|instrument|repository|database)\b", re.I),
            "SystemAsset",
            UpperOntologyCategory.SYSTEM_OR_ASSET,
            "Technical system, application, device, or data repository",
        ),
        (
            re.compile(r"\b(region|country|facility|jurisdiction|site|campus|location)\b", re.I),
            "Location",
            UpperOntologyCategory.LOCATION_OR_REGION,
            "Geographical region, jurisdiction, or physical facility",
        ),
        (
            re.compile(r"\b(event|incident|audit|review|release|trial|milestone)\b", re.I),
            "Event",
            UpperOntologyCategory.PROCESS_OR_EVENT,
            "Discrete event, incident, audit, or milestone",
        ),
    ],
    text_relation_triggers=[
        (
            ("governed by", "regulated by", "complies with", "subject to"),
            "GOVERNED_BY",
            "SystemAsset",
            "PolicyOrStandard",
            "Governance or regulatory compliance relationship",
            False,
            False,
        ),
        (
            ("depends on", "requires", "uses", "integrates with"),
            "DEPENDS_ON",
            "SystemAsset",
            "SystemAsset",
            "Dependency relationship between assets or concepts",
            False,
            True,
        ),
        (
            ("developed by", "operated by", "owned by", "managed by", "maintained by"),
            "MANAGED_BY",
            "SystemAsset",
            "Organization",
            "Stewardship or operational ownership",
            False,
            False,
        ),
        (
            ("located in", "hosted in", "operates in", "deployed in"),
            "LOCATED_IN",
            "SystemAsset",
            "Location",
            "Geographical or jurisdictional location",
            False,
            False,
        ),
    ],
)


BUILTIN_DOMAIN_PACKS: dict[str, DomainRulePack] = {
    "cloud_sre": CLOUD_SRE_DOMAIN_PACK,
    "generic": GENERIC_DOMAIN_PACK,
}


def resolve_domain_pack(pack: DomainRulePack | str | None = None) -> DomainRulePack:
    if isinstance(pack, DomainRulePack):
        return pack
    if isinstance(pack, str):
        key = pack.strip().lower()
        if key in BUILTIN_DOMAIN_PACKS:
            return BUILTIN_DOMAIN_PACKS[key]
        raise ValueError(
            f"Unknown domain pack '{pack}'. Available built-in packs: {sorted(BUILTIN_DOMAIN_PACKS.keys())}"
        )
    return CLOUD_SRE_DOMAIN_PACK
