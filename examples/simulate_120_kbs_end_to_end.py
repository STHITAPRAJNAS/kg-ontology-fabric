from __future__ import annotations

import asyncio
import json
from pathlib import Path

from enterprise_kg_fabric import (
    EnterpriseKGOntologyService,
    GraphRAGQueryRequest,
    KBSourceType,
    OntologyStatus,
    RawDocumentInput,
    UpperOntologyCategory,
    VectorChunkRecord,
    VectorKBRegistration,
)


async def run_120_kb_enterprise_simulation() -> dict:
    print("=" * 92)
    print(" ENTERPRISE MULTI-KB AUTOMATED ONTOLOGY & KNOWLEDGE GRAPH FABRIC — 120-KB E2E SIMULATION")
    print("=" * 92)

    service = EnterpriseKGOntologyService(
        canonical_embedding_dim=256,
        centroids_per_kb=2,
        boundary_outliers_per_kb=1,
        min_kb_support_for_approval=2,
        min_chunk_support_for_approval=2,
    )

    tenant_id = "t_global_bank"

    # -------------------------------------------------------------------------
    # STEP 1: Register & Hydrate 120 Heterogeneous pgvector KBs across 6 Divisions
    # -------------------------------------------------------------------------
    divisions = [
        ("sre_cloud", 384, "bge-small-en-v1.5", "Microservice", "CALLS_UPSTREAM", UpperOntologyCategory.SYSTEM_OR_ASSET),
        ("finops_economics", 768, "text-embedding-004", "ServiceComponent", "ALLOCATED_TO_COST_CENTER", UpperOntologyCategory.METRIC_OR_FINANCIAL),
        ("regulatory_compliance", 1536, "text-embedding-3-large", "RegulatoryPolicy", "COMPLIES_WITH", UpperOntologyCategory.POLICY_OR_STANDARD),
        ("secops_threat_intel", 768, "text-embedding-004", "SecurityVulnerability", "REMEDIATED_BY_WAF", UpperOntologyCategory.PROCESS_OR_EVENT),
        ("enterprise_customer_sla", 1024, "embed-english-v3.0", "EnterpriseCustomer", "SUBSCRIBES_TO_SLA", UpperOntologyCategory.ACTOR_OR_ORG),
        ("data_platform_ml", 384, "bge-small-en-v1.5", "DataArtifact", "DEPENDS_ON", UpperOntologyCategory.DATA_OR_ARTIFACT),
    ]

    registrations: list[VectorKBRegistration] = []
    total_hydrated_chunks = 0

    for kb_idx in range(1, 121):
        div_idx = (kb_idx - 1) // 20
        domain_tag, emb_dim, model_id, local_cls_term, local_rel_term, upper_cat = divisions[div_idx]
        kb_id = f"kb_{kb_idx:03d}_{domain_tag}"

        reg = VectorKBRegistration(
            kb_id=kb_id,
            tenant_id=tenant_id,
            domain_tag=domain_tag,
            source_type=KBSourceType.EXISTING_PGVECTOR_KB,
            postgres_dsn_alias=f"pgvector_shard_{(kb_idx % 4) + 1}",
            schema_name=f"schema_{domain_tag}",
            table_name=f"vec_store_{kb_id}",
            embedding_dim=emb_dim,
            embedding_model_id=model_id,
            owner_team=f"team-{domain_tag}",
        )
        registrations.append(reg)

    await service.register_vector_kbs_batch(registrations)

    # Populate each of the 120 KBs with 8 chunks (960 existing pgvector rows total)
    for kb_idx, reg in enumerate(registrations, start=1):
        div_idx = (kb_idx - 1) // 20
        domain_tag, emb_dim, _, local_cls_term, local_rel_term, upper_cat = divisions[div_idx]
        kb_chunks: list[VectorChunkRecord] = []

        for c_num in range(1, 9):
            chunk_id = f"{reg.kb_id}::vec_chk_{c_num:02d}"
            meta: dict = {
                "local_class_term": local_cls_term,
                "local_relation_term": local_rel_term,
                "upper_category": upper_cat.value,
            }
            t_valid = "2026-03-01T00:00:00Z"

            # Inject rich cross-KB overlapping topology + sibling entities + bi-temporal mutations
            synth_jwt = ".".join(["eyJ" + "0ZXN0SGVhZGVy", "cGF5bG9hZE1vY2s", "c2lnbmF0dXJl"])
            if kb_idx == 1 and c_num == 1:
                # Historical SRE deployment in us-east-1 + upstream/downstream dependencies
                text = (
                    "SRE Architecture Topology: Microservice api-edge-gw depends on payment-router-01. "
                    "payment-router-01 calls upstream aurora-ledger-01 and is maintained by sre-payments-squad. "
                    "payment-router-01 is deployed in primary us-east-1. "
                    f"Auth Header: Bearer {synth_jwt}"
                )
                t_valid = "2026-01-15T00:00:00Z"
            elif kb_idx == 5 and c_num == 1:
                # Sibling service payment-router-02 & aurora-ledger-02 (must NEVER merge with -01!)
                text = (
                    "Canary SRE Topology: Microservice payment-router-02 depends on aurora-ledger-02. "
                    "payment-router-02 is deployed in primary eu-west-1 and is maintained by sre-canary-squad."
                )
                t_valid = "2026-02-10T00:00:00Z"
            elif kb_idx == 14 and c_num == 1:
                # Later SRE failover event: payment-router-01 fails over from us-east-1 to eu-central-1
                text = (
                    "DR Failover Execution: payment-router-01 failed over to eu-central-1 after regional link saturation."
                )
                t_valid = "2026-10-20T00:00:00Z"
            elif kb_idx == 22 and c_num == 1:
                # FinOps KB #22: Surface variation "Payment Router 01" + initial CostCenter CC-104-Core
                text = (
                    "FinOps Ledger Q1: ServiceComponent Payment Router 01 is allocated to cost center CC-104-Core. "
                    "[Triple: Payment Router 01 (ServiceComponent) --ALLOCATED_TO_COST_CENTER--> CC-104-Core (BudgetUnit)]"
                )
                t_valid = "2026-02-01T00:00:00Z"
            elif kb_idx == 35 and c_num == 1:
                # FinOps KB #35: Bi-temporal cost center reallocation to CC-209-Global
                text = (
                    "FinOps Reorg Q4: payment-router-01 is owned by primary CC-209-Global."
                )
                t_valid = "2026-11-01T00:00:00Z"
            elif kb_idx == 45 and c_num == 1:
                # Compliance KB #45: Surface variation "Aurora Ledger 01" governed by PCI-DSS-4.0 & GDPR-Article-32
                text = (
                    "Compliance Audit Scope: Database cluster aurora-ledger-01 is governed by PCI-DSS-4.0 "
                    "and aurora-ledger-01 complies with GDPR-Article-32. "
                    "Legacy cluster aurora-ledger-02 complies with PCI-DSS-3.2."
                )
                t_valid = "2026-05-01T00:00:00Z"
            elif kb_idx == 64 and c_num == 1:
                # SecOps KB #64: CVE-2026-9981 affecting payment-router-01, mitigated by WAF-Shield-Rule-904
                text = (
                    "SecOps Advisory: Security vulnerability CVE-2026-9981 affects payment-router-01. "
                    "CVE-2026-9981 is mitigated by WAF-Shield-Rule-904. "
                    "Sibling advisory CVE-2026-9982 affects payment-router-02."
                )
                t_valid = "2026-09-12T00:00:00Z"
            elif kb_idx == 79 and c_num == 8:
                # Single-KB Emergent Zero-Day Ontology Class & Predicate in KB #79 (for Quarantine -> Promotion test)
                text = (
                    "Experimental Post-Quantum Security: "
                    "[OntologyClass: PostQuantumCipherVault | Upper: p:SystemOrAsset] "
                    "[OntologyRelation: PROTECTED_BY_LATTICE_KEM | Domain: DatabaseCluster | Range: PostQuantumCipherVault] "
                    "[Triple: aurora-ledger-01 (DatabaseCluster) --PROTECTED_BY_LATTICE_KEM--> pq-kyber-vault-01 (PostQuantumCipherVault)]"
                )
                t_valid = "2026-11-05T00:00:00Z"
            elif kb_idx == 85 and c_num == 1:
                # Customer SLA KB #85: Enterprise customer Acme-Global-Bank connected to payment-router-01
                text = (
                    "Enterprise Account Dossier: Enterprise customer Acme-Global-Bank uses platform service payment-router-01 "
                    "and Acme-Global-Bank subscribes to sla SLA-Tier-0-MissionCritical."
                )
                t_valid = "2026-04-01T00:00:00Z"
            elif kb_idx == 105 and c_num == 1:
                # Data Platform KB #105: ML Feature store depending on aurora-ledger-01
                text = (
                    "ML Feature Lineage: DataArtifact fraud-feature-store-01 depends on aurora-ledger-01 "
                    "and fraud-feature-store-01 is governed by GDPR-Article-32."
                )
                t_valid = "2026-06-15T00:00:00Z"
            else:
                # Standard domain background chunks across the 120 KBs
                svc_num = ((kb_idx - 1) % 15) + 10
                db_num = ((kb_idx - 1) % 8) + 10
                text = (
                    f"[{domain_tag.upper()} KB-{kb_idx:03d} Chunk-{c_num}] "
                    f"Microservice regional-svc-{svc_num} depends on regional-db-{db_num} "
                    f"and regional-svc-{svc_num} is governed by SOC2-Type-II."
                )

            meta["t_valid"] = t_valid
            native_vec = service.embedding_provider.embed_text(text, dim=emb_dim)
            kb_chunks.append(
                VectorChunkRecord(
                    chunk_id=chunk_id,
                    kb_id=reg.kb_id,
                    tenant_id=tenant_id,
                    domain_tag=domain_tag,
                    doc_id=f"doc_{reg.kb_id}_{c_num}",
                    text=text,
                    native_embedding=native_vec,
                    native_embedding_dim=emb_dim,
                    metadata=meta,
                    updated_at=t_valid,
                )
            )

        total_hydrated_chunks += await service.hydrate_existing_kb_chunks(reg.kb_id, kb_chunks)

    print(f"[1] Registered & Hydrated {len(registrations)} Heterogeneous pgvector KBs ({total_hydrated_chunks} vector rows)")

    # -------------------------------------------------------------------------
    # STEP 2: Ingest Raw Unstructured Markdown Documents via LlamaIndex Parser
    # -------------------------------------------------------------------------
    raw_docs = [
        RawDocumentInput(
            doc_id="raw_arch_runbook_2026",
            kb_id="kb_121_raw_runbooks",
            tenant_id=tenant_id,
            domain_tag="sre_cloud",
            title="Global Payments Architecture & Compliance Runbook",
            source_uri="s3://enterprise-docs/runbooks/global_payments_2026.md",
            content=f"""# Global Payments Core Topology
The mission-critical service payment-router-01 depends on aurora-ledger-01 for transactional settlement.
 Operator tokens such as Bearer {synth_jwt} and SSN 123-45-6789 must be scrubbed.

## Compliance & Audit Boundary
The settlement cluster aurora-ledger-01 is governed by SOC2-Type-II and PCI-DSS-4.0.
""",
        ),
    ]
    raw_doc_chunks = await service.ingest_raw_documents(raw_docs, native_kb_dim=768)
    print(f"[2] Ingested {len(raw_docs)} Raw Markdown Document(s) into {len(raw_doc_chunks)} LlamaIndex Hierarchical Chunks")

    # -------------------------------------------------------------------------
    # STEP 3: Automated Sub-Linear Vector-Cluster Ontology Induction & Alignment
    # -------------------------------------------------------------------------
    tbox, induction_report = await service.induce_and_align_ontology(
        tenant_id=tenant_id,
        tbox_version="1.0.0",
    )
    print(
        f"[3] Induced & Aligned Enterprise Ontology v{tbox.version} across {induction_report.kbs_scanned} KBs:\n"
        f"    - Total Chunks: {induction_report.total_chunks_across_kbs} -> Sampled Centroids: {induction_report.sampled_centroid_chunks} "
        f"(Compression: {induction_report.sampling_compression_ratio * 100:.1f}%)\n"
        f"    - Approved Classes: {induction_report.aligned_approved_classes} | Quarantined Candidate Classes: {induction_report.quarantined_candidate_classes}\n"
        f"    - Approved Relations: {induction_report.aligned_approved_relations} | Quarantined Candidate Relations: {induction_report.quarantined_candidate_relations}\n"
        f"    - Aligned Synonyms Merged: {json.dumps(induction_report.merged_synonym_mappings)}"
    )

    # -------------------------------------------------------------------------
    # STEP 4: Build Knowledge Graph + Cross-KB Entity Resolution + SHACL Validation
    # -------------------------------------------------------------------------
    build_report_1 = await service.build_or_update_knowledge_graph(
        tenant_id=tenant_id,
        incremental=True,
        validate_shacl=True,
    )
    print(
        f"[4] Built Knowledge Graph (Pass 1 - Full Cold Start):\n"
        f"    - Delta Chunks Extracted: {build_report_1.delta_chunks_extracted}/{build_report_1.total_chunks_inspected}\n"
        f"    - Canonical Entities after ER: {build_report_1.canonical_entities_after_er} "
        f"(Cross-KB Merged Entities: {build_report_1.cross_kb_merged_entities})\n"
        f"    - Sibling Collisions Prevented: {build_report_1.sibling_collisions_prevented}\n"
        f"    - Active Bi-Temporal Edges: {build_report_1.active_edges_upserted} "
        f"(Superseded Functional Edges Closed: {build_report_1.superseded_bitemporal_edges_closed})\n"
        f"    - W3C OWL/SHACL Validation Conforms: {build_report_1.shacl_conformance_passed}"
    )

    # -------------------------------------------------------------------------
    # STEP 5: Test Incremental SHA-256 Watermark Delta Sync (Add 1 new doc)
    # -------------------------------------------------------------------------
    delta_doc = RawDocumentInput(
        doc_id="raw_secops_bulletin_nov",
        kb_id="kb_121_raw_runbooks",
        tenant_id=tenant_id,
        domain_tag="secops_threat_intel",
        title="November SecOps Patch Verification",
        content="# Patch Verification\nSecurity vulnerability CVE-2026-9981 is mitigated by WAF-Shield-Rule-904.",
    )
    await service.ingest_raw_documents([delta_doc], native_kb_dim=768)
    build_report_2 = await service.build_or_update_knowledge_graph(
        tenant_id=tenant_id,
        incremental=True,
        validate_shacl=True,
    )
    print(
        f"[5] Incremental SHA-256 Watermark Sync (Pass 2 - After 1 New Doc):\n"
        f"    - Chunks Inspected: {build_report_2.total_chunks_inspected} | "
        f"Skipped via Watermark: {build_report_2.skipped_unchanged_chunks_via_watermark} | "
        f"Delta Extracted: {build_report_2.delta_chunks_extracted}"
    )

    # -------------------------------------------------------------------------
    # STEP 6: LlamaIndex 4-Path Hybrid GraphRAG Query (Multi-Hop Cross-KB Reasoning)
    # -------------------------------------------------------------------------
    rag_response = await service.query_graph_rag(
        GraphRAGQueryRequest(
            tenant_id=tenant_id,
            query=(
                "What downstream database, active cloud region, cost center, compliance standards, "
                "CVE vulnerabilities, and enterprise customers are connected to payment-router-01?"
            ),
            top_k_chunks=6,
            max_graph_hops=2,
        )
    )
    print(
        f"[6] LlamaIndex 4-Path Hybrid GraphRAG Query Executed in {rag_response.retrieval_latency_ms} ms:\n"
        f"    - Seed Entities Matched: {rag_response.seed_entities_matched}\n"
        f"    - Cross-KB Bridge Entities Discovered: {rag_response.cross_kb_bridge_entities}\n"
        f"    - Contributing KBs across Graph & Vector Paths: {rag_response.provenance_kbs_contributing}\n"
        f"    - Graph Facts Retrieved: {len(rag_response.graph_facts)} (2-Hop Neighborhood)"
    )

    # Historical Point-in-Time Bi-Temporal Query (`as_of_iso="2026-06-01T00:00:00Z"`)
    historical_rag = await service.query_graph_rag(
        GraphRAGQueryRequest(
            tenant_id=tenant_id,
            query="Where was payment-router-01 deployed and which cost center owned it in June 2026?",
            max_graph_hops=1,
            as_of_iso="2026-06-01T00:00:00Z",
        )
    )

    # -------------------------------------------------------------------------
    # STEP 7: Promote Quarantined Emergent Ontology Concepts & Verify
    # -------------------------------------------------------------------------
    promotion_result = await service.promote_quarantined_ontology_concepts(
        tenant_id=tenant_id,
        min_confidence=0.80,
    )
    print(
        f"[7] Bottom-Up Ontology Evolution (Quarantine -> Approved):\n"
        f"    - New TBox Version: {promotion_result['new_tbox_version']}\n"
        f"    - Promoted Classes: {promotion_result['promoted_classes']}\n"
        f"    - Promoted Relations: {promotion_result['promoted_relations']}"
    )

    turtle_str = service.export_ontology_turtle(tenant_id)
    telemetry = await service.get_enterprise_telemetry(tenant_id)

    active_targets = {f.target_name for f in rag_response.graph_facts}
    historical_targets = {f.target_name for f in historical_rag.graph_facts}

    summary = {
        "total_registered_kbs": telemetry["registered_kbs_count"],
        "heterogeneous_embedding_dims": telemetry["heterogeneous_embedding_dims_supported"],
        "sampling_compression_ratio": induction_report.sampling_compression_ratio,
        "synonym_alignment_verified": (
            tbox.resolve_class("Microservice") == "CloudService"
            and tbox.resolve_class("ServiceComponent") == "CloudService"
            and tbox.resolve_predicate("CALLS_UPSTREAM") == "DEPENDS_ON"
            and tbox.resolve_predicate("ALLOCATED_TO_COST_CENTER") == "OWNED_BY_PRIMARY"
        ),
        "cross_kb_entity_resolution_verified": (
            "payment-router-01" in rag_response.cross_kb_bridge_entities
            or "Payment Router 01" in rag_response.cross_kb_bridge_entities
        ),
        "sibling_collision_guard_verified": build_report_1.sibling_collisions_prevented > 0,
        "bitemporal_active_state_verified": (
            "eu-central-1" in active_targets
            and "CC-209-Global" in active_targets
            and "us-east-1" not in active_targets
            and "CC-104-Core" not in active_targets
        ),
        "bitemporal_historical_point_in_time_verified": (
            "us-east-1" in historical_targets
            and "CC-104-Core" in historical_targets
            and "eu-central-1" not in historical_targets
        ),
        "multihop_2hop_compliance_discovery_verified": (
            "PCI-DSS-4.0" in active_targets and "GDPR-Article-32" in active_targets
        ),
        "incremental_watermark_skip_ratio": round(
            build_report_2.skipped_unchanged_chunks_via_watermark
            / max(1, build_report_2.total_chunks_inspected),
            4,
        ),
        "bottom_up_emergent_ontology_promotion_verified": (
            "PostQuantumCipherVault" in promotion_result["promoted_classes"]
            and "PROTECTED_BY_LATTICE_KEM" in promotion_result["promoted_relations"]
        ),
        "w3c_owl_shacl_turtle_bytes": len(turtle_str),
        "shacl_conformance_passed": build_report_1.shacl_conformance_passed,
    }

    report_path = Path(__file__).resolve().parent.parent / "simulation_benchmark_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 92)
    print(" VERIFIED SIMULATION SUMMARY:")
    print(json.dumps(summary, indent=2))
    print("=" * 92)
    return summary


def run_coroutine_sync(coro):
    """Drives pure-Python async coroutines without requiring Windows loopback socketpair."""
    try:
        while True:
            coro.send(None)
    except StopIteration as exc:
        return exc.value


if __name__ == "__main__":
    run_coroutine_sync(run_120_kb_enterprise_simulation())
