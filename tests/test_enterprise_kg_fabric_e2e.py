from __future__ import annotations

from enterprise_kg_fabric import (
    EnterpriseKGOntologyService,
    FederatedEmbeddingModelRouter,
    GraphRAGQueryRequest,
    PIIRedactionMode,
    PIIRedactor,
    PresidioPIIRedactor,
    RawDocumentInput,
    SemanticFeatureHashEmbeddingProvider,
    SiblingCollisionGuard,
    SubLinearVectorCentroidSampler,
    VectorChunkRecord,
    VectorKBRegistration,
    create_kg_fabric_app,
    get_production_pgvector_ddl,
)
from enterprise_kg_fabric.api import (
    BatchKBRegistrationRequest,
    BatchRawDocumentIngestRequest,
    BuildGraphRequest,
    HydrateKBChunksRequest,
    InduceOntologyRequest,
    PromoteOntologyRequest,
)
from examples.simulate_120_kbs_end_to_end import (
    run_120_kb_enterprise_simulation,
    run_coroutine_sync,
)


def test_120_kbs_full_enterprise_simulation_e2e() -> None:
    """Runs the full 120-KB + Raw Documents enterprise simulation end-to-end and verifies all assertions."""
    summary = run_coroutine_sync(run_120_kb_enterprise_simulation())

    assert summary["total_registered_kbs"] == 121  # 120 existing pgvector KBs + 1 managed raw doc KB
    assert summary["heterogeneous_embedding_dims"] == [384, 768, 1024, 1536]
    assert summary["sampling_compression_ratio"] >= 0.60
    assert summary["synonym_alignment_verified"] is True
    assert summary["cross_kb_entity_resolution_verified"] is True
    assert summary["sibling_collision_guard_verified"] is True
    assert summary["bitemporal_active_state_verified"] is True
    assert summary["bitemporal_historical_point_in_time_verified"] is True
    assert summary["multihop_2hop_compliance_discovery_verified"] is True
    assert summary["incremental_watermark_skip_ratio"] >= 0.99
    assert summary["bottom_up_emergent_ontology_promotion_verified"] is True
    assert summary["shacl_conformance_passed"] is True
    assert summary["w3c_owl_shacl_turtle_bytes"] > 1500


def test_sibling_collision_guard_blocks_numeric_region_and_env_merges() -> None:
    # Surface variations of the SAME entity MUST be allowed to merge
    ok_1, _ = SiblingCollisionGuard.can_merge("payment-router-01", "Payment Router 01")
    ok_2, _ = SiblingCollisionGuard.can_merge("aurora-ledger-01", "Aurora Ledger 1")
    assert ok_1 is True
    assert ok_2 is True

    # Sibling numbered entities MUST NEVER merge
    can_num, reason_num = SiblingCollisionGuard.can_merge("payment-router-01", "payment-router-02")
    assert can_num is False
    assert "Numeric" in reason_num

    # Versioned standards MUST NEVER merge
    can_ver, _ = SiblingCollisionGuard.can_merge("PCI-DSS-4.0", "PCI-DSS-3.2")
    assert can_ver is False

    # Regional siblings MUST NEVER merge
    can_reg, reason_reg = SiblingCollisionGuard.can_merge("gateway-us-east-1", "gateway-eu-central-1")
    assert can_reg is False
    assert "Region" in reason_reg

    # Environment siblings MUST NEVER merge
    can_env, reason_env = SiblingCollisionGuard.can_merge("ledger-prod", "ledger-staging")
    assert can_env is False
    assert "Environment" in reason_env


def test_sublinear_vector_centroid_sampler_compression_and_outlier_capture() -> None:
    embedder = SemanticFeatureHashEmbeddingProvider(default_dim=384)
    sampler = SubLinearVectorCentroidSampler(centroids_per_kb=3, boundary_outliers_per_kb=1)

    chunks = [
        VectorChunkRecord(
            chunk_id=f"kb_test::c_{i}",
            kb_id="kb_test",
            doc_id="doc_1",
            text=f"Microservice payment-svc-{i % 3} depends on aurora-db-{i % 2} in us-east-1.",
            native_embedding=embedder.embed_text(
                f"Microservice payment-svc-{i % 3} depends on aurora-db-{i % 2} in us-east-1.",
                dim=384,
            ),
        )
        for i in range(25)
    ]
    # Add a distinct outlier chunk about a Zero-Day quantum CVE
    outlier_text = "Zero-day advisory CVE-2026-8888 exposes PostQuantumVault mitigated by LatticeShield."
    chunks.append(
        VectorChunkRecord(
            chunk_id="kb_test::c_outlier",
            kb_id="kb_test",
            doc_id="doc_outlier",
            text=outlier_text,
            native_embedding=embedder.embed_text(outlier_text, dim=384),
        )
    )

    sampled = sampler.sample_representative_chunks_for_kb(chunks, embedder)
    assert len(sampled) == 4  # 3 medoids + 1 boundary outlier
    sampled_ids = {c.chunk_id for c in sampled}
    assert "kb_test::c_outlier" in sampled_ids


def test_presidio_pii_redactor_preserves_kg_identifiers_and_supports_hmac_mode() -> None:
    synth_jwt = ".".join(["eyJ" + "0ZXN0SGVhZGVy", "cGF5bG9hZE1vY2s", "c2lnbmF0dXJl"])
    synth_aws = "AKIA" + "0000000000000000"
    raw = (
        "Operator alice.smith@globalbank.com (phone 415-555-0199, SSN 123-45-6789) "
        f"connected using Bearer {synth_jwt} "
        f"and AWS key {synth_aws} regarding CVE-2026-9981 on PCI-DSS-4.0 "
        "and CC-209-Global in us-east-1."
    )
    # 1. Default REPLACE_TAG mode
    res_tag = PIIRedactor.analyze_and_redact(raw)
    assert "alice.smith@globalbank.com" not in res_tag.scrubbed_text
    assert "415-555-0199" not in res_tag.scrubbed_text
    assert "123-45-6789" not in res_tag.scrubbed_text
    assert synth_jwt not in res_tag.scrubbed_text
    assert synth_aws not in res_tag.scrubbed_text
    assert "<EMAIL_ADDRESS>" in res_tag.scrubbed_text
    assert "<US_SSN>" in res_tag.scrubbed_text
    assert "<BEARER_JWT_TOKEN>" in res_tag.scrubbed_text
    assert "<AWS_ACCESS_KEY_ID>" in res_tag.scrubbed_text
    # Critical: technical KG identifiers MUST be preserved intact!
    assert "CVE-2026-9981" in res_tag.scrubbed_text
    assert "PCI-DSS-4.0" in res_tag.scrubbed_text
    assert "CC-209-Global" in res_tag.scrubbed_text
    assert "us-east-1" in res_tag.scrubbed_text
    assert res_tag.protected_technical_tokens_preserved >= 4
    assert res_tag.total_redactions >= 5

    # 2. Deterministic HMAC_PSEUDONYMIZE mode (for safe cross-KB entity linkage)
    hmac_redactor = PresidioPIIRedactor(
        mode=PIIRedactionMode.HMAC_PSEUDONYMIZE,
        hmac_secret_key="test-kms-key",
    )
    r1 = hmac_redactor.analyze_and_redact("Ticket opened by alice.smith@globalbank.com for CVE-2026-9981.")
    r2 = hmac_redactor.analyze_and_redact("Follow-up from ALICE.SMITH@GLOBALBANK.COM on PCI-DSS-4.0.")
    assert "alice.smith@globalbank.com" not in r1.scrubbed_text.lower()
    assert "CVE-2026-9981" in r1.scrubbed_text
    # Same email produces identical deterministic HMAC token across KBs
    tok1 = [w for w in r1.scrubbed_text.split() if w.startswith("<EMAIL_ADDRESS_")][0]
    tok2 = [w for w in r2.scrubbed_text.split() if w.startswith("<EMAIL_ADDRESS_")][0]
    assert tok1 == tok2

    ddl = get_production_pgvector_ddl(canonical_dim=256)
    assert "CREATE EXTENSION IF NOT EXISTS vector;" in ddl
    assert "USING hnsw (canonical_embedding vector_cosine_ops)" in ddl
    assert "ENABLE ROW LEVEL SECURITY" in ddl


def test_federated_multi_model_query_fanout_and_rrf() -> None:
    """
    Verifies that KBs using completely different embedding models (384d BGE vs 1536d OpenAI)
    are searched natively per model group and merged via scale-invariant rank-based RRF.
    """
    async def _run_multi_model() -> None:
        svc = EnterpriseKGOntologyService(
            min_kb_support_for_approval=1,
            min_chunk_support_for_approval=1,
        )
        await svc.register_vector_kbs_batch(
            [
                VectorKBRegistration(
                    kb_id="kb_bge_384",
                    tenant_id="t_mm",
                    domain_tag="sre",
                    table_name="vec_bge",
                    embedding_dim=384,
                    embedding_model_id="bge-small-en-v1.5",
                ),
                VectorKBRegistration(
                    kb_id="kb_openai_1536",
                    tenant_id="t_mm",
                    domain_tag="compliance",
                    table_name="vec_openai",
                    embedding_dim=1536,
                    embedding_model_id="text-embedding-3-large",
                ),
            ]
        )
        await svc.hydrate_existing_kb_chunks(
            "kb_bge_384",
            [
                VectorChunkRecord(
                    chunk_id="c_bge_1",
                    kb_id="kb_bge_384",
                    tenant_id="t_mm",
                    domain_tag="sre",
                    doc_id="d_bge",
                    text="Microservice billing-api-01 depends on aurora-billing-01 in us-east-1.",
                )
            ],
        )
        await svc.hydrate_existing_kb_chunks(
            "kb_openai_1536",
            [
                VectorChunkRecord(
                    chunk_id="c_oai_1",
                    kb_id="kb_openai_1536",
                    tenant_id="t_mm",
                    domain_tag="compliance",
                    doc_id="d_oai",
                    text="Database cluster aurora-billing-01 is governed by PCI-DSS-4.0.",
                )
            ],
        )
        await svc.induce_and_align_ontology(tenant_id="t_mm")
        await svc.build_or_update_knowledge_graph(tenant_id="t_mm")

        rag = await svc.query_graph_rag(
            GraphRAGQueryRequest(
                tenant_id="t_mm",
                query="Which compliance standard governs aurora-billing-01 used by billing-api-01?",
                max_graph_hops=2,
            )
        )
        channels_seen = {ch for ctx in rag.chunk_contexts for ch in ctx.retrieval_channels}
        assert "pgvector_native_bge-small-en-v1.5_384d" in channels_seen
        assert "pgvector_native_text-embedding-3-large_1536d" in channels_seen
        assert "aurora-billing-01" in rag.cross_kb_bridge_entities
        assert "PCI-DSS-4.0" in rag.synthesized_answer

    run_coroutine_sync(_run_multi_model())


def test_fastapi_rest_endpoints_e2e() -> None:
    async def _run_api_flow() -> None:
        service = EnterpriseKGOntologyService(
            min_kb_support_for_approval=1,
            min_chunk_support_for_approval=1,
        )
        app = create_kg_fabric_app(service=service)
        routes = {r.path: r.endpoint for r in app.routes if hasattr(r, "path")}

        # 1. Health check
        health_res = await routes["/health"]()
        assert health_res["status"] == "healthy"

        # 2. Register 2 heterogeneous vector KBs
        reg_res = await routes["/v1/kbs/register-batch"](
            BatchKBRegistrationRequest(
                registrations=[
                    VectorKBRegistration(
                        kb_id="kb_api_sre",
                        tenant_id="t_api",
                        domain_tag="sre",
                        table_name="vec_sre",
                        embedding_dim=384,
                    ),
                    VectorKBRegistration(
                        kb_id="kb_api_compliance",
                        tenant_id="t_api",
                        domain_tag="compliance",
                        table_name="vec_comp",
                        embedding_dim=1536,
                    ),
                ]
            )
        )
        assert reg_res["registered_count"] == 2

        # 3. Hydrate existing pgvector KB chunks
        hyd_res = await routes["/v1/kbs/{kb_id}/hydrate"](
            "kb_api_sre",
            HydrateKBChunksRequest(
                chunks=[
                    VectorChunkRecord(
                        chunk_id="c_sre_1",
                        kb_id="kb_api_sre",
                        tenant_id="t_api",
                        domain_tag="sre",
                        doc_id="d1",
                        text="Microservice checkout-api-01 depends on aurora-orders-01 and is deployed in primary us-east-1.",
                    )
                ]
            ),
        )
        assert hyd_res["hydrated_chunks"] == 1

        # 4. Ingest raw document into compliance KB
        doc_res = await routes["/v1/documents/ingest"](
            BatchRawDocumentIngestRequest(
                documents=[
                    RawDocumentInput(
                        doc_id="doc_comp_1",
                        kb_id="kb_api_compliance",
                        tenant_id="t_api",
                        domain_tag="compliance",
                        title="Orders Compliance Policy",
                        content="# Compliance Scope\nDatabase cluster aurora-orders-01 is governed by PCI-DSS-4.0.",
                    )
                ],
                native_kb_dim=1536,
            )
        )
        assert doc_res["chunks_created"] >= 1

        # 5. Induce ontology
        ind_res = await routes["/v1/ontology/induce"](
            InduceOntologyRequest(tenant_id="t_api")
        )
        assert ind_res.kbs_scanned == 2

        # 6. Build knowledge graph
        build_res = await routes["/v1/graph/build"](
            BuildGraphRequest(tenant_id="t_api", incremental=True, validate_shacl=True)
        )
        assert build_res.shacl_conformance_passed is True
        assert build_res.cross_kb_merged_entities >= 1

        # 7. Export W3C OWL/SHACL Turtle
        ttl_res = await routes["/v1/ontology/export.ttl"](tenant_id="t_api")
        assert "owl:Ontology" in ttl_res
        assert "sh:NodeShape" in ttl_res

        # 8. Query Hybrid GraphRAG across both KBs
        rag_res = await routes["/v1/rag/query"](
            GraphRAGQueryRequest(
                tenant_id="t_api",
                query="Which compliance standard governs the database that checkout-api-01 depends on?",
                max_graph_hops=2,
            )
        )
        assert "aurora-orders-01" in rag_res.cross_kb_bridge_entities
        assert "PCI-DSS-4.0" in rag_res.synthesized_answer
        assert set(rag_res.provenance_kbs_contributing) == {"kb_api_sre", "kb_api_compliance"}

        # 9. Promote ontology & check telemetry
        await routes["/v1/ontology/promote"](PromoteOntologyRequest(tenant_id="t_api"))
        tel_res = await routes["/v1/telemetry"](tenant_id="t_api")
        assert tel_res["registered_kbs_count"] == 2

    run_coroutine_sync(_run_api_flow())
