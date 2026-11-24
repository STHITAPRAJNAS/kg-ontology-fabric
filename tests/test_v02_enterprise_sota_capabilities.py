from __future__ import annotations

from enterprise_kg_fabric import (
    AdaptiveQueryRouter,
    EnterpriseKGOntologyService,
    EvalTestCase,
    GlobalCommunitySearchRequest,
    GraphRAGQueryRequest,
    MultimodalDocumentIngestor,
    PPRSearchRequest,
    QueryRouteDecision,
    RawDocumentInput,
    SyncJobStatus,
    VectorChunkRecord,
    VectorKBRegistration,
    create_kg_fabric_app,
)
from enterprise_kg_fabric.api import BuildGraphRequest, RunEvalSuiteRequest
from examples.simulate_120_kbs_end_to_end import run_coroutine_sync


def test_adaptive_query_router_classification() -> None:
    assert (
        AdaptiveQueryRouter.classify_query("Summarize the top 5 systemic risks across all KBs")
        == QueryRouteDecision.GLOBAL_SENSEMAKING
    )
    assert (
        AdaptiveQueryRouter.classify_query("How many services are governed by PCI-DSS-4.0?")
        == QueryRouteDecision.STRUCTURED_AGGREGATION
    )
    assert (
        AdaptiveQueryRouter.classify_query("What downstream database is connected to payment-router-01?")
        == QueryRouteDecision.LOCAL_MULTIHOP_PPR
    )
    assert (
        AdaptiveQueryRouter.classify_query("What is the office cafeteria taco schedule?")
        == QueryRouteDecision.FACTUAL_VECTOR
    )
    assert (
        AdaptiveQueryRouter.classify_query("Status check", as_of_iso="2026-06-01T00:00:00Z")
        == QueryRouteDecision.LOCAL_MULTIHOP_PPR
    )


def test_hipporag_ppr_hub_damping_and_leiden_global_communities() -> None:
    async def _run() -> None:
        svc = EnterpriseKGOntologyService(
            min_kb_support_for_approval=1,
            min_chunk_support_for_approval=1,
        )
        await svc.register_vector_kbs_batch(
            [
                VectorKBRegistration(
                    kb_id="kb_core", tenant_id="t_ppr", domain_tag="sre", table_name="t_core", embedding_dim=384
                ),
                VectorKBRegistration(
                    kb_id="kb_noise", tenant_id="t_ppr", domain_tag="compliance", table_name="t_noise", embedding_dim=768
                ),
            ]
        )
        # Specific target path: payment-router-01 -> aurora-ledger-01 -> PCI-DSS-4.0
        # Plus a supernode hub SOC2-Type-II connected to 15 unrelated noise services!
        chunks = [
            VectorChunkRecord(
                chunk_id="c_core_1",
                kb_id="kb_core",
                tenant_id="t_ppr",
                domain_tag="sre",
                doc_id="d1",
                text=(
                    "Microservice payment-router-01 depends on aurora-ledger-01. "
                    "payment-router-01 is governed by SOC2-Type-II."
                ),
            ),
            VectorChunkRecord(
                chunk_id="c_core_2",
                kb_id="kb_noise",
                tenant_id="t_ppr",
                domain_tag="compliance",
                doc_id="d2",
                text="Database cluster aurora-ledger-01 is governed by PCI-DSS-4.0.",
            ),
        ]
        for i in range(1, 16):
            chunks.append(
                VectorChunkRecord(
                    chunk_id=f"c_hub_noise_{i}",
                    kb_id="kb_noise",
                    tenant_id="t_ppr",
                    domain_tag="compliance",
                    doc_id=f"d_noise_{i}",
                    text=f"Microservice unrelated-svc-{i:02d} is governed by SOC2-Type-II.",
                )
            )
        await svc.hydrate_existing_kb_chunks("kb_core", chunks[:1])
        await svc.hydrate_existing_kb_chunks("kb_noise", chunks[1:])
        await svc.induce_and_align_ontology(tenant_id="t_ppr")
        await svc.build_or_update_knowledge_graph(tenant_id="t_ppr", validate_shacl=True)

        # 1. HippoRAG 2 PPR Search seeded from payment-router-01
        ppr_res = await svc.search_ppr_subgraph(
            PPRSearchRequest(tenant_id="t_ppr", seed_entities=["payment-router-01"], top_k_facts=5)
        )
        ppr_by_target = {f.target_name: f.ppr_score for f in ppr_res.ranked_facts}
        # High-specificity neighbor aurora-ledger-01 MUST outrank noisy supernode spokes!
        assert "aurora-ledger-01" in ppr_by_target
        top_edge = ppr_res.ranked_facts[0]
        assert top_edge.source_name == "payment-router-01"
        assert top_edge.target_name == "aurora-ledger-01"
        assert top_edge.ppr_score > 0.0

        # 2. Microsoft GraphRAG Leiden/Louvain Community Detection + Adaptive Global Sensemaking
        global_res = await svc.query_graph_rag(
            GraphRAGQueryRequest(
                tenant_id="t_ppr",
                query="Summarize the overall systemic compliance landscape across all KBs",
                use_adaptive_routing=True,
            )
        )
        assert global_res.route_taken == QueryRouteDecision.GLOBAL_SENSEMAKING
        assert len(global_res.community_summaries_used) >= 1
        assert "Global Sensemaking Synthesis" in global_res.synthesized_answer

    run_coroutine_sync(_run())


def test_abac_zero_leakage_nary_qualifiers_and_multimodal_tables() -> None:
    async def _run() -> None:
        svc = EnterpriseKGOntologyService(
            min_kb_support_for_approval=1,
            min_chunk_support_for_approval=1,
        )

        # 1. Public Markdown Architecture Table (auto-parsed into structured triples!)
        md_table_doc = RawDocumentInput(
            doc_id="arch_matrix_2026",
            kb_id="kb_public_sre",
            tenant_id="t_sec",
            domain_tag="sre",
            title="Production Architecture Table",
            acl_groups=["public"],
            content="""# Production Service Matrix
| Service | Database | Region | Cost Center |
| --- | --- | --- | --- |
| api-edge-gw | payment-router-01 | eu-central-1 | CC-209-Global |

Note: payment-router-01 failed over to eu-central-1 only if latency exceeds 200ms and approved by VP-SRE-Alice.
""",
        )

        # 2. Restricted Legal Confidential Doc (ABAC: ["legal_confidential"])
        confidential_doc = RawDocumentInput(
            doc_id="ma_legal_secret",
            kb_id="kb_legal_confidential",
            tenant_id="t_sec",
            domain_tag="legal",
            title="Confidential M&A Regulatory Disclosure",
            acl_groups=["legal_confidential"],
            content=(
                "# Confidential Scope\n"
                "Microservice payment-router-01 is governed by GDPR-Article-32 "
                "and payment-router-01 depends on aurora-secret-merger-db."
            ),
        )

        await svc.ingest_raw_documents([md_table_doc, confidential_doc])
        await svc.induce_and_align_ontology(tenant_id="t_sec")
        await svc.build_or_update_knowledge_graph(tenant_id="t_sec", validate_shacl=True)

        # 3. Verify CSV Table Parser directly
        csv_text = "Service,Database,Compliance\ncheckout-svc-01,aurora-orders-01,PCI-DSS-4.0\n"
        md_out, csv_triples = MultimodalDocumentIngestor.parse_csv_to_markdown_and_triples(csv_text)
        assert "| checkout-svc-01 |" in md_out
        assert len(csv_triples) == 2

        # 4. Verify N-ary Reified Qualifiers extracted on failover edge
        pub_resp = await svc.query_graph_rag(
            GraphRAGQueryRequest(
                tenant_id="t_sec",
                query="Where did payment-router-01 fail over and what depends on it?",
                caller_acl_groups=["public"],
                max_graph_hops=2,
            )
        )
        failover_edges = [f for f in pub_resp.graph_facts if f.predicate == "DEPLOYED_IN_PRIMARY"]
        assert len(failover_edges) >= 1
        assert "approved_by" in failover_edges[0].qualifiers
        assert "VP-SRE-Alice" in failover_edges[0].qualifiers["approved_by"]

        # 5. Verify ABAC ZERO LEAKAGE for `public` caller vs full visibility for `legal_confidential` caller
        pub_blob = pub_resp.synthesized_answer + " ".join(f.target_name for f in pub_resp.graph_facts)
        assert "aurora-secret-merger-db" not in pub_blob
        assert "GDPR-Article-32" not in pub_blob

        legal_resp = await svc.query_graph_rag(
            GraphRAGQueryRequest(
                tenant_id="t_sec",
                query="What compliance standard and secret database are connected to payment-router-01?",
                caller_acl_groups=["public", "legal_confidential"],
                max_graph_hops=2,
            )
        )
        legal_targets = {f.target_name for f in legal_resp.graph_facts}
        assert "aurora-secret-merger-db" in legal_targets
        assert "GDPR-Article-32" in legal_targets

        # 6. Verify Evaluation Harness measures 100% ABAC Zero-Leakage & Multi-Hop Recall
        eval_report = await svc.run_eval_suite(
            tenant_id="t_sec",
            cases=[
                EvalTestCase(
                    query="Where did payment-router-01 fail over and what is its cost center?",
                    expected_facts=["eu-central-1", "CC-209-Global"],
                    forbidden_facts=["aurora-secret-merger-db", "GDPR-Article-32"],
                    caller_acl_groups=["public"],
                ),
                EvalTestCase(
                    query="What secret database and GDPR standard are connected to payment-router-01?",
                    expected_facts=["aurora-secret-merger-db", "GDPR-Article-32"],
                    caller_acl_groups=["public", "legal_confidential"],
                ),
            ],
        )
        assert eval_report.abac_zero_leakage_rate == 1.0
        assert eval_report.mean_multihop_recall == 1.0
        assert eval_report.citation_groundedness_rate == 1.0

    run_coroutine_sync(_run())


def test_persistent_tbox_async_jobs_dlq_and_v02_api_mcp_endpoints() -> None:
    async def _run() -> None:
        svc = EnterpriseKGOntologyService(
            min_kb_support_for_approval=1,
            min_chunk_support_for_approval=1,
        )
        app = create_kg_fabric_app(service=svc)
        routes = {r.path: r.endpoint for r in app.routes if hasattr(r, "path")}

        await svc.ingest_raw_documents(
            [
                RawDocumentInput(
                    doc_id="d_job_1",
                    kb_id="kb_jobs",
                    tenant_id="t_jobs",
                    domain_tag="sre",
                    title="Job Test",
                    content="Microservice billing-api-01 depends on aurora-billing-01.",
                )
            ]
        )
        await svc.induce_and_align_ontology(tenant_id="t_jobs", tbox_version="2.0.0")

        # 1. Persistent Multi-Pod TBox Hydration Proof: fresh service sharing same vector_store
        pod2_svc = EnterpriseKGOntologyService(
            vector_store=svc.vector_store,
            graph_store=svc.graph_store,
        )
        hydrated_tbox = await pod2_svc.load_or_get_tbox("t_jobs")
        assert hydrated_tbox.version == "2.0.0"
        assert "CloudService" in hydrated_tbox.classes

        # 2. Async Sync Job Tracking & DLQ Fault Isolation
        # Inject a poison-pill failure on extractor once to verify DLQ catches it without aborting job
        original_extract = svc.kg_extractor.extract_from_chunk
        fail_once = {"triggered": False}

        def flaky_extract(chunk, tbox, allow_dynamic_emergent=True):
            if not fail_once["triggered"]:
                fail_once["triggered"] = True
                raise RuntimeError("Simulated LLM malformed JSON")
            return original_extract(chunk, tbox, allow_dynamic_emergent=allow_dynamic_emergent)

        svc.kg_extractor.extract_from_chunk = flaky_extract  # type: ignore[method-assign]
        job_rec = await routes["/v1/jobs/build"](
            BuildGraphRequest(tenant_id="t_jobs", incremental=True)
        )
        assert job_rec.status == SyncJobStatus.COMPLETED
        assert job_rec.dlq_count == 1

        dlq_items = await routes["/v1/jobs/dlq"](tenant_id="t_jobs")
        assert len(dlq_items) == 1
        assert "Simulated LLM malformed JSON" in dlq_items[0].error_message

        # Restore extractor and re-run job -> previously failed chunk wasn't watermarked so it succeeds now!
        svc.kg_extractor.extract_from_chunk = original_extract  # type: ignore[method-assign]
        job_rec_2 = await routes["/v1/jobs/build"](
            BuildGraphRequest(tenant_id="t_jobs", incremental=True)
        )
        assert job_rec_2.report is not None
        assert job_rec_2.report.delta_chunks_extracted == 1

        fetched_job = await routes["/v1/jobs/{job_id}"](job_rec_2.job_id)
        assert fetched_job.job_id == job_rec_2.job_id

        # 3. Verify v0.2.0 FastAPI & MCP endpoints (/v1/search/ppr, /v1/search/global, /v1/eval/run, MCP tools)
        ppr_api = await routes["/v1/search/ppr"](
            PPRSearchRequest(tenant_id="t_jobs", seed_entities=["billing-api-01"])
        )
        assert len(ppr_api.ranked_facts) >= 1

        global_api = await routes["/v1/search/global"](
            GlobalCommunitySearchRequest(tenant_id="t_jobs", query="billing architecture")
        )
        assert len(global_api.communities_matched) >= 1

        eval_api = await routes["/v1/eval/run"](
            RunEvalSuiteRequest(
                tenant_id="t_jobs",
                cases=[EvalTestCase(query="What does billing-api-01 depend on?", expected_facts=["aurora-billing-01"])],
            )
        )
        assert eval_api.mean_multihop_recall == 1.0

        mcp_ppr = await routes["/mcp"](
            {
                "jsonrpc": "2.0",
                "id": 100,
                "method": "tools/call",
                "params": {
                    "name": "search_ppr_subgraph",
                    "arguments": {"tenant_id": "t_jobs", "seed_entities": ["billing-api-01"]},
                },
            }
        )
        assert mcp_ppr["result"]["isError"] is False
        assert len(mcp_ppr["result"]["structuredContent"]["ranked_facts"]) >= 1

        mcp_global = await routes["/mcp"](
            {
                "jsonrpc": "2.0",
                "id": 101,
                "method": "tools/call",
                "params": {
                    "name": "query_global_communities",
                    "arguments": {"tenant_id": "t_jobs", "query": "billing architecture"},
                },
            }
        )
        assert mcp_global["result"]["isError"] is False
        assert len(mcp_global["result"]["structuredContent"]["communities_matched"]) >= 1

    run_coroutine_sync(_run())
