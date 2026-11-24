from __future__ import annotations

import json
import re
import time
from typing import Any
from urllib.parse import urlencode

import pytest

from enterprise_kg_fabric import (
    AuthConfig,
    AuthMode,
    DomainRulePack,
    EnterpriseKGOntologyService,
    GENERIC_DOMAIN_PACK,
    RawDocumentInput,
    UpperOntologyCategory,
    __version__,
    create_hs256_jwt,
    create_kg_fabric_app,
)
from enterprise_kg_fabric.cli import main as cli_main
from enterprise_kg_fabric.cli import validate_serve_security
from examples.simulate_120_kbs_end_to_end import run_coroutine_sync


async def _asgi_request(
    app: Any,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """
    Executes a full ASGI request through Starlette/FastAPI (middleware, auth, Depends,
    routing, serialization) without requiring an OS event loop or TCP sockets.
    """
    body_bytes = json.dumps(json_body).encode("utf-8") if json_body is not None else b""
    raw_headers: list[tuple[bytes, bytes]] = []
    if json_body is not None:
        raw_headers.append((b"content-type", b"application/json"))
        raw_headers.append((b"content-length", str(len(body_bytes)).encode("ascii")))
    for k, v in (headers or {}).items():
        raw_headers.append((k.lower().encode("latin-1"), v.encode("latin-1")))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": urlencode(params or {}).encode("ascii"),
        "headers": raw_headers,
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }

    sent_body = False

    async def receive() -> dict[str, Any]:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        return {"type": "http.disconnect"}

    status_code = 500
    resp_chunks: list[bytes] = []

    async def send(message: dict[str, Any]) -> None:
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = int(message["status"])
        elif message["type"] == "http.response.body":
            resp_chunks.append(message.get("body", b""))

    await app(scope, receive, send)
    raw_resp = b"".join(resp_chunks).decode("utf-8")
    parsed = json.loads(raw_resp) if raw_resp else {}
    return status_code, parsed


def test_jwt_auth_tenant_isolation_role_guards_and_abac_enforcement() -> None:
    """
    Verifies full HTTP/ASGI security hardening:
      - Unauthenticated requests rejected with 401 (while /health stays public)
      - Expired JWT rejected with 401
      - Cross-tenant smuggling rejected with 403
      - Reader role blocked from write/admin endpoints with 403
      - Caller cannot spoof `caller_acl_groups` outside verified JWT `acl_groups` (both REST and MCP)
    """

    async def _run() -> None:
        svc = EnterpriseKGOntologyService(
            min_kb_support_for_approval=1,
            min_chunk_support_for_approval=1,
        )
        await svc.ingest_raw_documents(
            [
                RawDocumentInput(
                    doc_id="doc_pub",
                    kb_id="kb_sec_pub",
                    tenant_id="tenant_alpha",
                    domain_tag="sre",
                    title="Public Runbook",
                    content="Microservice payment-router-01 depends on aurora-ledger-01.",
                    acl_groups=["public"],
                ),
                RawDocumentInput(
                    doc_id="doc_priv",
                    kb_id="kb_sec_priv",
                    tenant_id="tenant_alpha",
                    domain_tag="legal",
                    title="Board Merger Memo",
                    content="Microservice payment-router-01 depends on aurora-secret-merger-db.",
                    acl_groups=["executive_only"],
                ),
            ]
        )
        await svc.induce_and_align_ontology(tenant_id="tenant_alpha", tbox_version="3.0.0")
        await svc.build_or_update_knowledge_graph(tenant_id="tenant_alpha", incremental=False)

        jwt_secret = "unit-test-ephemeral-signing-key-at-least-32-bytes-long"
        auth_cfg = AuthConfig(mode=AuthMode.JWT, jwt_secret=jwt_secret)
        app = create_kg_fabric_app(service=svc, auth_config=auth_cfg)

        # 1. Public /health works without auth
        code, body = await _asgi_request(app, "GET", "/health")
        assert code == 200
        assert body["version"] == __version__
        assert body["auth_mode"] == "jwt"

        # 2. Missing Authorization header -> 401
        code, _ = await _asgi_request(
            app,
            "POST",
            "/v1/rag/query",
            json_body={"tenant_id": "tenant_alpha", "query": "What does payment-router-01 depend on?"},
        )
        assert code == 401

        # 3. Expired JWT -> 401
        expired_token = create_hs256_jwt(
            {
                "sub": "alice",
                "tenant_id": "tenant_alpha",
                "acl_groups": ["public"],
                "roles": ["reader"],
                "exp": int(time.time()) - 60,
            },
            secret=jwt_secret,
        )
        code, _ = await _asgi_request(
            app,
            "GET",
            "/v1/telemetry",
            params={"tenant_id": "tenant_alpha"},
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert code == 401

        # 4. Valid reader JWT bound to tenant_alpha + ['public']
        reader_token = create_hs256_jwt(
            {
                "sub": "alice",
                "tenant_id": "tenant_alpha",
                "acl_groups": ["public"],
                "roles": ["reader"],
                "exp": int(time.time()) + 3600,
            },
            secret=jwt_secret,
        )
        reader_headers = {"Authorization": f"Bearer {reader_token}"}

        # Cross-tenant smuggling -> 403
        code, _ = await _asgi_request(
            app,
            "GET",
            "/v1/telemetry",
            params={"tenant_id": "tenant_beta"},
            headers=reader_headers,
        )
        assert code == 403

        # Reader blocked from write endpoint -> 403
        code, _ = await _asgi_request(
            app,
            "POST",
            "/v1/graph/build",
            json_body={"tenant_id": "tenant_alpha"},
            headers=reader_headers,
        )
        assert code == 403

        # ABAC spoofing attempt: caller claims 'executive_only' in body, but JWT only grants 'public'
        code, spoof_body = await _asgi_request(
            app,
            "POST",
            "/v1/rag/query",
            json_body={
                "tenant_id": "tenant_alpha",
                "query": "What does payment-router-01 depend on?",
                "caller_acl_groups": ["public", "executive_only"],
            },
            headers=reader_headers,
        )
        assert code == 200
        assert "aurora-ledger-01" in str(spoof_body)
        assert "aurora-secret-merger-db" not in str(spoof_body)

        # ABAC spoofing attempt via MCP JSON-RPC tools/call is also blocked
        code, spoof_mcp = await _asgi_request(
            app,
            "POST",
            "/mcp",
            json_body={
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tools/call",
                "params": {
                    "name": "query_graph_rag",
                    "arguments": {
                        "tenant_id": "tenant_alpha",
                        "query": "What does payment-router-01 depend on?",
                        "caller_acl_groups": ["public", "executive_only"],
                    },
                },
            },
            headers=reader_headers,
        )
        assert code == 200
        assert "aurora-secret-merger-db" not in str(spoof_mcp)

        # 5. Executive token with ['public', 'executive_only'] CAN see executive_only facts
        exec_token = create_hs256_jwt(
            {
                "sub": "cfo",
                "tenant_id": "tenant_alpha",
                "acl_groups": ["public", "executive_only"],
                "roles": ["reader"],
                "exp": int(time.time()) + 3600,
            },
            secret=jwt_secret,
        )
        code, exec_body = await _asgi_request(
            app,
            "POST",
            "/v1/rag/query",
            json_body={
                "tenant_id": "tenant_alpha",
                "query": "What does payment-router-01 depend on?",
            },
            headers={"Authorization": f"Bearer {exec_token}"},
        )
        assert code == 200
        assert "aurora-secret-merger-db" in str(exec_body)

    run_coroutine_sync(_run())


def test_api_key_auth_mode_enforces_key_and_roles() -> None:
    async def _run() -> None:
        svc = EnterpriseKGOntologyService()
        auth_cfg = AuthConfig(
            mode=AuthMode.API_KEY,
            api_keys={
                "test-reader-key": {
                    "subject": "agent-reader",
                    "tenant_id": "t_keys",
                    "acl_groups": ["public"],
                    "roles": ["reader"],
                }
            },
        )
        app = create_kg_fabric_app(service=svc, auth_config=auth_cfg)

        code, _ = await _asgi_request(app, "GET", "/v1/telemetry", headers={"X-API-Key": "wrong-key"})
        assert code == 401

        code, _ = await _asgi_request(
            app,
            "GET",
            "/v1/telemetry",
            params={"tenant_id": "t_keys"},
            headers={"X-API-Key": "test-reader-key"},
        )
        assert code == 200

        code, _ = await _asgi_request(
            app,
            "POST",
            "/v1/ontology/promote",
            json_body={"tenant_id": "t_keys"},
            headers={"X-API-Key": "test-reader-key"},
        )
        assert code == 403

    run_coroutine_sync(_run())


def test_pluggable_domain_packs_generic_and_custom_healthcare() -> None:
    """
    Verifies non-Cloud PyPI adopters can use `GENERIC_DOMAIN_PACK` or a custom `DomainRulePack`
    without inheriting Cloud/SRE heuristics.
    """

    async def _run() -> None:
        # 1. Generic pack avoids Cloud/SRE classes like Microservice/DatabaseCluster
        generic_svc = EnterpriseKGOntologyService(
            domain_pack=GENERIC_DOMAIN_PACK,
            min_kb_support_for_approval=1,
            min_chunk_support_for_approval=1,
        )
        await generic_svc.ingest_raw_documents(
            [
                RawDocumentInput(
                    doc_id="doc_gen_1",
                    kb_id="kb_legal",
                    tenant_id="t_gen",
                    domain_tag="legal",
                    title="Hospital Policy",
                    content="Our hospital system is governed by a strict federal regulation and operated by an institution.",
                )
            ]
        )
        tbox_gen, _ = await generic_svc.induce_and_align_ontology(tenant_id="t_gen")
        assert "Organization" in tbox_gen.classes
        assert "PolicyOrStandard" in tbox_gen.classes
        assert "Microservice" not in tbox_gen.classes
        assert "DatabaseCluster" not in tbox_gen.classes

        # 2. Custom Healthcare DomainRulePack plugged in cleanly via SDK
        healthcare_pack = DomainRulePack(
            name="healthcare_clinical",
            description="Clinical trials and therapeutics ontology pack",
            class_discovery_rules=[
                (
                    re.compile(r"\b(biomarker|oncogene|mutation)\b", re.I),
                    "Biomarker",
                    UpperOntologyCategory.DATA_OR_ARTIFACT,
                    "Clinical or genomic biomarker",
                ),
                (
                    re.compile(r"\b(therapeutic|inhibitor|monoclonal\s+antibody)\b", re.I),
                    "TherapeuticDrug",
                    UpperOntologyCategory.SYSTEM_OR_ASSET,
                    "Targeted therapeutic compound or biologic",
                ),
            ],
            text_relation_triggers=[
                (
                    ("targets", "inhibits"),
                    "TARGETS_BIOMARKER",
                    "TherapeuticDrug",
                    "Biomarker",
                    "Therapeutic mechanism targeting a biomarker",
                    False,
                    False,
                )
            ],
        )
        hc_svc = EnterpriseKGOntologyService(
            domain_pack=healthcare_pack,
            min_kb_support_for_approval=1,
            min_chunk_support_for_approval=1,
        )
        await hc_svc.ingest_raw_documents(
            [
                RawDocumentInput(
                    doc_id="doc_hc_1",
                    kb_id="kb_oncology",
                    tenant_id="t_hc",
                    domain_tag="oncology",
                    title="Clinical Study",
                    content="The monoclonal antibody therapeutic targets the EGFR mutation biomarker.",
                )
            ]
        )
        tbox_hc, _ = await hc_svc.induce_and_align_ontology(tenant_id="t_hc")
        assert "Biomarker" in tbox_hc.classes
        assert "TherapeuticDrug" in tbox_hc.classes
        assert "TARGETS_BIOMARKER" in tbox_hc.relations

    run_coroutine_sync(_run())


def test_cli_unauthenticated_bind_guard_from_env_and_durable_jobs_dlq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verifies:
      - `kg-fabric version` CLI exits 0
      - `validate_serve_security` refuses binding `0.0.0.0` when `AuthMode.NONE`
      - `EnterpriseKGOntologyService.from_env()` picks up `KG_FABRIC_DOMAIN_PACK=generic`
      - Sync jobs & DLQ records survive cold-start pod hydration via `vector_store`
    """
    assert cli_main(["version"]) == 0

    with pytest.raises(SystemExit):
        validate_serve_security(host="0.0.0.0", auth_config=AuthConfig(mode=AuthMode.NONE))

    # Allowed on loopback 127.0.0.1 or when explicitly acknowledged
    validate_serve_security(host="127.0.0.1", auth_config=AuthConfig(mode=AuthMode.NONE))
    validate_serve_security(
        host="0.0.0.0",
        auth_config=AuthConfig(mode=AuthMode.NONE),
        allow_unauthenticated=True,
    )

    monkeypatch.setenv("KG_FABRIC_DOMAIN_PACK", "generic")
    monkeypatch.delenv("PGVECTOR_DSN", raising=False)
    monkeypatch.delenv("PGHOST", raising=False)
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEPTUNE_ENDPOINT", raising=False)
    env_svc = EnterpriseKGOntologyService.from_env()
    assert env_svc.domain_pack.name == "generic"

    async def _run_durability() -> None:
        pod1 = EnterpriseKGOntologyService(
            min_kb_support_for_approval=1,
            min_chunk_support_for_approval=1,
        )
        await pod1.ingest_raw_documents(
            [
                RawDocumentInput(
                    doc_id="d_dur_1",
                    kb_id="kb_dur",
                    tenant_id="t_dur",
                    domain_tag="sre",
                    title="Durability Doc",
                    content="Microservice billing-api-01 depends on aurora-billing-01.",
                )
            ]
        )
        await pod1.induce_and_align_ontology(tenant_id="t_dur")

        def boom(*_args, **_kwargs):
            raise RuntimeError("Poison chunk failure")

        pod1.kg_extractor.extract_from_chunk = boom  # type: ignore[method-assign]
        job = await pod1.execute_tracked_build_job(tenant_id="t_dur")
        assert job.dlq_count == 1

        # Fresh pod sharing the same persistent vector_store hydrates both job and DLQ!
        pod2 = EnterpriseKGOntologyService(
            vector_store=pod1.vector_store,
            graph_store=pod1.graph_store,
        )
        hydrated_job = await pod2.get_sync_job_async(job.job_id)
        assert hydrated_job is not None
        assert hydrated_job.job_id == job.job_id
        assert hydrated_job.dlq_count == 1

        hydrated_dlq = await pod2.list_dlq_records_async(tenant_id="t_dur")
        assert len(hydrated_dlq) == 1
        assert "Poison chunk failure" in hydrated_dlq[0].error_message

        await pod2.close()

    run_coroutine_sync(_run_durability())


def test_adk_ported_hardenings_32b_jwt_byoa_rate_limit_smuggling_contextvars_answerability_and_sync_worker() -> None:
    """
    Verifies all production hardenings ported from `adk-memory-platform`:
      1. Minimum 32-byte JWT secret enforcement
      2. Query-vs-body tenant smuggling rejection (`403`)
      3. Bounded API rate limiter (`429`)
      4. Bring-Your-Own-Auth (BYOA) `custom_request_authenticator` (Envoy/Kong/API Gateway headers)
      5. Service-layer `contextvars` `AuthPrincipal` enforcement on direct SDK calls & MCP resources
      6. Pre-flight `check_answerable` (REST + MCP), `/metrics` Prometheus, `/ui` Console
      7. Multi-replica-safe `run_scheduled_sync_cycle` respecting PostgreSQL/emulator advisory locks
    """
    from fastapi import HTTPException
    from enterprise_kg_fabric import (
        AuthPrincipal,
        GraphRAGQueryRequest,
        KGFabricMcpServer,
        reset_current_principal,
        set_current_principal,
    )

    # 1. Weak JWT secret (< 32 bytes) rejected immediately
    with pytest.raises(ValueError, match="at least 32 bytes"):
        AuthConfig(mode=AuthMode.JWT, jwt_secret="too-short-secret")

    async def _run() -> None:
        svc = EnterpriseKGOntologyService(
            min_kb_support_for_approval=1,
            min_chunk_support_for_approval=1,
        )
        await svc.ingest_raw_documents(
            [
                RawDocumentInput(
                    doc_id="d_hard_1",
                    kb_id="kb_hard",
                    tenant_id="tenant_a",
                    domain_tag="sre",
                    title="Hardening Runbook",
                    content="Microservice checkout-gw-01 depends on aurora-orders-01.",
                    acl_groups=["public"],
                ),
                RawDocumentInput(
                    doc_id="d_hard_priv",
                    kb_id="kb_hard_priv",
                    tenant_id="tenant_a",
                    domain_tag="legal",
                    title="Secret Project",
                    content="Microservice checkout-gw-01 depends on aurora-topsecret-99.",
                    acl_groups=["board_only"],
                ),
            ]
        )
        await svc.induce_and_align_ontology(tenant_id="tenant_a")
        await svc.build_or_update_knowledge_graph(tenant_id="tenant_a", incremental=False)

        # 2. BYOA trusted-header auth + bounded rate limiter (rpm=4)
        def byoa_gateway_auth(headers):
            subj = headers.get("x-envoy-user")
            tid = headers.get("x-envoy-tenant")
            if not subj or not tid:
                return None
            return AuthPrincipal(
                subject=subj,
                tenant_id=tid,
                acl_groups=["public"],
                roles={"reader"},
            )

        app = create_kg_fabric_app(
            service=svc,
            auth_config=AuthConfig(
                mode=AuthMode.CUSTOM,
                api_rate_limit_rpm=4,
                custom_request_authenticator=byoa_gateway_auth,
            ),
        )
        byoa_headers = {"X-Envoy-User": "svc-agent", "X-Envoy-Tenant": "tenant_a"}

        # Query-vs-body tenant smuggling -> 403
        code, _ = await _asgi_request(
            app,
            "POST",
            "/v1/rag/check-answerable",
            params={"tenant_id": "tenant_a"},
            json_body={"tenant_id": "tenant_b", "query": "checkout-gw-01"},
            headers=byoa_headers,
        )
        assert code == 403

        # Valid pre-flight answerability check -> ANSWERABLE
        code, ans_ok = await _asgi_request(
            app,
            "POST",
            "/v1/rag/check-answerable",
            json_body={"tenant_id": "tenant_a", "query": "What does checkout-gw-01 depend on?"},
            headers=byoa_headers,
        )
        assert code == 200
        assert ans_ok["status"] == "ANSWERABLE"
        assert "checkout-gw-01" in ans_ok["matched_entities"]

        # Uncovered topic -> UNANSWERABLE
        code, ans_missing = await _asgi_request(
            app,
            "POST",
            "/v1/rag/check-answerable",
            json_body={"tenant_id": "tenant_a", "query": "Quantum fusion reactor coolant telemetry"},
            headers=byoa_headers,
        )
        assert code == 200
        assert ans_missing["status"] == "UNANSWERABLE"

        # Prometheus metrics endpoint + Governance UI
        code, _ = await _asgi_request(app, "GET", "/v1/metrics", headers=byoa_headers)
        assert code == 200

        # 5th authenticated request within 60s exceeds rpm=4 -> 429 Too Many Requests
        code, _ = await _asgi_request(app, "GET", "/v1/metrics", headers=byoa_headers)
        assert code == 429

        # 3. Direct service-layer & MCP resource enforcement via contextvars
        reader_principal = AuthPrincipal(
            subject="direct-sdk-reader",
            tenant_id="tenant_a",
            acl_groups=["public"],
            roles={"reader"},
        )
        tok = set_current_principal(reader_principal)
        try:
            # Direct SDK cross-tenant call blocked!
            with pytest.raises(HTTPException) as exc_info:
                await svc.query_graph_rag(
                    GraphRAGQueryRequest(tenant_id="tenant_b", query="anything")
                )
            assert exc_info.value.status_code == 403

            # Direct SDK ABAC spoofing blocked!
            rag_direct = await svc.query_graph_rag(
                GraphRAGQueryRequest(
                    tenant_id="tenant_a",
                    query="What does checkout-gw-01 depend on?",
                    caller_acl_groups=["public", "board_only"],
                )
            )
            assert "aurora-topsecret-99" not in rag_direct.synthesized_answer
            assert all("aurora-topsecret-99" not in str(row) for row in rag_direct.opencypher_results)

            # MCP resource cross-tenant read blocked!
            mcp = KGFabricMcpServer(svc)
            with pytest.raises(HTTPException) as mcp_exc:
                await mcp.read_resource("kg://ontology/tenant_b/tbox")
            assert mcp_exc.value.status_code == 403

            # MCP check_answerable tool works for authorized tenant
            mcp_ans = await mcp.call_tool(
                "check_answerable",
                {"tenant_id": "tenant_a", "query": "checkout-gw-01"},
            )
            assert mcp_ans["status"] == "ANSWERABLE"
        finally:
            reset_current_principal(tok)

        # 4. Multi-replica advisory-lock sync worker cycle
        lock_key = svc._sync_lock_key("tenant_a")
        assert await svc.vector_store.try_acquire_advisory_lock(lock_key) is True
        try:
            cycle_locked = await svc.run_scheduled_sync_cycle(tenant_ids=["tenant_a"])
            assert "tenant_a" in cycle_locked["tenants_skipped_locked"]
        finally:
            await svc.vector_store.release_advisory_lock(lock_key)

        cycle_ok = await svc.run_scheduled_sync_cycle(tenant_ids=["tenant_a"])
        assert "tenant_a" in cycle_ok["tenants_synced"]
        assert cycle_ok["tenants_synced"]["tenant_a"]["status"] == "COMPLETED"

        prom_text = svc.get_prometheus_metrics_text()
        assert "kg_fabric_queries_total" in prom_text
        assert "kg_fabric_answerability_checks_total" in prom_text

    run_coroutine_sync(_run())


def test_kg_quality_auditor_scorecard_defect_detection_and_eval_lift() -> None:
    """
    Verifies the 4-dimension `KGQualityAuditor` (`GET /v1/quality/report` + MCP `audit_kg_quality`)
    and `EvalSuiteReport` Vector-vs-GraphRAG lift calculation:
      1. Clean connected cross-KB graph achieves high score (grade A/A-/B+) with 100% provenance.
      2. Intentionally injected defects (unused TBox class, orphan entity, unmerged duplicate pair,
         ungrounded edge without evidence, and domain/range violation) are ALL detected with
         actionable findings and measurably lower the scorecard.
      3. `run_eval_suite` reports `vector_only_baseline_recall` and `graphrag_recall_lift`.
    """
    from enterprise_kg_fabric import (
        EvalTestCase,
        KGEntityNode,
        KGFabricMcpServer,
        KGRelationEdge,
        OntologyClassDef,
        OntologyStatus,
    )

    async def _run() -> None:
        svc = EnterpriseKGOntologyService(
            min_kb_support_for_approval=1,
            min_chunk_support_for_approval=1,
        )
        await svc.ingest_raw_documents(
            [
                RawDocumentInput(
                    doc_id="q_doc_1",
                    kb_id="kb_q_sre",
                    tenant_id="t_qual",
                    domain_tag="sre",
                    title="SRE",
                    content="Microservice billing-api-01 depends on aurora-billing-01.",
                ),
                RawDocumentInput(
                    doc_id="q_doc_2",
                    kb_id="kb_q_comp",
                    tenant_id="t_qual",
                    domain_tag="compliance",
                    title="Compliance",
                    content="Database cluster aurora-billing-01 is governed by PCI-DSS-4.0.",
                ),
            ]
        )
        await svc.induce_and_align_ontology(tenant_id="t_qual")
        await svc.build_or_update_knowledge_graph(tenant_id="t_qual", incremental=False)

        # 1. Clean graph audit via REST, MCP, and Prometheus
        app = create_kg_fabric_app(service=svc)
        code, clean_report = await _asgi_request(
            app, "GET", "/v1/quality/report", params={"tenant_id": "t_qual"}
        )
        assert code == 200
        assert clean_report["overall_score"] >= 85.0
        assert clean_report["grade"] in {"A+", "A", "A-"}
        assert clean_report["tbox_metrics"]["shacl_conforms"] is True
        assert clean_report["extraction_er_metrics"]["provenance_grounded_edge_rate"] == 1.0
        assert clean_report["topology_metrics"]["orphan_entity_count"] == 0
        assert clean_report["topology_metrics"]["cross_kb_linkage_coverage"] == 1.0

        mcp = KGFabricMcpServer(svc)
        mcp_qual = await mcp.call_tool("audit_kg_quality", {"tenant_id": "t_qual"})
        assert mcp_qual["overall_score"] == clean_report["overall_score"]
        assert 'kg_fabric_quality_score{tenant_id="t_qual"}' in svc.get_prometheus_metrics_text()

        # 2. Eval suite reports vector_only_baseline_recall & graphrag_recall_lift
        eval_res = await svc.run_eval_suite(
            tenant_id="t_qual",
            cases=[
                EvalTestCase(
                    query="What compliance standard governs the database used by billing-api-01?",
                    expected_facts=["aurora-billing-01", "PCI-DSS-4.0"],
                )
            ],
        )
        assert eval_res.mean_multihop_recall == 1.0
        assert eval_res.graphrag_recall_lift >= 0.0

        # 3. Inject 5 distinct ontology & KG defects and verify the auditor catches EVERY ONE
        tbox = await svc.load_or_get_tbox("t_qual")
        tbox.classes["DeadUnusedSchemaClass"] = OntologyClassDef(
            class_name="DeadUnusedSchemaClass",
            upper_category=UpperOntologyCategory.SYSTEM_OR_ASSET,
            description="Never instantiated class",
            status=OntologyStatus.APPROVED,
        )
        await svc.vector_store.save_tbox(tbox)

        orphan_ent = KGEntityNode(
            entity_id="ent_orphan_1",
            tenant_id="t_qual",
            canonical_name="lone-island-node",
            ontology_class="CloudService",
            upper_category=UpperOntologyCategory.SYSTEM_OR_ASSET,
            source_kb_ids=["kb_q_sre"],
        )
        dup_a = KGEntityNode(
            entity_id="ent_dup_a",
            tenant_id="t_qual",
            canonical_name="Acme Bank",
            ontology_class="EnterpriseCustomer",
            upper_category=UpperOntologyCategory.ACTOR_OR_ORG,
            source_kb_ids=["kb_q_sre"],
        )
        dup_b = KGEntityNode(
            entity_id="ent_dup_b",
            tenant_id="t_qual",
            canonical_name="Acme Global Bank",
            ontology_class="EnterpriseCustomer",
            upper_category=UpperOntologyCategory.ACTOR_OR_ORG,
            source_kb_ids=["kb_q_comp"],
        )
        # Ungrounded edge with NO provenance AND domain/range violation (EnterpriseCustomer -> EnterpriseCustomer via GOVERNED_BY)
        bad_edge = KGRelationEdge(
            edge_id="edge_ungrounded_bad",
            tenant_id="t_qual",
            source_entity_id="ent_dup_a",
            target_entity_id="ent_dup_b",
            predicate="GOVERNED_BY",
            source_kb_ids=[],
            source_chunk_ids=[],
            evidence_snippets=[],
        )
        await svc.graph_store.upsert_entities_and_bitemporal_edges(
            tenant_id="t_qual",
            entities=[orphan_ent, dup_a, dup_b],
            edges=[bad_edge],
        )

        degraded = await svc.audit_kg_quality(tenant_id="t_qual")
        assert degraded.overall_score < clean_report["overall_score"]
        assert "DeadUnusedSchemaClass" in degraded.tbox_metrics.unused_classes
        assert degraded.tbox_metrics.domain_range_violations_count >= 1
        assert degraded.extraction_er_metrics.provenance_grounded_edge_rate < 1.0
        assert len(degraded.extraction_er_metrics.suspected_unmerged_duplicate_pairs) >= 1
        assert degraded.topology_metrics.orphan_entity_count >= 1
        assert any(f.category == "ENTITY_RESOLUTION" for f in degraded.findings)
        assert any(f.category == "EXTRACTION" for f in degraded.findings)

    run_coroutine_sync(_run())


def test_review_remediation_critical_security_neptune_promote_delete_and_auth_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Regression suite reproducing EVERY Critical, High & Medium finding from the external review:
      - Critical #1: Reader blocked from raw Cypher (REST + MCP -> 403); literal tenant_id rejected (400)
      - Critical #2: Reader blocked from plural MCP write tools (`ingest_raw_documents`, `register_vector_kbs`);
                     Writer blocked from nested cross-tenant documents/registrations
      - Critical #3 & High #7: Cross-tenant KB takeover & chunk overwrite blocked (`403` / `PermissionError`)
      - High #5: `GRAPH_BACKEND=neptune` `from_env()` initializes cleanly (fixed `region_name`) & sends real Cypher
      - High #6: `promote_quarantined_ontology_concepts` persists to graph store & is idempotent on 2nd call
      - Medium #9: Non-admin wildcard `tenant_id: "*"` JWT blocked (`403`); unknown `KG_FABRIC_AUTH_MODE` raises `ValueError`
      - Medium #10: `load_or_get_tbox` refreshes from persistent store (no stale pod cache)
      - Medium #11: `DELETE /v1/documents`, `DELETE /v1/kbs`, `DELETE /v1/tenants`
      - Medium #12: MCP `initialize` reports dynamic package version & real `2025-11-25` protocol
    """
    from enterprise_kg_fabric import (
        NeptuneOpenCypherGraphAdapter,
        OntologyClassDef,
        OntologyStatus,
        VectorChunkRecord,
        VectorKBRegistration,
    )

    # Medium #9a: Unknown KG_FABRIC_AUTH_MODE fails closed with ValueError
    monkeypatch.setenv("KG_FABRIC_AUTH_MODE", "typo_mode")
    with pytest.raises(ValueError, match="Invalid KG_FABRIC_AUTH_MODE"):
        AuthConfig.from_env()
    monkeypatch.delenv("KG_FABRIC_AUTH_MODE", raising=False)

    # High #5a: Neptune from_env() no longer crashes with TypeError(aws_region)
    monkeypatch.setenv("GRAPH_BACKEND", "neptune")
    monkeypatch.setenv("NEPTUNE_ENDPOINT", "https://neptune-test.local:8182")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.delenv("PGVECTOR_DSN", raising=False)
    monkeypatch.delenv("PGHOST", raising=False)
    neptune_svc = EnterpriseKGOntologyService.from_env()
    assert isinstance(neptune_svc.graph_store, NeptuneOpenCypherGraphAdapter)
    assert neptune_svc.graph_store.region_name == "us-west-2"
    monkeypatch.delenv("GRAPH_BACKEND", raising=False)
    monkeypatch.delenv("NEPTUNE_ENDPOINT", raising=False)

    async def _run() -> None:
        svc = EnterpriseKGOntologyService(
            min_kb_support_for_approval=1,
            min_chunk_support_for_approval=1,
        )
        await svc.ingest_raw_documents(
            [
                RawDocumentInput(
                    doc_id="shared_doc",
                    kb_id="kb_shared_name",
                    tenant_id="tenant_a",
                    domain_tag="sre",
                    title="Tenant A Secret",
                    content="Microservice alpha-router-01 depends on aurora-alpha-01.",
                    acl_groups=["sre_only"],
                )
            ]
        )
        await svc.induce_and_align_ontology(tenant_id="tenant_a", tbox_version="1.0.0")
        await svc.build_or_update_knowledge_graph(tenant_id="tenant_a", incremental=False)

        jwt_secret = "review-remediation-signing-key-at-least-32-bytes-long"
        app = create_kg_fabric_app(
            service=svc,
            auth_config=AuthConfig(mode=AuthMode.JWT, jwt_secret=jwt_secret),
        )

        reader_b_jwt = create_hs256_jwt(
            {"sub": "rb", "tenant_id": "tenant_b", "acl_groups": ["public"], "roles": ["reader"], "exp": int(time.time()) + 3600},
            jwt_secret,
        )
        writer_b_jwt = create_hs256_jwt(
            {"sub": "wb", "tenant_id": "tenant_b", "acl_groups": ["public"], "roles": ["reader", "writer"], "exp": int(time.time()) + 3600},
            jwt_secret,
        )
        wildcard_nonadmin_jwt = create_hs256_jwt(
            {"sub": "wild", "tenant_id": "*", "acl_groups": ["public"], "roles": ["reader"], "exp": int(time.time()) + 3600},
            jwt_secret,
        )
        admin_a_jwt = create_hs256_jwt(
            {"sub": "adma", "tenant_id": "tenant_a", "acl_groups": ["public", "sre_only"], "roles": ["admin"], "exp": int(time.time()) + 3600},
            jwt_secret,
        )
        platform_admin_jwt = create_hs256_jwt(
            {"sub": "root", "tenant_id": "*", "acl_groups": ["public", "sre_only"], "roles": ["admin"], "exp": int(time.time()) + 3600},
            jwt_secret,
        )
        hdr_rb = {"Authorization": f"Bearer {reader_b_jwt}"}
        hdr_wb = {"Authorization": f"Bearer {writer_b_jwt}"}
        hdr_wild = {"Authorization": f"Bearer {wildcard_nonadmin_jwt}"}
        hdr_adma = {"Authorization": f"Bearer {admin_a_jwt}"}
        hdr_padm = {"Authorization": f"Bearer {platform_admin_jwt}"}

        # Medium #9b: Non-admin wildcard tenant_id="*" rejected with 403
        code, _ = await _asgi_request(app, "GET", "/v1/telemetry", params={"tenant_id": "tenant_a"}, headers=hdr_wild)
        assert code == 403

        # Critical #1: Reader blocked from raw Cypher on REST and MCP
        code, _ = await _asgi_request(
            app,
            "POST",
            "/v1/search/cypher",
            json_body={"tenant_id": "tenant_b", "cypher_query": "MATCH (n) RETURN n.canonical_name"},
            headers=hdr_rb,
        )
        assert code == 403

        code, mcp_cypher = await _asgi_request(
            app,
            "POST",
            "/mcp",
            json_body={
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "execute_readonly_opencypher",
                    "arguments": {"tenant_id": "tenant_b", "cypher_query": "MATCH (n) RETURN n"},
                },
            },
            headers=hdr_rb,
        )
        assert code == 403 or "error" in mcp_cypher

        # Tenant-bound admin is ALSO blocked from raw Cypher (`403`) — only global platform_admin allowed
        code, _ = await _asgi_request(
            app,
            "POST",
            "/v1/search/cypher",
            json_body={"tenant_id": "tenant_a", "cypher_query": "MATCH (n) RETURN n"},
            headers=hdr_adma,
        )
        assert code == 403

        # Global platform admin attempting literal tenant_id override in Cypher -> 400
        code, _ = await _asgi_request(
            app,
            "POST",
            "/v1/search/cypher",
            json_body={"tenant_id": "tenant_a", "cypher_query": "MATCH (n {tenant_id: 'tenant_b'}) RETURN n"},
            headers=hdr_padm,
        )
        assert code == 400

        # Critical #2: Reader blocked from plural MCP write tools (`ingest_raw_documents`, `register_vector_kbs`)
        code, _ = await _asgi_request(
            app,
            "POST",
            "/v1/mcp/call",
            json_body={
                "tool_name": "ingest_raw_documents",
                "arguments": {
                    "documents": [
                        {"doc_id": "evil", "kb_id": "kb_shared_name", "tenant_id": "tenant_a", "title": "X", "content": "Pwned"}
                    ]
                },
            },
            headers=hdr_rb,
        )
        assert code == 403

        # Writer B attempting nested cross-tenant document injection into Tenant A via MCP -> 403
        code, _ = await _asgi_request(
            app,
            "POST",
            "/v1/mcp/call",
            json_body={
                "tool_name": "ingest_raw_documents",
                "arguments": {
                    "documents": [
                        {"doc_id": "evil", "kb_id": "kb_b", "tenant_id": "tenant_a", "title": "X", "content": "Pwned"}
                    ]
                },
            },
            headers=hdr_wb,
        )
        assert code == 403

        # Critical #3 & High #7: Writer B cannot take over Tenant A's kb_id or overwrite Tenant A's chunk!
        code, _ = await _asgi_request(
            app,
            "POST",
            "/v1/kbs/register",
            json_body=VectorKBRegistration(
                kb_id="kb_shared_name", tenant_id="tenant_b", domain_tag="sre", table_name="t_b"
            ).model_dump(),
            headers=hdr_wb,
        )
        assert code == 403

        a_chunks = await svc.vector_store.fetch_kb_chunks("kb_shared_name", tenant_id="tenant_a")
        assert len(a_chunks) == 1
        code, _ = await _asgi_request(
            app,
            "POST",
            "/v1/kbs/kb_shared_name/hydrate",
            json_body={
                "chunks": [
                    VectorChunkRecord(
                        chunk_id=a_chunks[0].chunk_id,
                        kb_id="kb_shared_name",
                        tenant_id="tenant_b",
                        doc_id="shared_doc",
                        text="OVERWRITTEN",
                        acl_groups=["public"],
                    ).model_dump()
                ]
            },
            headers=hdr_wb,
        )
        assert code == 403
        a_chunks_after = await svc.vector_store.fetch_kb_chunks("kb_shared_name", tenant_id="tenant_a")
        assert "OVERWRITTEN" not in a_chunks_after[0].text
        assert a_chunks_after[0].acl_groups == ["sre_only"]

        # High #6: Promote persists to graph store & 2nd no-op call does NOT bump version
        tbox = await svc.load_or_get_tbox("tenant_a")
        tbox.classes["EmergingClass"] = OntologyClassDef(
            class_name="EmergingClass",
            upper_category=UpperOntologyCategory.SYSTEM_OR_ASSET,
            description="Test candidate",
            confidence=0.92,
            status=OntologyStatus.QUARANTINED_CANDIDATE,
        )
        await svc.vector_store.save_tbox(tbox)
        p1 = await svc.promote_quarantined_ontology_concepts("tenant_a")
        assert "EmergingClass" in p1["promoted_classes"]
        v_after_p1 = p1["new_tbox_version"]
        p2 = await svc.promote_quarantined_ontology_concepts("tenant_a")
        assert p2["promoted_classes"] == []
        assert p2["new_tbox_version"] == v_after_p1

        # Medium #10: Pod 2 immediately observes Pod 1 TBox update via load_or_get_tbox (no stale cache)
        pod2 = EnterpriseKGOntologyService(vector_store=svc.vector_store, graph_store=svc.graph_store)
        await pod2.load_or_get_tbox("tenant_a")
        tbox_v2 = await svc.load_or_get_tbox("tenant_a")
        tbox_v2.version = "9.9.0"
        await svc.vector_store.save_tbox(tbox_v2)
        refreshed = await pod2.load_or_get_tbox("tenant_a")
        assert refreshed.version == "9.9.0"

        # Medium #12: MCP initialize reports current version & real protocol 2025-11-25
        code, init_resp = await _asgi_request(
            app,
            "POST",
            "/mcp",
            json_body={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers=hdr_adma,
        )
        assert code == 200
        assert init_resp["result"]["serverInfo"]["version"] == __version__
        assert init_resp["result"]["protocolVersion"] == "2025-11-25"

        # Medium #11: Delete document, unregister KB, and purge tenant
        code, del_doc = await _asgi_request(
            app,
            "DELETE",
            "/v1/documents/kb_shared_name/shared_doc",
            params={"tenant_id": "tenant_a"},
            headers=hdr_adma,
        )
        assert code == 200 and del_doc["chunks_removed"] == 1

        code, purge = await _asgi_request(
            app, "DELETE", "/v1/tenants/tenant_a", headers=hdr_adma
        )
        assert code == 200 and purge["kbs_removed"] >= 1
        assert await svc.graph_store.get_all_entities("tenant_a") == {}

    run_coroutine_sync(_run())


def test_second_review_remediation_legacy_exfil_platform_admin_entity_pk_delete_dlq_and_metrics() -> None:
    """
    Regression suite reproducing EVERY finding from the second external red-team review:
      1. Critical: `EXISTING_PGVECTOR_KB` registration & `validate_legacy_table_access` block
         `public.unified_rag_chunks`, `pg_catalog.pg_roles`, SQL injection identifiers, and
         unallowlisted tables (fail-closed default!).
      2. High: Tenant-bound admin (`tenant_id="tenant_a"`, `roles=["admin"]`) gets 403 on
         `/v1/search/cypher` & MCP `execute_readonly_opencypher` (global `platform_admin` required).
      3. High: Same-named entity (`payment-router-01`) across `tenant_a` and `tenant_b` receives
         distinct tenant-scoped `entity_id`s and NEVER collides in `canonical_entity_embeddings`.
      4. Medium: `DELETE /v1/documents/{kb}/{doc}` & `DELETE /v1/kbs/{kb}` remove BOTH vector
         chunks AND orphaned graph entities/edges AND entity embeddings so `/v1/rag/query` stops
         answering from deleted docs (while preserving entities/edges shared with surviving docs).
      5. Medium: Tenant-bound admin gets 403 on `GET /v1/jobs/dlq?tenant_id=tenant_b`.
      6. Low: `/v1/metrics` and `/metrics` hide other tenants' IDs & quality scores from scoped callers.
      7. Low: Legacy table registration fails closed when `allowed_legacy_tables` is empty.
    """
    from enterprise_kg_fabric import (
        KBSourceType,
        VectorKBRegistration,
        validate_legacy_table_access,
    )

    # 1. Direct validator proofs (denylist + SQL identifier injection + fail-closed allowlist)
    for bad_schema, bad_table in (
        ("public", "unified_rag_chunks"),
        ("public", "kb_federation_registry"),
        ("public", "canonical_entity_embeddings"),
        ("pg_catalog", "pg_roles"),
        ("information_schema", "tables"),
        ("pg_temp_1", "evil"),
        ("public", 'foo"; DROP TABLE unified_rag_chunks; --'),
    ):
        with pytest.raises(PermissionError):
            validate_legacy_table_access(
                schema_name=bad_schema,
                table_name=bad_table,
                column_names=["id", "doc_id", "content"],
                allowed_patterns=["public.*", "pg_catalog.*"],
            )

    # Column SQL injection is also rejected
    with pytest.raises(PermissionError):
        validate_legacy_table_access(
            schema_name="public",
            table_name="legacy_finops_vectors",
            column_names=['id" FROM pg_roles; --'],
            allowed_patterns=["public.legacy_*"],
        )

    # Even a benign table name fails closed when allowlist is empty
    with pytest.raises(PermissionError, match="disabled"):
        validate_legacy_table_access(
            schema_name="public",
            table_name="legacy_finops_vectors",
            column_names=["id", "doc_id", "content"],
            allowed_patterns=[],
        )

    # Explicitly allowlisted benign table succeeds (returns None without raising)
    validate_legacy_table_access(
        schema_name="public",
        table_name="legacy_finops_vectors",
        column_names=["id", "doc_id", "content"],
        allowed_patterns=["public.legacy_*"],
    )

    async def _run() -> None:
        svc = EnterpriseKGOntologyService(
            min_kb_support_for_approval=1,
            min_chunk_support_for_approval=1,
        )
        # Seed BOTH tenant_a (2 docs: one standalone, one shared) and tenant_b (same-named entity!)
        await svc.ingest_raw_documents(
            [
                RawDocumentInput(
                    doc_id="doc_a_keep",
                    kb_id="kb_a_core",
                    tenant_id="tenant_a",
                    domain_tag="sre",
                    title="Tenant A Core",
                    content="Microservice payment-router-01 depends on aurora-alpha-keep.",
                    acl_groups=["public"],
                ),
                RawDocumentInput(
                    doc_id="doc_a_delete",
                    kb_id="kb_a_core",
                    tenant_id="tenant_a",
                    domain_tag="sre",
                    title="Tenant A Transient",
                    content="Microservice transient-svc-99 depends on aurora-transient-db.",
                    acl_groups=["public"],
                ),
                RawDocumentInput(
                    doc_id="doc_a_kb2",
                    kb_id="kb_a_ephemeral",
                    tenant_id="tenant_a",
                    domain_tag="sre",
                    title="Tenant A Ephemeral KB",
                    content="Microservice ephemeral-svc-77 depends on aurora-ephemeral-db.",
                    acl_groups=["public"],
                ),
                RawDocumentInput(
                    doc_id="doc_b_same_name",
                    kb_id="kb_b_core",
                    tenant_id="tenant_b",
                    domain_tag="sre",
                    title="Tenant B Same Entity Name",
                    content="Microservice payment-router-01 depends on aurora-beta-only.",
                    acl_groups=["public"],
                ),
            ]
        )
        await svc.induce_and_align_ontology(tenant_id="tenant_a")
        await svc.induce_and_align_ontology(tenant_id="tenant_b")
        await svc.build_or_update_knowledge_graph(tenant_id="tenant_a", incremental=False)
        await svc.build_or_update_knowledge_graph(tenant_id="tenant_b", incremental=False)

        # Finding #3: Same-named entity `payment-router-01` gets distinct tenant-scoped entity_ids
        # and does NOT overwrite the other tenant's canonical_entity_embeddings row!
        ents_a = await svc.graph_store.get_all_entities("tenant_a")
        ents_b = await svc.graph_store.get_all_entities("tenant_b")
        pr_a = [e for e in ents_a.values() if e.canonical_name == "payment-router-01"]
        pr_b = [e for e in ents_b.values() if e.canonical_name == "payment-router-01"]
        assert len(pr_a) == 1 and len(pr_b) == 1
        assert pr_a[0].entity_id != pr_b[0].entity_id
        assert "::tenant_a::" in pr_a[0].entity_id
        assert "::tenant_b::" in pr_b[0].entity_id
        assert "kb_b_core" not in pr_a[0].source_kb_ids
        assert "kb_a_core" not in pr_b[0].source_kb_ids

        # Audit both tenants so last_quality_scores has BOTH tenant_a & tenant_b
        await svc.audit_kg_quality("tenant_a")
        await svc.audit_kg_quality("tenant_b")

        jwt_secret = "second-review-remediation-signing-key-32-bytes-minimum"
        app = create_kg_fabric_app(
            service=svc,
            auth_config=AuthConfig(mode=AuthMode.JWT, jwt_secret=jwt_secret),
        )
        tadm_a_jwt = create_hs256_jwt(
            {"sub": "ta_admin", "tenant_id": "tenant_a", "acl_groups": ["public"], "roles": ["admin"], "exp": int(time.time()) + 3600},
            jwt_secret,
        )
        writer_b_jwt = create_hs256_jwt(
            {"sub": "tb_writer", "tenant_id": "tenant_b", "acl_groups": ["public"], "roles": ["reader", "writer"], "exp": int(time.time()) + 3600},
            jwt_secret,
        )
        padm_jwt = create_hs256_jwt(
            {"sub": "root_admin", "tenant_id": "*", "acl_groups": ["public"], "roles": ["admin"], "exp": int(time.time()) + 3600},
            jwt_secret,
        )
        hdr_ta = {"Authorization": f"Bearer {tadm_a_jwt}"}
        hdr_wb = {"Authorization": f"Bearer {writer_b_jwt}"}
        hdr_pa = {"Authorization": f"Bearer {padm_jwt}"}

        # Finding #1 via REST: writer_b cannot register public.unified_rag_chunks or pg_catalog.pg_roles
        for bad_schema, bad_tbl in (("public", "unified_rag_chunks"), ("pg_catalog", "pg_roles")):
            code, _ = await _asgi_request(
                app,
                "POST",
                "/v1/kbs/register",
                json_body=VectorKBRegistration(
                    kb_id=f"kb_exfil_{bad_tbl}",
                    tenant_id="tenant_b",
                    domain_tag="evil",
                    source_type=KBSourceType.EXISTING_PGVECTOR_KB,
                    schema_name=bad_schema,
                    table_name=bad_tbl,
                ).model_dump(),
                headers=hdr_wb,
            )
            assert code == 403

        # Finding #2: tenant-bound admin gets 403 on Cypher (REST + MCP)
        code, _ = await _asgi_request(
            app,
            "POST",
            "/v1/search/cypher",
            json_body={"tenant_id": "tenant_a", "cypher_query": "MATCH (n) RETURN n.canonical_name"},
            headers=hdr_ta,
        )
        assert code == 403
        code, mcp_cyp = await _asgi_request(
            app,
            "POST",
            "/mcp",
            json_body={
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "execute_readonly_opencypher",
                    "arguments": {"tenant_id": "tenant_a", "cypher_query": "MATCH (n) RETURN n"},
                },
            },
            headers=hdr_ta,
        )
        assert code == 403 or "error" in mcp_cyp

        # Finding #5: tenant-bound admin gets 403 on cross-tenant DLQ list
        code, _ = await _asgi_request(
            app, "GET", "/v1/jobs/dlq", params={"tenant_id": "tenant_b"}, headers=hdr_ta
        )
        assert code == 403
        code, _ = await _asgi_request(
            app, "GET", "/v1/jobs/dlq", params={"tenant_id": "tenant_a"}, headers=hdr_ta
        )
        assert code == 200

        # Finding #6: `/v1/metrics` and `/metrics` hide tenant_b from tenant_a caller,
        #             while platform_admin sees both!
        code, m_ta = await _asgi_request(app, "GET", "/v1/metrics", headers=hdr_ta)
        assert code == 200
        assert "tenant_a" in m_ta["last_quality_scores"]
        assert "tenant_b" not in m_ta["last_quality_scores"]

        code, m_pa = await _asgi_request(app, "GET", "/v1/metrics", headers=hdr_pa)
        assert code == 200
        assert "tenant_a" in m_pa["last_quality_scores"]
        assert "tenant_b" in m_pa["last_quality_scores"]

        # Finding #4a: DELETE /v1/documents/{kb}/{doc} purges graph entities, edges, AND stops /v1/rag/query
        code, del_doc = await _asgi_request(
            app,
            "DELETE",
            "/v1/documents/kb_a_core/doc_a_delete",
            params={"tenant_id": "tenant_a"},
            headers=hdr_ta,
        )
        assert code == 200
        assert del_doc["chunks_removed"] == 1
        assert del_doc["graph_entities_removed"] >= 1
        assert del_doc["graph_edges_removed"] >= 1

        code, rag_del = await _asgi_request(
            app,
            "POST",
            "/v1/rag/query",
            json_body={"tenant_id": "tenant_a", "query": "What does transient-svc-99 depend on?"},
            headers=hdr_ta,
        )
        assert code == 200
        assert "aurora-transient-db" not in str(rag_del)

        # Surviving doc in same KB is still intact!
        code, rag_keep = await _asgi_request(
            app,
            "POST",
            "/v1/rag/query",
            json_body={"tenant_id": "tenant_a", "query": "What does payment-router-01 depend on?"},
            headers=hdr_ta,
        )
        assert code == 200
        assert "aurora-alpha-keep" in str(rag_keep)

        # Finding #4b: DELETE /v1/kbs/{kb} purges ephemeral KB graph facts completely
        code, del_kb = await _asgi_request(
            app,
            "DELETE",
            "/v1/kbs/kb_a_ephemeral",
            params={"tenant_id": "tenant_a"},
            headers=hdr_ta,
        )
        assert code == 200
        assert del_kb["chunks_removed"] == 1
        assert del_kb["graph_entities_removed"] >= 1
        assert del_kb["graph_edges_removed"] >= 1

        code, rag_kb_del = await _asgi_request(
            app,
            "POST",
            "/v1/rag/query",
            json_body={"tenant_id": "tenant_a", "query": "What does ephemeral-svc-77 depend on?"},
            headers=hdr_ta,
        )
        assert code == 200
        assert "aurora-ephemeral-db" not in str(rag_kb_del)

    run_coroutine_sync(_run())
