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
        AnswerabilityCheckRequest,
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
