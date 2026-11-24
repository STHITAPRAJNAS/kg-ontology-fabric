from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any
from urllib.parse import parse_qs

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .auth import (
    AuthConfig,
    AuthEnforcer,
    AuthPrincipal,
    reset_current_principal,
    set_current_principal,
)
from .mcp_server import KGFabricMcpServer
from .models import (
    AnswerabilityCheckRequest,
    AnswerabilityCheckResponse,
    ChunkSearchRequest,
    DLQFailedChunkRecord,
    EntitySearchMatch,
    EntitySearchRequest,
    EvalSuiteReport,
    EvalTestCase,
    GlobalCommunitySearchRequest,
    GlobalCommunitySearchResponse,
    GraphRAGQueryRequest,
    GraphRAGQueryResponse,
    KGGraphBuildReport,
    KGQualityScorecardReport,
    OntologyAuditEntry,
    OntologyConceptReviewRequest,
    OntologyInductionReport,
    PPRSearchRequest,
    PPRSearchResponse,
    RawDocumentInput,
    ReadOnlyCypherSearchRequest,
    ReadOnlyCypherSearchResponse,
    RetrievedChunkContext,
    SubgraphSearchRequest,
    SubgraphSearchResponse,
    SyncJobRecord,
    VectorChunkRecord,
    VectorKBRegistration,
)
from .service import EnterpriseKGOntologyService

try:
    API_VERSION = _pkg_version("enterprise-kg-ontology-fabric")
except PackageNotFoundError:
    API_VERSION = "0.3.0"


class BatchKBRegistrationRequest(BaseModel):
    registrations: list[VectorKBRegistration]


class HydrateKBChunksRequest(BaseModel):
    chunks: list[VectorChunkRecord]


class BatchRawDocumentIngestRequest(BaseModel):
    documents: list[RawDocumentInput]
    native_kb_dim: int = 768


class InduceOntologyRequest(BaseModel):
    tenant_id: str = "enterprise_default"
    tbox_version: str = "1.0.0"
    kb_ids: list[str] | None = None


class BuildGraphRequest(BaseModel):
    tenant_id: str = "enterprise_default"
    incremental: bool = True
    validate_shacl: bool = True
    allow_dynamic_emergent: bool = True
    wait: bool = False


class PromoteOntologyRequest(BaseModel):
    tenant_id: str = "enterprise_default"
    min_confidence: float = Field(default=0.80, ge=0.0, le=1.0)


class McpToolCallRestRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class RunEvalSuiteRequest(BaseModel):
    tenant_id: str = "enterprise_default"
    cases: list[EvalTestCase]


def _unwrap_principal(value: Any) -> AuthPrincipal | None:
    return value if isinstance(value, AuthPrincipal) else None


_GOVERNANCE_UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Enterprise KG Ontology Fabric — Governance Console</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }
    h1 { color: #38bdf8; margin-bottom: 4px; }
    .sub { color: #94a3b8; margin-bottom: 20px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; }
    input, button { padding: 8px 10px; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: #f8fafc; }
    button { background: #0284c7; border-color: #0284c7; cursor: pointer; font-weight: 600; }
    pre { background: #020617; padding: 12px; border-radius: 6px; overflow: auto; max-height: 320px; font-size: 12px; }
  </style>
</head>
<body>
  <h1>Enterprise KG Ontology Fabric</h1>
  <div class="sub">Ontology TBox, Telemetry, DLQ &amp; Hybrid GraphRAG Governance Console</div>
  <div style="margin-bottom:16px; display:flex; gap:8px; flex-wrap:wrap;">
    <input id="tenant" value="enterprise_default" placeholder="tenant_id" />
    <input id="token" type="password" placeholder="Optional Bearer JWT or X-API-Key" style="width:280px;" />
    <button onclick="refreshAll()">Refresh Dashboard</button>
  </div>
  <div class="grid">
    <div class="card"><h3>Health &amp; Worker</h3><pre id="health">Loading...</pre></div>
    <div class="card"><h3>KG &amp; Ontology Quality Scorecard</h3><pre id="quality">Click Refresh</pre></div>
    <div class="card"><h3>Tenant Telemetry</h3><pre id="telemetry">Click Refresh</pre></div>
    <div class="card">
      <h3>Per-Concept Ontology Review</h3>
      <div style="display:flex; gap:6px; flex-wrap:wrap; margin-bottom:8px;">
        <select id="revKind" style="padding:8px;border-radius:6px;background:#0f172a;color:#f8fafc;border:1px solid #475569;">
          <option value="CLASS">CLASS</option>
          <option value="RELATION">RELATION</option>
        </select>
        <input id="revConcept" placeholder="Concept name" style="width:140px;" />
        <input id="revTarget" placeholder="Target (for merge)" style="width:140px;" />
        <input id="revReason" placeholder="Reason" style="width:140px;" />
      </div>
      <div style="display:flex; gap:6px; margin-bottom:8px;">
        <button onclick="submitReview('approve')">Approve</button>
        <button onclick="submitReview('reject')" style="background:#dc2626;border-color:#dc2626;">Reject</button>
        <button onclick="submitReview('merge')" style="background:#7c3aed;border-color:#7c3aed;">Merge</button>
      </div>
      <pre id="revResult">Ready for review actions</pre>
    </div>
    <div class="card"><h3>Compiled TBox Ontology</h3><pre id="tbox">Click Refresh</pre></div>
    <div class="card"><h3>Ontology Audit Trail</h3><pre id="audit">Click Refresh</pre></div>
    <div class="card"><h3>Build Jobs</h3><pre id="jobs">Click Refresh</pre></div>
    <div class="card"><h3>Dead-Letter Queue (DLQ)</h3><pre id="dlq">Click Refresh</pre></div>
  </div>
  <script>
    function authHeaders() {
      const t = document.getElementById('token').value.trim();
      if (!t) return {};
      return t.includes('.') ? {'Authorization': 'Bearer ' + t} : {'X-API-Key': t};
    }
    async function loadJson(url, elId) {
      try {
        const res = await fetch(url, {headers: authHeaders()});
        document.getElementById(elId).textContent = JSON.stringify(await res.json(), null, 2);
      } catch (e) { document.getElementById(elId).textContent = String(e); }
    }
    async function submitReview(action) {
      const tid = document.getElementById('tenant').value || 'enterprise_default';
      const body = {
        tenant_id: tid,
        concept_kind: document.getElementById('revKind').value,
        concept_name: document.getElementById('revConcept').value.trim(),
        action: action,
        target_name: document.getElementById('revTarget').value.trim() || null,
        reason: document.getElementById('revReason').value.trim(),
      };
      try {
        const res = await fetch('/v1/ontology/review', {
          method: 'POST',
          headers: {'Content-Type': 'application/json', ...authHeaders()},
          body: JSON.stringify(body),
        });
        document.getElementById('revResult').textContent = JSON.stringify(await res.json(), null, 2);
        refreshAll();
      } catch (e) { document.getElementById('revResult').textContent = String(e); }
    }
    function refreshAll() {
      const tid = encodeURIComponent(document.getElementById('tenant').value || 'enterprise_default');
      loadJson('/health', 'health');
      loadJson('/v1/quality/report?tenant_id=' + tid, 'quality');
      loadJson('/v1/telemetry?tenant_id=' + tid, 'telemetry');
      loadJson('/v1/ontology/tbox?tenant_id=' + tid, 'tbox');
      loadJson('/v1/ontology/audit?tenant_id=' + tid, 'audit');
      loadJson('/v1/jobs?tenant_id=' + tid, 'jobs');
      loadJson('/v1/jobs/dlq?tenant_id=' + tid, 'dlq');
    }
    loadJson('/health', 'health');
  </script>
</body>
</html>
"""


def create_kg_fabric_app(
    service: EnterpriseKGOntologyService | None = None,
    auth_config: AuthConfig | None = None,
    enable_sync_worker: bool | None = None,
    sync_interval_sec: float | None = None,
) -> FastAPI:
    """
    Creates the co-hosted FastAPI REST + MCP (`2026-07-28`) application.

    - When `service` is omitted, auto-wires PostgreSQL/Neo4j/Neptune/Presidio/DomainPack
      via `EnterpriseKGOntologyService.from_env()` (falling back cleanly to in-memory emulators).
    - When `auth_config` is omitted, reads `KG_FABRIC_AUTH_MODE` (`none` | `api_key` | `jwt`).
    - Pure-ASGI auth middleware enforces AuthN, bounded rate limiting (`429`), query-vs-body
      tenant smuggling rejection (`403`), and propagates `AuthPrincipal` via `contextvars`
      into `EnterpriseKGOntologyService` and `KGFabricMcpServer`.
    """
    svc = service or EnterpriseKGOntologyService.from_env()
    effective_auth = auth_config or AuthConfig.from_env()
    enforcer = AuthEnforcer(effective_auth)
    mcp_server = KGFabricMcpServer(svc)

    worker_enabled = (
        enable_sync_worker
        if enable_sync_worker is not None
        else os.getenv("KG_FABRIC_ENABLE_SYNC_WORKER", "false").strip().lower() in {"1", "true", "yes"}
    )
    worker_interval = (
        sync_interval_sec
        if sync_interval_sec is not None
        else float(os.getenv("KG_FABRIC_SYNC_INTERVAL_SEC", "300"))
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if worker_enabled:
            svc.start_sync_worker(interval_sec=worker_interval)
        try:
            yield
        finally:
            await svc.close()

    app = FastAPI(
        title="Enterprise Multi-KB Ontology & Knowledge Graph Fabric (FastAPI + MCP)",
        version=API_VERSION,
        description=(
            "Co-hosted FastAPI REST & Model Context Protocol (MCP 2026-07-28 / 2025-11-25) "
            "Service for Automated Ontology Induction, Cross-KB Entity Resolution, "
            "HippoRAG 2 PPR, Leiden Global Search, ABAC Security Trimming & "
            "Hybrid GraphRAG over 100s of pgvector KBs and Neo4j/Neptune"
        ),
        lifespan=lifespan,
    )
    app.state.kg_service = svc
    app.state.mcp_server = mcp_server
    app.state.auth_enforcer = enforcer

    @app.exception_handler(PermissionError)
    async def _permission_error_handler(_request: Request, exc: PermissionError) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    _public_paths = {"/health", "/healthz", "/docs", "/openapi.json", "/redoc", "/ui"}

    class _AuthASGIMiddleware:
        """
        Pure-ASGI security middleware:
          - Authenticates request (JWT / API key / BYOA custom hooks)
          - Enforces bounded per-principal / per-IP rate limiting (`429`)
          - Rejects query-vs-body `tenant_id` smuggling (`403`)
          - Propagates `AuthPrincipal` via `contextvars` to service & MCP layers
        """

        def __init__(self, inner_app: Any) -> None:
            self.inner_app = inner_app

        async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
            if scope.get("type") != "http" or scope.get("path") in _public_paths:
                await self.inner_app(scope, receive, send)
                return

            request = Request(scope, receive=receive)
            try:
                principal = enforcer.authenticate_request(request)
                enforcer.check_rate_limit(request, principal)
            except HTTPException as exc:
                response = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
                await response(scope, receive, send)
                return

            # Query-vs-body tenant smuggling defense for JSON POST/PUT/PATCH requests
            replay_receive = receive
            query_params = parse_qs(scope.get("query_string", b"").decode("latin-1"))
            query_tenants = query_params.get("tenant_id", [])
            if query_tenants and scope.get("method", "").upper() in {"POST", "PUT", "PATCH"}:
                chunks: list[bytes] = []
                while True:
                    msg = await receive()
                    chunks.append(msg.get("body", b""))
                    if not msg.get("more_body", False):
                        break
                full_body = b"".join(chunks)
                try:
                    parsed_body = json.loads(full_body.decode("utf-8")) if full_body else None
                except Exception:
                    parsed_body = None
                if isinstance(parsed_body, dict):
                    body_tenant = parsed_body.get("tenant_id")
                    if body_tenant and str(body_tenant) != str(query_tenants[0]):
                        resp = JSONResponse(
                            status_code=403,
                            content={
                                "detail": (
                                    f"Conflicting tenant_id between query ('{query_tenants[0]}') "
                                    f"and request body ('{body_tenant}')"
                                )
                            },
                        )
                        await resp(scope, receive, send)
                        return

                sent_cached = False

                async def _replay() -> dict[str, Any]:
                    nonlocal sent_cached
                    if not sent_cached:
                        sent_cached = True
                        return {"type": "http.request", "body": full_body, "more_body": False}
                    return {"type": "http.disconnect"}

                replay_receive = _replay

            scope.setdefault("state", {})["principal"] = principal
            ctx_token = set_current_principal(principal)
            try:
                await self.inner_app(scope, replay_receive, send)
            finally:
                reset_current_principal(ctx_token)

    app.add_middleware(_AuthASGIMiddleware)

    async def get_principal(request: Request) -> AuthPrincipal | None:
        if hasattr(request.state, "principal"):
            return request.state.principal
        return enforcer.authenticate_request(request)

    @app.get("/health")
    @app.get("/healthz")
    async def health() -> dict[str, Any]:
        worker = svc.sync_worker_status()
        return {
            "status": "healthy",
            "service": "enterprise-kg-ontology-fabric",
            "version": API_VERSION,
            "auth_mode": effective_auth.mode.value,
            "domain_pack": svc.domain_pack.name,
            "mcp_cohosted": True,
            "mcp_tools_count": len(mcp_server.list_tools_metadata()),
            "sync_worker_running": worker["running"],
            "sync_worker_last_cycle": worker["last_cycle"],
        }

    @app.get("/ui", response_class=HTMLResponse)
    async def governance_console_ui() -> str:
        return _GOVERNANCE_UI_HTML

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics(
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> str:
        p = _unwrap_principal(principal)
        scoped_tid = None if (p is None or p.is_platform_admin) else p.tenant_id
        return svc.get_prometheus_metrics_text(tenant_id=scoped_tid)

    @app.get("/v1/metrics")
    async def json_metrics(
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> dict[str, Any]:
        p = _unwrap_principal(principal)
        scoped_tid = None if (p is None or p.is_platform_admin) else p.tenant_id
        return svc.get_metrics_snapshot(tenant_id=scoped_tid)

    # -------------------------------------------------------------------------
    # 1. Control & Ingestion Plane (`/v1/kbs/*`, `/v1/documents/*`, `/v1/ontology/*`, `/v1/graph/*`)
    # -------------------------------------------------------------------------

    @app.post("/v1/kbs/register", response_model=VectorKBRegistration)
    async def register_kb(
        reg: VectorKBRegistration,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> VectorKBRegistration:
        p = _unwrap_principal(principal)
        enforcer.require_role(p, "writer")
        reg.tenant_id = enforcer.enforce_tenant(p, reg.tenant_id)
        return await svc.register_vector_kb(reg)

    @app.post("/v1/kbs/register-batch")
    async def register_kbs_batch(
        req: BatchKBRegistrationRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> dict[str, int]:
        p = _unwrap_principal(principal)
        enforcer.require_role(p, "writer")
        for reg in req.registrations:
            reg.tenant_id = enforcer.enforce_tenant(p, reg.tenant_id)
        count = await svc.register_vector_kbs_batch(req.registrations)
        return {"registered_count": count}

    @app.post("/v1/kbs/{kb_id}/hydrate")
    async def hydrate_kb(
        kb_id: str,
        req: HydrateKBChunksRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> dict[str, Any]:
        p = _unwrap_principal(principal)
        enforcer.require_role(p, "writer")
        for chk in req.chunks:
            chk.tenant_id = enforcer.enforce_tenant(p, chk.tenant_id)
        count = await svc.hydrate_existing_kb_chunks(kb_id, req.chunks)
        return {"kb_id": kb_id, "hydrated_chunks": count}

    @app.post("/v1/documents/ingest")
    async def ingest_documents(
        req: BatchRawDocumentIngestRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> dict[str, Any]:
        p = _unwrap_principal(principal)
        enforcer.require_role(p, "writer")
        for doc in req.documents:
            doc.tenant_id = enforcer.enforce_tenant(p, doc.tenant_id)
        chunks = await svc.ingest_raw_documents(
            req.documents, native_kb_dim=req.native_kb_dim
        )
        return {
            "documents_ingested": len(req.documents),
            "chunks_created": len(chunks),
            "chunk_ids": [c.chunk_id for c in chunks],
        }

    @app.post("/v1/ontology/induce", response_model=OntologyInductionReport)
    async def induce_ontology(
        req: InduceOntologyRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> OntologyInductionReport:
        p = _unwrap_principal(principal)
        enforcer.require_role(p, "writer")
        tenant = enforcer.enforce_tenant(p, req.tenant_id)
        _, report = await svc.induce_and_align_ontology(
            tenant_id=tenant,
            tbox_version=req.tbox_version,
            kb_ids=req.kb_ids,
        )
        return report

    @app.get("/v1/ontology/tbox")
    async def get_tbox(
        tenant_id: str = "enterprise_default",
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> dict[str, Any]:
        p = _unwrap_principal(principal)
        tenant = enforcer.enforce_tenant(p, tenant_id)
        tbox = await svc.load_or_get_tbox(tenant)
        return tbox.model_dump()

    @app.get("/v1/ontology/export.ttl", response_class=PlainTextResponse)
    async def export_turtle(
        tenant_id: str = "enterprise_default",
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> str:
        p = _unwrap_principal(principal)
        tenant = enforcer.enforce_tenant(p, tenant_id)
        await svc.load_or_get_tbox(tenant)
        return svc.export_ontology_turtle(tenant)

    @app.post("/v1/ontology/promote")
    async def promote_ontology(
        req: PromoteOntologyRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> dict[str, Any]:
        p = _unwrap_principal(principal)
        enforcer.require_role(p, "admin")
        tenant = enforcer.enforce_tenant(p, req.tenant_id)
        return await svc.promote_quarantined_ontology_concepts(
            tenant_id=tenant,
            min_confidence=req.min_confidence,
        )

    @app.post("/v1/graph/build", response_model=KGGraphBuildReport)
    async def build_graph(
        req: BuildGraphRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> KGGraphBuildReport:
        p = _unwrap_principal(principal)
        enforcer.require_role(p, "writer")
        tenant = enforcer.enforce_tenant(p, req.tenant_id)
        return await svc.build_or_update_knowledge_graph(
            tenant_id=tenant,
            incremental=req.incremental,
            validate_shacl=req.validate_shacl,
            allow_dynamic_emergent=req.allow_dynamic_emergent,
        )

    # -------------------------------------------------------------------------
    # 2. Search & Retrieval Plane:
    #    - Level 1: One-Shot Orchestrated Adaptive Hybrid GraphRAG (`/v1/rag/*`)
    #    - Level 2: Granular Agentic Search Primitives (`/v1/search/*`)
    # -------------------------------------------------------------------------

    @app.post("/v1/rag/query", response_model=GraphRAGQueryResponse)
    async def query_rag(
        req: GraphRAGQueryRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> GraphRAGQueryResponse:
        p = _unwrap_principal(principal)
        req.tenant_id = enforcer.enforce_tenant(p, req.tenant_id)
        req.caller_acl_groups = enforcer.enforce_acl_groups(p, req.caller_acl_groups)
        return await svc.query_graph_rag(req)

    @app.post("/v1/rag/check-answerable", response_model=AnswerabilityCheckResponse)
    async def check_answerable_endpoint(
        req: AnswerabilityCheckRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> AnswerabilityCheckResponse:
        p = _unwrap_principal(principal)
        req.tenant_id = enforcer.enforce_tenant(p, req.tenant_id)
        req.caller_acl_groups = enforcer.enforce_acl_groups(p, req.caller_acl_groups)
        return await svc.check_answerable(req)

    @app.post("/v1/search/chunks", response_model=list[RetrievedChunkContext])
    async def search_chunks(
        req: ChunkSearchRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> list[RetrievedChunkContext]:
        p = _unwrap_principal(principal)
        req.tenant_id = enforcer.enforce_tenant(p, req.tenant_id)
        req.caller_acl_groups = enforcer.enforce_acl_groups(p, req.caller_acl_groups)
        return await svc.search_vector_chunks(req)

    @app.post("/v1/search/entities", response_model=list[EntitySearchMatch])
    async def search_entities(
        req: EntitySearchRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> list[EntitySearchMatch]:
        p = _unwrap_principal(principal)
        req.tenant_id = enforcer.enforce_tenant(p, req.tenant_id)
        req.caller_acl_groups = enforcer.enforce_acl_groups(p, req.caller_acl_groups)
        return await svc.search_kg_entities(req)

    @app.post("/v1/search/subgraph", response_model=SubgraphSearchResponse)
    async def search_subgraph(
        req: SubgraphSearchRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> SubgraphSearchResponse:
        p = _unwrap_principal(principal)
        req.tenant_id = enforcer.enforce_tenant(p, req.tenant_id)
        req.caller_acl_groups = enforcer.enforce_acl_groups(p, req.caller_acl_groups)
        return await svc.traverse_entity_subgraph(req)

    @app.post("/v1/search/cypher", response_model=ReadOnlyCypherSearchResponse)
    async def search_cypher(
        req: ReadOnlyCypherSearchRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> ReadOnlyCypherSearchResponse:
        p = _unwrap_principal(principal)
        enforcer.require_platform_admin(p)
        req.tenant_id = enforcer.enforce_tenant(p, req.tenant_id)
        try:
            return await svc.execute_readonly_opencypher(req)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/search/ppr", response_model=PPRSearchResponse)
    async def search_ppr(
        req: PPRSearchRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> PPRSearchResponse:
        p = _unwrap_principal(principal)
        req.tenant_id = enforcer.enforce_tenant(p, req.tenant_id)
        req.caller_acl_groups = enforcer.enforce_acl_groups(p, req.caller_acl_groups)
        return await svc.search_ppr_subgraph(req)

    @app.post("/v1/search/global", response_model=GlobalCommunitySearchResponse)
    async def search_global_communities(
        req: GlobalCommunitySearchRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> GlobalCommunitySearchResponse:
        p = _unwrap_principal(principal)
        req.tenant_id = enforcer.enforce_tenant(p, req.tenant_id)
        req.caller_acl_groups = enforcer.enforce_acl_groups(p, req.caller_acl_groups)
        return await svc.query_global_communities(req)

    @app.post("/v1/jobs/build", response_model=SyncJobRecord, status_code=202)
    async def submit_build_job(
        req: BuildGraphRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> SyncJobRecord:
        p = _unwrap_principal(principal)
        enforcer.require_role(p, "writer")
        tenant = enforcer.enforce_tenant(p, req.tenant_id)
        if req.wait:
            return await svc.execute_tracked_build_job(
                tenant_id=tenant,
                incremental=req.incremental,
                validate_shacl=req.validate_shacl,
            )
        return await svc.submit_async_build_job(
            tenant_id=tenant,
            incremental=req.incremental,
            validate_shacl=req.validate_shacl,
        )

    @app.get("/v1/jobs", response_model=list[SyncJobRecord])
    async def list_jobs(
        tenant_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> list[SyncJobRecord]:
        p = _unwrap_principal(principal)
        if p is not None and not p.is_platform_admin:
            tenant_id = enforcer.enforce_tenant(p, tenant_id or p.tenant_id or "")
        return await svc.list_sync_jobs_async(
            tenant_id=tenant_id, limit=max(1, min(limit, 200)), offset=max(0, offset)
        )

    @app.get("/v1/jobs/dlq", response_model=list[DLQFailedChunkRecord])
    async def list_dlq(
        tenant_id: str | None = None,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> list[DLQFailedChunkRecord]:
        p = _unwrap_principal(principal)
        if p is not None and not p.is_platform_admin:
            tenant_id = enforcer.enforce_tenant(p, tenant_id or p.tenant_id or "")
        return await svc.list_dlq_records_async(tenant_id=tenant_id)

    @app.get("/v1/jobs/{job_id}", response_model=SyncJobRecord)
    async def get_job_status(
        job_id: str,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> SyncJobRecord:
        p = _unwrap_principal(principal)
        job = await svc.get_sync_job_async(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        enforcer.enforce_tenant(p, job.tenant_id)
        return job

    @app.post("/v1/jobs/{job_id}/cancel", response_model=SyncJobRecord)
    async def cancel_job(
        job_id: str,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> SyncJobRecord:
        p = _unwrap_principal(principal)
        enforcer.require_role(p, "writer")
        job = await svc.get_sync_job_async(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        enforcer.enforce_tenant(p, job.tenant_id)
        cancelled = await svc.cancel_sync_job(job_id)
        assert cancelled is not None
        return cancelled

    @app.get("/v1/kbs", response_model=list[VectorKBRegistration])
    async def list_kbs(
        tenant_id: str = "enterprise_default",
        limit: int = 50,
        offset: int = 0,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> list[VectorKBRegistration]:
        p = _unwrap_principal(principal)
        tenant = enforcer.enforce_tenant(p, tenant_id)
        return await svc.list_registered_kbs_paginated(
            tenant_id=tenant, limit=max(1, min(limit, 200)), offset=max(0, offset)
        )

    @app.get("/v1/kbs/{kb_id}/documents")
    async def list_kb_docs(
        kb_id: str,
        tenant_id: str = "enterprise_default",
        limit: int = 50,
        offset: int = 0,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> list[dict[str, Any]]:
        p = _unwrap_principal(principal)
        tenant = enforcer.enforce_tenant(p, tenant_id)
        return await svc.list_kb_documents(
            tenant_id=tenant, kb_id=kb_id, limit=max(1, min(limit, 200)), offset=max(0, offset)
        )

    @app.get("/v1/entities/{entity_id}")
    async def get_entity(
        entity_id: str,
        tenant_id: str = "enterprise_default",
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> dict[str, Any]:
        p = _unwrap_principal(principal)
        tenant = enforcer.enforce_tenant(p, tenant_id)
        acls = enforcer.enforce_acl_groups(p, None)
        details = await svc.get_entity_details(
            tenant_id=tenant, entity_id=entity_id, caller_acl_groups=acls
        )
        if details is None:
            raise HTTPException(status_code=404, detail=f"Entity not found: {entity_id}")
        return details

    @app.post("/v1/ontology/review")
    async def review_ontology_concept(
        req: OntologyConceptReviewRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> dict[str, Any]:
        p = _unwrap_principal(principal)
        enforcer.require_role(p, "admin")
        req.tenant_id = enforcer.enforce_tenant(p, req.tenant_id)
        try:
            return await svc.review_ontology_concept(req)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/ontology/audit", response_model=list[OntologyAuditEntry])
    async def get_ontology_audit(
        tenant_id: str = "enterprise_default",
        limit: int = 100,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> list[OntologyAuditEntry]:
        p = _unwrap_principal(principal)
        tenant = enforcer.enforce_tenant(p, tenant_id)
        return await svc.get_ontology_audit_trail(
            tenant_id=tenant, limit=max(1, min(limit, 500))
        )

    @app.post("/v1/eval/run", response_model=EvalSuiteReport)
    async def run_eval(
        req: RunEvalSuiteRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> EvalSuiteReport:
        p = _unwrap_principal(principal)
        tenant = enforcer.enforce_tenant(p, req.tenant_id)
        for case in req.cases:
            case.caller_acl_groups = enforcer.enforce_acl_groups(p, case.caller_acl_groups)
        return await svc.run_eval_suite(tenant_id=tenant, cases=req.cases)

    @app.get("/v1/quality/report", response_model=KGQualityScorecardReport)
    async def get_quality_report(
        tenant_id: str = "enterprise_default",
        sample_size: int = 25,
        use_llm_judge: bool = False,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> KGQualityScorecardReport:
        p = _unwrap_principal(principal)
        if use_llm_judge:
            enforcer.require_role(p, "writer")
        tenant = enforcer.enforce_tenant(p, tenant_id)
        return await svc.audit_kg_quality(
            tenant_id=tenant,
            sample_size_faithfulness=sample_size,
            use_llm_judge=use_llm_judge,
        )

    @app.delete("/v1/kbs/{kb_id}")
    async def delete_kb(
        kb_id: str,
        tenant_id: str = "enterprise_default",
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> dict[str, Any]:
        p = _unwrap_principal(principal)
        enforcer.require_role(p, "admin")
        tenant = enforcer.enforce_tenant(p, tenant_id)
        return await svc.unregister_kb(tenant_id=tenant, kb_id=kb_id)

    @app.delete("/v1/documents/{kb_id}/{doc_id}")
    async def delete_document_endpoint(
        kb_id: str,
        doc_id: str,
        tenant_id: str = "enterprise_default",
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> dict[str, Any]:
        p = _unwrap_principal(principal)
        enforcer.require_role(p, "writer")
        tenant = enforcer.enforce_tenant(p, tenant_id)
        return await svc.delete_document(tenant_id=tenant, kb_id=kb_id, doc_id=doc_id)

    @app.delete("/v1/tenants/{tenant_id}")
    async def purge_tenant_endpoint(
        tenant_id: str,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> dict[str, Any]:
        p = _unwrap_principal(principal)
        enforcer.require_role(p, "admin")
        tenant = enforcer.enforce_tenant(p, tenant_id)
        return await svc.purge_tenant(tenant_id=tenant)

    @app.get("/v1/telemetry")
    async def get_telemetry(
        tenant_id: str = "enterprise_default",
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> dict[str, Any]:
        p = _unwrap_principal(principal)
        tenant = enforcer.enforce_tenant(p, tenant_id)
        return await svc.get_enterprise_telemetry(tenant_id=tenant)

    # -------------------------------------------------------------------------
    # 3. Co-Hosted Model Context Protocol (MCP) Surface (`/mcp`, `/mcp/http`, `/v1/mcp/*`)
    # -------------------------------------------------------------------------

    @app.post("/mcp")
    @app.post("/mcp/http")
    async def mcp_jsonrpc_endpoint(
        payload: dict[str, Any],
        mcp_protocol_version: str | None = Header(default="2025-11-25", alias="MCP-Protocol-Version"),
    ) -> dict[str, Any]:
        proto = mcp_protocol_version if isinstance(mcp_protocol_version, str) else "2025-11-25"
        return await mcp_server.handle_jsonrpc(payload, protocol_version_header=proto)

    @app.get("/v1/mcp/tools")
    async def list_mcp_tools() -> dict[str, Any]:
        return {
            "tools": mcp_server.list_tools_metadata(),
            "resources": mcp_server.list_resources_metadata(),
            "resourceTemplates": mcp_server.list_resource_templates_metadata(),
            "prompts": mcp_server.list_prompts_metadata(),
        }

    @app.post("/v1/mcp/call")
    async def call_mcp_tool_rest(req: McpToolCallRestRequest) -> dict[str, Any]:
        try:
            result = await mcp_server.call_tool(req.tool_name, req.arguments)
            return {"tool_name": req.tool_name, "result": result, "isError": False}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def create_kg_fabric_app_from_env() -> FastAPI:
    """Zero-argument ASGI factory for `uvicorn` and Docker/EKS deployments."""
    return create_kg_fabric_app()
