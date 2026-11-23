from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .auth import AuthConfig, AuthEnforcer, AuthPrincipal
from .mcp_server import KGFabricMcpServer
from .models import (
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


def _apply_principal_to_mcp_arguments(
    enforcer: AuthEnforcer,
    principal: AuthPrincipal | None,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Enforces tenant isolation, role checks, and ABAC ACL groups on MCP tool calls."""
    if principal is None:
        return arguments

    write_tools = {"register_vector_kb", "ingest_raw_document", "induce_ontology", "build_knowledge_graph"}
    if tool_name in write_tools:
        enforcer.require_role(principal, "writer")

    sanitized = dict(arguments)
    req_tenant = str(sanitized.get("tenant_id") or principal.tenant_id or "enterprise_default")
    sanitized["tenant_id"] = enforcer.enforce_tenant(principal, req_tenant)

    acl_tools = {"query_graph_rag", "search_vector_chunks", "traverse_entity_subgraph", "search_ppr_subgraph", "query_global_communities"}
    if tool_name in acl_tools or "caller_acl_groups" in sanitized:
        req_acls = sanitized.get("caller_acl_groups")
        sanitized["caller_acl_groups"] = enforcer.enforce_acl_groups(
            principal,
            list(req_acls) if isinstance(req_acls, list) else None,
        )
    return sanitized


def create_kg_fabric_app(
    service: EnterpriseKGOntologyService | None = None,
    auth_config: AuthConfig | None = None,
) -> FastAPI:
    """
    Creates the co-hosted FastAPI REST + MCP (`2026-07-28`) application.

    - When `service` is omitted, auto-wires PostgreSQL/Neo4j/Neptune/Presidio/DomainPack
      via `EnterpriseKGOntologyService.from_env()` (falling back cleanly to in-memory emulators).
    - When `auth_config` is omitted, reads `KG_FABRIC_AUTH_MODE` (`none` | `api_key` | `jwt`).
    """
    svc = service or EnterpriseKGOntologyService.from_env()
    effective_auth = auth_config or AuthConfig.from_env()
    enforcer = AuthEnforcer(effective_auth)
    mcp_server = KGFabricMcpServer(svc)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
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

    _public_paths = {"/health", "/docs", "/openapi.json", "/redoc"}

    class _AuthASGIMiddleware:
        """Pure-ASGI authentication middleware (zero BaseHTTPMiddleware overhead or SSE buffering)."""

        def __init__(self, inner_app: Any) -> None:
            self.inner_app = inner_app

        async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
            if scope.get("type") != "http" or scope.get("path") in _public_paths:
                await self.inner_app(scope, receive, send)
                return
            request = Request(scope, receive=receive)
            try:
                principal = enforcer.authenticate_request(request)
                scope.setdefault("state", {})["principal"] = principal
            except HTTPException as exc:
                response = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
                await response(scope, receive, send)
                return
            await self.inner_app(scope, receive, send)

    app.add_middleware(_AuthASGIMiddleware)

    async def get_principal(request: Request) -> AuthPrincipal | None:
        if hasattr(request.state, "principal"):
            return request.state.principal
        return enforcer.authenticate_request(request)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "service": "enterprise-kg-ontology-fabric",
            "version": API_VERSION,
            "auth_mode": effective_auth.mode.value,
            "domain_pack": svc.domain_pack.name,
            "mcp_cohosted": True,
            "mcp_tools_count": len(mcp_server.list_tools_metadata()),
        }

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
    #    - Level 1: One-Shot Orchestrated Adaptive Hybrid GraphRAG (`/v1/rag/query`)
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

    @app.post("/v1/jobs/build", response_model=SyncJobRecord)
    async def submit_build_job(
        req: BuildGraphRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> SyncJobRecord:
        p = _unwrap_principal(principal)
        enforcer.require_role(p, "writer")
        tenant = enforcer.enforce_tenant(p, req.tenant_id)
        return await svc.execute_tracked_build_job(
            tenant_id=tenant,
            incremental=req.incremental,
            validate_shacl=req.validate_shacl,
        )

    @app.get("/v1/jobs/dlq", response_model=list[DLQFailedChunkRecord])
    async def list_dlq(
        tenant_id: str | None = None,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> list[DLQFailedChunkRecord]:
        p = _unwrap_principal(principal)
        if p is not None and not p.is_admin and p.tenant_id:
            tenant_id = enforcer.enforce_tenant(p, tenant_id or p.tenant_id)
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
        mcp_protocol_version: str | None = Header(default="2026-07-28", alias="MCP-Protocol-Version"),
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> dict[str, Any]:
        p = _unwrap_principal(principal)
        proto = mcp_protocol_version if isinstance(mcp_protocol_version, str) else "2026-07-28"
        if p is not None and payload.get("method") == "tools/call":
            params = dict(payload.get("params") or {})
            tool_name = str(params.get("name", ""))
            raw_args = dict(params.get("arguments") or {})
            params["arguments"] = _apply_principal_to_mcp_arguments(enforcer, p, tool_name, raw_args)
            payload = {**payload, "params": params}
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
    async def call_mcp_tool_rest(
        req: McpToolCallRestRequest,
        principal: AuthPrincipal | None = Depends(get_principal),
    ) -> dict[str, Any]:
        p = _unwrap_principal(principal)
        safe_args = _apply_principal_to_mcp_arguments(enforcer, p, req.tool_name, req.arguments)
        try:
            result = await mcp_server.call_tool(req.tool_name, safe_args)
            return {"tool_name": req.tool_name, "result": result, "isError": False}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def create_kg_fabric_app_from_env() -> FastAPI:
    """Zero-argument ASGI factory for `uvicorn` and Docker/EKS deployments."""
    return create_kg_fabric_app()
