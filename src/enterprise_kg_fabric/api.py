from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .mcp_server import KGFabricMcpServer
from .models import (
    ChunkSearchRequest,
    EntitySearchMatch,
    EntitySearchRequest,
    GraphRAGQueryRequest,
    GraphRAGQueryResponse,
    KGGraphBuildReport,
    OntologyInductionReport,
    RawDocumentInput,
    ReadOnlyCypherSearchRequest,
    ReadOnlyCypherSearchResponse,
    RetrievedChunkContext,
    SubgraphSearchRequest,
    SubgraphSearchResponse,
    VectorChunkRecord,
    VectorKBRegistration,
)
from .service import EnterpriseKGOntologyService


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


def create_kg_fabric_app(
    service: EnterpriseKGOntologyService | None = None,
) -> FastAPI:
    svc = service or EnterpriseKGOntologyService()
    mcp_server = KGFabricMcpServer(svc)

    app = FastAPI(
        title="Enterprise Multi-KB Ontology & Knowledge Graph Fabric (FastAPI + MCP)",
        version="0.1.0",
        description=(
            "Co-hosted FastAPI REST & Model Context Protocol (MCP 2026-07-28 / 2025-11-25) "
            "Service for Automated Ontology Induction, Cross-KB Entity Resolution & "
            "LlamaIndex Hybrid GraphRAG over 100s of pgvector KBs and Neo4j/Neptune"
        ),
    )
    app.state.kg_service = svc
    app.state.mcp_server = mcp_server

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "service": "enterprise-kg-ontology-fabric",
            "mcp_cohosted": True,
            "mcp_tools_count": len(mcp_server.list_tools_metadata()),
        }

    # -------------------------------------------------------------------------
    # 1. Control & Ingestion Plane (`/v1/kbs/*`, `/v1/documents/*`, `/v1/ontology/*`, `/v1/graph/*`)
    # -------------------------------------------------------------------------

    @app.post("/v1/kbs/register", response_model=VectorKBRegistration)
    async def register_kb(reg: VectorKBRegistration) -> VectorKBRegistration:
        return await svc.register_vector_kb(reg)

    @app.post("/v1/kbs/register-batch")
    async def register_kbs_batch(req: BatchKBRegistrationRequest) -> dict[str, int]:
        count = await svc.register_vector_kbs_batch(req.registrations)
        return {"registered_count": count}

    @app.post("/v1/kbs/{kb_id}/hydrate")
    async def hydrate_kb(kb_id: str, req: HydrateKBChunksRequest) -> dict[str, Any]:
        count = await svc.hydrate_existing_kb_chunks(kb_id, req.chunks)
        return {"kb_id": kb_id, "hydrated_chunks": count}

    @app.post("/v1/documents/ingest")
    async def ingest_documents(req: BatchRawDocumentIngestRequest) -> dict[str, Any]:
        chunks = await svc.ingest_raw_documents(
            req.documents, native_kb_dim=req.native_kb_dim
        )
        return {
            "documents_ingested": len(req.documents),
            "chunks_created": len(chunks),
            "chunk_ids": [c.chunk_id for c in chunks],
        }

    @app.post("/v1/ontology/induce", response_model=OntologyInductionReport)
    async def induce_ontology(req: InduceOntologyRequest) -> OntologyInductionReport:
        _, report = await svc.induce_and_align_ontology(
            tenant_id=req.tenant_id,
            tbox_version=req.tbox_version,
            kb_ids=req.kb_ids,
        )
        return report

    @app.get("/v1/ontology/tbox")
    async def get_tbox(tenant_id: str = "enterprise_default") -> dict[str, Any]:
        return svc.get_tbox(tenant_id).model_dump()

    @app.get("/v1/ontology/export.ttl", response_class=PlainTextResponse)
    async def export_turtle(tenant_id: str = "enterprise_default") -> str:
        return svc.export_ontology_turtle(tenant_id)

    @app.post("/v1/ontology/promote")
    async def promote_ontology(req: PromoteOntologyRequest) -> dict[str, Any]:
        return await svc.promote_quarantined_ontology_concepts(
            tenant_id=req.tenant_id,
            min_confidence=req.min_confidence,
        )

    @app.post("/v1/graph/build", response_model=KGGraphBuildReport)
    async def build_graph(req: BuildGraphRequest) -> KGGraphBuildReport:
        return await svc.build_or_update_knowledge_graph(
            tenant_id=req.tenant_id,
            incremental=req.incremental,
            validate_shacl=req.validate_shacl,
            allow_dynamic_emergent=req.allow_dynamic_emergent,
        )

    # -------------------------------------------------------------------------
    # 2. Search & Retrieval Plane:
    #    - Level 1: One-Shot Orchestrated 4-Path Hybrid GraphRAG (`/v1/rag/query`)
    #    - Level 2: Granular Agentic Search Primitives (`/v1/search/*`)
    # -------------------------------------------------------------------------

    @app.post("/v1/rag/query", response_model=GraphRAGQueryResponse)
    async def query_rag(req: GraphRAGQueryRequest) -> GraphRAGQueryResponse:
        return await svc.query_graph_rag(req)

    @app.post("/v1/search/chunks", response_model=list[RetrievedChunkContext])
    async def search_chunks(req: ChunkSearchRequest) -> list[RetrievedChunkContext]:
        return await svc.search_vector_chunks(req)

    @app.post("/v1/search/entities", response_model=list[EntitySearchMatch])
    async def search_entities(req: EntitySearchRequest) -> list[EntitySearchMatch]:
        return await svc.search_kg_entities(req)

    @app.post("/v1/search/subgraph", response_model=SubgraphSearchResponse)
    async def search_subgraph(req: SubgraphSearchRequest) -> SubgraphSearchResponse:
        return await svc.traverse_entity_subgraph(req)

    @app.post("/v1/search/cypher", response_model=ReadOnlyCypherSearchResponse)
    async def search_cypher(req: ReadOnlyCypherSearchRequest) -> ReadOnlyCypherSearchResponse:
        try:
            return await svc.execute_readonly_opencypher(req)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/telemetry")
    async def get_telemetry(tenant_id: str = "enterprise_default") -> dict[str, Any]:
        return await svc.get_enterprise_telemetry(tenant_id=tenant_id)

    # -------------------------------------------------------------------------
    # 3. Co-Hosted Model Context Protocol (MCP) Surface (`/mcp`, `/mcp/http`, `/v1/mcp/*`)
    # -------------------------------------------------------------------------

    @app.post("/mcp")
    @app.post("/mcp/http")
    async def mcp_jsonrpc_endpoint(
        payload: dict[str, Any],
        mcp_protocol_version: str | None = Header(default="2026-07-28", alias="MCP-Protocol-Version"),
    ) -> dict[str, Any]:
        proto = mcp_protocol_version if isinstance(mcp_protocol_version, str) else "2026-07-28"
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
