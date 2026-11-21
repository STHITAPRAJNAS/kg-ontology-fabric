from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .models import (
    GraphRAGQueryRequest,
    GraphRAGQueryResponse,
    KGGraphBuildReport,
    OntologyInductionReport,
    RawDocumentInput,
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


def create_kg_fabric_app(
    service: EnterpriseKGOntologyService | None = None,
) -> FastAPI:
    svc = service or EnterpriseKGOntologyService()
    app = FastAPI(
        title="Enterprise Multi-KB Ontology & Knowledge Graph Fabric",
        version="0.1.0",
        description="Automated Ontology Induction, Cross-KB Entity Resolution & LlamaIndex Hybrid GraphRAG over 100s of pgvector KBs and Neo4j/Neptune",
    )
    app.state.kg_service = svc

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy", "service": "enterprise-kg-ontology-fabric"}

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

    @app.post("/v1/rag/query", response_model=GraphRAGQueryResponse)
    async def query_rag(req: GraphRAGQueryRequest) -> GraphRAGQueryResponse:
        return await svc.query_graph_rag(req)

    @app.get("/v1/telemetry")
    async def get_telemetry(tenant_id: str = "enterprise_default") -> dict[str, Any]:
        return await svc.get_enterprise_telemetry(tenant_id=tenant_id)

    return app
