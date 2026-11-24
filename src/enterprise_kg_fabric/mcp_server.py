from __future__ import annotations

import json
import re
from typing import Any

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .auth import AuthEnforcer, get_current_principal
from .models import (
    AnswerabilityCheckRequest,
    ChunkSearchRequest,
    EntitySearchRequest,
    GlobalCommunitySearchRequest,
    GraphRAGQueryRequest,
    OntologyConceptReviewRequest,
    PPRSearchRequest,
    RawDocumentInput,
    ReadOnlyCypherSearchRequest,
    SubgraphSearchRequest,
    VectorKBRegistration,
)
from .service import EnterpriseKGOntologyService

try:
    _MCP_SERVER_VERSION = _pkg_version("enterprise-kg-ontology-fabric")
except PackageNotFoundError:
    _MCP_SERVER_VERSION = "0.3.0"


class KGFabricMcpServer:
    """
    Model Context Protocol (MCP `2025-11-25` / `2025-03-26` Streamable HTTP & JSON-RPC)
    Engine for `enterprise-kg-ontology-fabric`.
    Co-hosted directly inside FastAPI (`/mcp`, `/mcp/http`, `/v1/mcp/call`) and exported via
    `FastMCP` (`stdio` / `sse`) with 14 Tools, 4 Resources, and 2 Prompts.
    """

    SUPPORTED_VERSIONS = ["2025-11-25", "2025-03-26", "2024-11-05"]
    _WRITER_TOOLS = {
        "register_vector_kbs",
        "register_vector_kb",
        "ingest_raw_documents",
        "ingest_raw_document",
        "induce_and_align_ontology",
        "build_or_update_knowledge_graph",
    }
    _ADMIN_TOOLS = {
        "promote_quarantined_ontology",
        "review_ontology_concept",
    }
    _PLATFORM_ADMIN_TOOLS = {
        "execute_readonly_opencypher",
    }

    def __init__(self, service: EnterpriseKGOntologyService):
        self.service = service
        self._enforcer = AuthEnforcer()

    @staticmethod
    def list_tools_metadata() -> list[dict[str, Any]]:
        return [
            {
                "name": "query_hybrid_graph_rag",
                "title": "One-Shot 4-Path Hybrid GraphRAG Query",
                "description": (
                    "Executes the full 4-Path Hybrid GraphRAG pipeline (Multi-Model pgvector HNSW + "
                    "BM25 + K-Hop Neo4j/Neptune Graph Expansion + Text-to-OpenCypher) across all KBs."
                ),
                "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string", "default": "enterprise_default"},
                        "query": {"type": "string", "description": "Natural language question"},
                        "max_graph_hops": {"type": "integer", "default": 2},
                        "top_k_chunks": {"type": "integer", "default": 6},
                        "as_of_iso": {"type": "string", "description": "Optional ISO timestamp for bi-temporal time-travel"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_vector_chunks",
                "title": "Granular Multi-KB Vector + BM25 Chunk Search",
                "description": (
                    "Granular search primitive: queries heterogeneous pgvector KBs via "
                    "FederatedEmbeddingModelRouter + BM25 Reciprocal Rank Fusion without graph traversal."
                ),
                "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string", "default": "enterprise_default"},
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 6},
                        "kb_filter": {"type": "array", "items": {"type": "string"}},
                        "domain_filter": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_kg_entities",
                "title": "Granular Knowledge Graph Entity Lookup & Resolution",
                "description": (
                    "Granular search primitive: resolves an entity name, alias, or concept to canonical "
                    "Knowledge Graph nodes and lists all KBs where each entity appears."
                ),
                "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string", "default": "enterprise_default"},
                        "query": {"type": "string"},
                        "ontology_class_filter": {"type": "string"},
                        "top_k": {"type": "integer", "default": 8},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "traverse_entity_subgraph",
                "title": "Granular K-Hop Bi-Temporal Subgraph Traversal",
                "description": (
                    "Granular search primitive: expands 1..K hops in Neo4j/Neptune around specific "
                    "seed entities with optional predicate filters and bi-temporal `as_of_iso` time-travel."
                ),
                "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string", "default": "enterprise_default"},
                        "seed_entities": {"type": "array", "items": {"type": "string"}},
                        "max_hops": {"type": "integer", "default": 2},
                        "predicate_filter": {"type": "array", "items": {"type": "string"}},
                        "as_of_iso": {"type": "string"},
                    },
                    "required": ["seed_entities"],
                },
            },
            {
                "name": "execute_readonly_opencypher",
                "title": "Guarded Read-Only OpenCypher Query Execution",
                "description": (
                    "Granular search primitive: executes a read-only OpenCypher query on Neo4j/Neptune. "
                    "Mutating keywords (CREATE, MERGE, DELETE, SET, DROP) are strictly blocked."
                ),
                "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string", "default": "enterprise_default"},
                        "cypher_query": {"type": "string"},
                        "params": {"type": "object"},
                    },
                    "required": ["cypher_query"],
                },
            },
            {
                "name": "search_ppr_subgraph",
                "title": "HippoRAG 2 Personalized PageRank (PPR) Subgraph Retrieval",
                "description": (
                    "Hub-damped Personalized PageRank traversal seeded from specific entities, "
                    "penalizing noisy supernodes and enforcing ABAC ACLs."
                ),
                "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string", "default": "enterprise_default"},
                        "seed_entities": {"type": "array", "items": {"type": "string"}},
                        "top_k_facts": {"type": "integer", "default": 15},
                        "caller_acl_groups": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["seed_entities"],
                },
            },
            {
                "name": "query_global_communities",
                "title": "Microsoft GraphRAG Global Community Sensemaking",
                "description": (
                    "Answers corpus-wide executive questions across all KBs using Leiden/Louvain "
                    "hierarchical community detection and LightRAG high-level theme summaries."
                ),
                "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string", "default": "enterprise_default"},
                        "query": {"type": "string"},
                        "top_k_communities": {"type": "integer", "default": 4},
                        "caller_acl_groups": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "register_vector_kbs",
                "title": "Register Heterogeneous pgvector KBs",
                "description": "Registers one or more existing PostgreSQL pgvector knowledge bases.",
                "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "registrations": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["registrations"],
                },
            },
            {
                "name": "ingest_raw_documents",
                "title": "Ingest Raw Documents with Presidio PII Scrubbing",
                "description": "Scrubs PII via Microsoft Presidio, chunks hierarchically via LlamaIndex, and stores in pgvector.",
                "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "documents": {"type": "array", "items": {"type": "object"}},
                        "native_kb_dim": {"type": "integer", "default": 768},
                    },
                    "required": ["documents"],
                },
            },
            {
                "name": "induce_and_align_ontology",
                "title": "Automated Sub-Linear Vector-Cluster Ontology Induction",
                "description": "Samples cluster medoids & boundary outliers across KBs to induce and align the enterprise TBox.",
                "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string", "default": "enterprise_default"},
                        "tbox_version": {"type": "string", "default": "1.0.0"},
                    },
                },
            },
            {
                "name": "build_or_update_knowledge_graph",
                "title": "Incremental SHA-256 Watermark KG Build & SHACL Validation",
                "description": "Extracts delta chunks, resolves cross-KB entities, validates SHACL, and updates Neo4j/Neptune.",
                "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string", "default": "enterprise_default"},
                        "incremental": {"type": "boolean", "default": True},
                        "validate_shacl": {"type": "boolean", "default": True},
                    },
                },
            },
            {
                "name": "promote_quarantined_ontology",
                "title": "Promote Quarantined Candidate Ontology Concepts",
                "description": "Promotes corroborated candidate classes and predicates into the approved enterprise TBox.",
                "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string", "default": "enterprise_default"},
                        "min_confidence": {"type": "number", "default": 0.80},
                    },
                },
            },
            {
                "name": "check_answerable",
                "title": "Pre-Flight Ontology & Multi-KB Coverage Check",
                "description": (
                    "Evaluates whether the tenant's compiled TBox and authorized KB graph cover a user query "
                    "before generating an answer (returns ANSWERABLE, PARTIAL, or UNANSWERABLE)."
                ),
                "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "tenant_id": {"type": "string", "default": "enterprise_default"},
                        "caller_acl_groups": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "audit_kg_quality",
                "title": "Automated Ontology & Knowledge Graph Quality Scorecard",
                "description": (
                    "Audits the tenant's compiled TBox and Neo4j/Neptune ABox across SHACL conformance, "
                    "schema utilization, domain/range compliance, extraction faithfulness, entity resolution "
                    "duplicates/over-merges, orphan nodes, connected components, cross-KB linkage, and modularity."
                ),
                "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string", "default": "enterprise_default"},
                        "sample_size_faithfulness": {"type": "integer", "default": 25},
                        "use_llm_judge": {"type": "boolean", "default": False},
                    },
                },
            },
            {
                "name": "review_ontology_concept",
                "title": "Per-Concept Ontology Review (Approve / Reject / Merge)",
                "description": (
                    "Approves, rejects (deprecates), or merges a specific TBox class or relation, "
                    "updates Neo4j/Neptune entities/edges, bumps TBox version, and appends an immutable audit entry."
                ),
                "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string", "default": "enterprise_default"},
                        "concept_kind": {"type": "string", "enum": ["CLASS", "RELATION"]},
                        "concept_name": {"type": "string"},
                        "action": {"type": "string", "enum": ["approve", "reject", "merge"]},
                        "target_name": {"type": "string"},
                        "reason": {"type": "string", "default": ""},
                    },
                    "required": ["concept_kind", "concept_name", "action"],
                },
            },
        ]

    @staticmethod
    def list_resources_metadata() -> list[dict[str, Any]]:
        return [
            {
                "uri": "kg://fabric/health",
                "name": "fabric_health_and_capabilities",
                "mimeType": "application/json",
                "description": "Live health, PII engine status, and supported MCP capabilities",
            },
            {
                "uri": "kg://kbs/enterprise_default/registry",
                "name": "registered_vector_kbs_catalog",
                "mimeType": "application/json",
                "description": "Catalog of registered heterogeneous pgvector KBs and embedding dimensions",
            },
            {
                "uri": "kg://ontology/enterprise_default/tbox",
                "name": "tenant_compiled_tbox_json",
                "mimeType": "application/json",
                "description": "Compiled & aligned Enterprise Ontology (TBox classes, relations, synonyms)",
            },
            {
                "uri": "kg://ontology/enterprise_default/shapes.ttl",
                "name": "tenant_owl_shacl_turtle",
                "mimeType": "text/turtle",
                "description": "W3C OWL 2 / RDFS / SHACL Turtle (.ttl) ontology export",
            },
        ]

    @staticmethod
    def list_resource_templates_metadata() -> list[dict[str, Any]]:
        return [
            {
                "uriTemplate": "kg://kbs/{tenant_id}/registry",
                "name": "tenant_vector_kbs_registry",
                "mimeType": "application/json",
                "description": "List all registered pgvector KBs for any tenant_id",
            },
            {
                "uriTemplate": "kg://ontology/{tenant_id}/tbox",
                "name": "tenant_compiled_tbox",
                "mimeType": "application/json",
                "description": "Inspect the compiled TBox ontology for any tenant_id",
            },
            {
                "uriTemplate": "kg://ontology/{tenant_id}/shapes.ttl",
                "name": "tenant_owl_shacl_shapes",
                "mimeType": "text/turtle",
                "description": "Export W3C OWL 2 / SHACL Turtle (.ttl) for any tenant_id",
            },
        ]

    @staticmethod
    def list_prompts_metadata() -> list[dict[str, Any]]:
        return [
            {
                "name": "cross_kb_graphrag_investigator",
                "description": "Pre-populates 4-Path Hybrid GraphRAG context + Cross-KB bridges for multi-silo investigation",
                "arguments": [
                    {"name": "tenant_id", "required": False},
                    {"name": "query", "required": True},
                ],
            },
            {
                "name": "ontology_cypher_analyst",
                "description": "Injects the tenant's induced TBox classes/predicates to guide read-only OpenCypher generation",
                "arguments": [
                    {"name": "tenant_id", "required": False},
                    {"name": "question", "required": True},
                ],
            },
        ]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(arguments or {})
        principal = get_current_principal()
        if principal is not None:
            if tool_name in self._PLATFORM_ADMIN_TOOLS:
                self._enforcer.require_platform_admin(principal)
            elif tool_name in self._ADMIN_TOOLS:
                self._enforcer.require_role(principal, "admin")
            elif tool_name in self._WRITER_TOOLS:
                self._enforcer.require_role(principal, "writer")
            req_tenant = str(args.get("tenant_id") or principal.tenant_id or "enterprise_default")
            args["tenant_id"] = self._enforcer.enforce_tenant(principal, req_tenant)
            # Enforce tenant isolation recursively on nested registrations & documents
            if isinstance(args.get("registrations"), list):
                sanitized_regs = []
                for item in args["registrations"]:
                    if isinstance(item, dict):
                        it = dict(item)
                        it["tenant_id"] = self._enforcer.enforce_tenant(
                            principal, str(it.get("tenant_id") or args["tenant_id"])
                        )
                        sanitized_regs.append(it)
                args["registrations"] = sanitized_regs
            if isinstance(args.get("documents"), list):
                sanitized_docs = []
                for item in args["documents"]:
                    if isinstance(item, dict):
                        it = dict(item)
                        it["tenant_id"] = self._enforcer.enforce_tenant(
                            principal, str(it.get("tenant_id") or args["tenant_id"])
                        )
                        sanitized_docs.append(it)
                args["documents"] = sanitized_docs
            if "caller_acl_groups" in args or tool_name in {
                "query_hybrid_graph_rag",
                "query_graph_rag",
                "check_answerable",
                "search_vector_chunks",
                "search_kg_entities",
                "traverse_entity_subgraph",
                "search_ppr_subgraph",
                "query_global_communities",
            }:
                raw_acls = args.get("caller_acl_groups")
                args["caller_acl_groups"] = self._enforcer.enforce_acl_groups(
                    principal, list(raw_acls) if isinstance(raw_acls, list) else None
                )

        if tool_name in {"query_hybrid_graph_rag", "query_graph_rag"}:
            res = await self.service.query_graph_rag(GraphRAGQueryRequest(**args))
            return res.model_dump()

        if tool_name == "check_answerable":
            res = await self.service.check_answerable(AnswerabilityCheckRequest(**args))
            return res.model_dump()

        if tool_name == "audit_kg_quality":
            report = await self.service.audit_kg_quality(
                tenant_id=args.get("tenant_id", "enterprise_default"),
                sample_size_faithfulness=int(args.get("sample_size_faithfulness", 25)),
                use_llm_judge=bool(args.get("use_llm_judge", False)),
            )
            return report.model_dump()

        if tool_name == "search_vector_chunks":
            chunks = await self.service.search_vector_chunks(ChunkSearchRequest(**args))
            return {"chunks": [c.model_dump() for c in chunks], "count": len(chunks)}

        if tool_name == "search_kg_entities":
            ents = await self.service.search_kg_entities(EntitySearchRequest(**args))
            return {"entities": [e.model_dump() for e in ents], "count": len(ents)}

        if tool_name == "traverse_entity_subgraph":
            sub = await self.service.traverse_entity_subgraph(SubgraphSearchRequest(**args))
            return sub.model_dump()

        if tool_name == "execute_readonly_opencypher":
            c_res = await self.service.execute_readonly_opencypher(ReadOnlyCypherSearchRequest(**args))
            return c_res.model_dump()

        if tool_name == "search_ppr_subgraph":
            p_res = await self.service.search_ppr_subgraph(PPRSearchRequest(**args))
            return p_res.model_dump()

        if tool_name == "query_global_communities":
            g_res = await self.service.query_global_communities(GlobalCommunitySearchRequest(**args))
            return g_res.model_dump()

        if tool_name == "register_vector_kbs":
            regs = [VectorKBRegistration.model_validate(r) for r in args.get("registrations", [])]
            cnt = await self.service.register_vector_kbs_batch(regs)
            return {"registered_count": cnt}

        if tool_name == "ingest_raw_documents":
            docs = [RawDocumentInput.model_validate(d) for d in args.get("documents", [])]
            dim = int(args.get("native_kb_dim", 768))
            created = await self.service.ingest_raw_documents(docs, native_kb_dim=dim)
            return {"documents_ingested": len(docs), "chunks_created": len(created)}

        if tool_name == "induce_and_align_ontology":
            tenant_id = args.get("tenant_id", "enterprise_default")
            version = args.get("tbox_version", "1.0.0")
            _, report = await self.service.induce_and_align_ontology(
                tenant_id=tenant_id, tbox_version=version
            )
            return report.model_dump()

        if tool_name == "build_or_update_knowledge_graph":
            report = await self.service.build_or_update_knowledge_graph(
                tenant_id=args.get("tenant_id", "enterprise_default"),
                incremental=bool(args.get("incremental", True)),
                validate_shacl=bool(args.get("validate_shacl", True)),
            )
            return report.model_dump()

        if tool_name == "promote_quarantined_ontology":
            return await self.service.promote_quarantined_ontology_concepts(
                tenant_id=args.get("tenant_id", "enterprise_default"),
                min_confidence=float(args.get("min_confidence", 0.80)),
            )

        if tool_name == "review_ontology_concept":
            return await self.service.review_ontology_concept(
                OntologyConceptReviewRequest(**args)
            )

        raise ValueError(f"Unknown MCP tool: {tool_name}")

    async def read_resource(self, uri: str) -> dict[str, Any]:
        if uri == "kg://fabric/health":
            return {
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(
                    {
                        "status": "healthy",
                        "service": "enterprise-kg-ontology-fabric",
                        "pii_engine": self.service.pii_redactor.engine_name,
                        "mcp_tools_count": len(self.list_tools_metadata()),
                    }
                ),
            }

        m_reg = re.match(r"^kg://kbs/([^/]+)/registry$", uri)
        if m_reg:
            tenant_id = self._enforcer.enforce_tenant(get_current_principal(), m_reg.group(1))
            kbs = await self.service.list_registered_kbs_paginated(tenant_id=tenant_id, limit=500)
            return {
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps([k.model_dump() for k in kbs]),
            }

        m_tbox = re.match(r"^kg://ontology/([^/]+)/tbox$", uri)
        if m_tbox:
            tenant_id = self._enforcer.enforce_tenant(get_current_principal(), m_tbox.group(1))
            tbox = await self.service.load_or_get_tbox(tenant_id)
            return {
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(tbox.model_dump()),
            }

        m_ttl = re.match(r"^kg://ontology/([^/]+)/shapes\.ttl$", uri)
        if m_ttl:
            tenant_id = self._enforcer.enforce_tenant(get_current_principal(), m_ttl.group(1))
            await self.service.load_or_get_tbox(tenant_id)
            return {
                "uri": uri,
                "mimeType": "text/turtle",
                "text": self.service.export_ontology_turtle(tenant_id),
            }

        raise ValueError(f"Unknown MCP resource URI: {uri}")

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = arguments or {}
        tenant_id = self._enforcer.enforce_tenant(
            get_current_principal(), args.get("tenant_id", "enterprise_default")
        )

        if name == "cross_kb_graphrag_investigator":
            query = args.get("query", "")
            rag = await self.service.query_graph_rag(
                GraphRAGQueryRequest(tenant_id=tenant_id, query=query, max_graph_hops=3)
            )
            return {
                "description": "Cross-KB Hybrid GraphRAG context with multi-hop lineage",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": (
                                f"You are an Enterprise Cross-KB GraphRAG Investigator.\n"
                                f"Question: {query}\n\n"
                                f"Cross-KB Bridge Entities: {rag.cross_kb_bridge_entities}\n"
                                f"Contributing KBs: {rag.provenance_kbs_contributing}\n\n"
                                f"Retrieved Evidence:\n{rag.synthesized_answer}"
                            ),
                        },
                    }
                ],
            }

        if name == "ontology_cypher_analyst":
            question = args.get("question", "")
            tbox = self.service.get_tbox(tenant_id)
            classes_list = list(tbox.classes.keys())
            rels_list = list(tbox.relations.keys())
            return {
                "description": "Ontology-grounded OpenCypher query synthesis prompt",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": (
                                f"Write a read-only OpenCypher query for tenant '{tenant_id}' using "
                                f"Labels `:Entity` and Relationship `:KG_RELATION` (`r.predicate`, `r.t_invalid = 'ACTIVE'`).\n"
                                f"Approved Ontology Classes: {classes_list}\n"
                                f"Approved Ontology Predicates: {rels_list}\n"
                                f"User Question: {question}"
                            ),
                        },
                    }
                ],
            }

        raise ValueError(f"Unknown MCP prompt: {name}")

    async def handle_jsonrpc(
        self,
        payload: dict[str, Any],
        protocol_version_header: str | None = None,
    ) -> dict[str, Any]:
        req_id = payload.get("id")
        method = payload.get("method", "")
        params = payload.get("params") or {}
        negotiated_version = (
            protocol_version_header
            if protocol_version_header in self.SUPPORTED_VERSIONS
            else "2025-11-25"
        )

        try:
            if method == "initialize":
                client_ver = params.get("protocolVersion")
                ver = client_ver if client_ver in self.SUPPORTED_VERSIONS else negotiated_version
                result = {
                    "protocolVersion": ver,
                    "serverInfo": {
                        "name": "enterprise-kg-ontology-fabric",
                        "version": _MCP_SERVER_VERSION,
                    },
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                        "prompts": {"listChanged": False},
                    },
                }
            elif method in ("ping", "notifications/initialized"):
                result = {}
            elif method == "tools/list":
                result = {"tools": self.list_tools_metadata()}
            elif method == "tools/call":
                t_name = params.get("name", "")
                t_args = params.get("arguments") or {}
                structured = await self.call_tool(t_name, t_args)
                result = {
                    "content": [{"type": "text", "text": json.dumps(structured)}],
                    "structuredContent": structured,
                    "isError": False,
                }
            elif method == "resources/list":
                result = {"resources": self.list_resources_metadata()}
            elif method == "resources/templates/list":
                result = {"resourceTemplates": self.list_resource_templates_metadata()}
            elif method == "resources/read":
                uri = params.get("uri", "")
                contents = await self.read_resource(uri)
                result = {"contents": [contents]}
            elif method == "prompts/list":
                result = {"prompts": self.list_prompts_metadata()}
            elif method == "prompts/get":
                p_name = params.get("name", "")
                p_args = params.get("arguments") or {}
                result = await self.get_prompt(p_name, p_args)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": str(exc)},
            }


def create_fastmcp_server(service: EnterpriseKGOntologyService | None = None) -> Any:
    """
    Creates an official MCP SDK server instance (`MCPServer` on `mcp>=2.0`, `FastMCP` on `mcp 1.x`)
    for `stdio` or `sse` transport (compatible with Claude Desktop, Cursor, Windsurf, and Google Antigravity).
    """
    try:
        from mcp.server.mcpserver import MCPServer as _MCPServerCls
    except ImportError:
        from mcp.server.fastmcp import FastMCP as _MCPServerCls  # type: ignore[no-redef]

    svc = service or EnterpriseKGOntologyService()
    mcp_engine = KGFabricMcpServer(svc)
    fastmcp = _MCPServerCls("enterprise-kg-ontology-fabric")

    @fastmcp.tool()
    async def query_hybrid_graph_rag(
        query: str,
        tenant_id: str = "enterprise_default",
        max_graph_hops: int = 2,
        top_k_chunks: int = 6,
        as_of_iso: str | None = None,
    ) -> dict[str, Any]:
        """Executes the 4-Path Hybrid GraphRAG pipeline across all pgvector KBs and Neo4j/Neptune."""
        return await mcp_engine.call_tool(
            "query_hybrid_graph_rag",
            {
                "tenant_id": tenant_id,
                "query": query,
                "max_graph_hops": max_graph_hops,
                "top_k_chunks": top_k_chunks,
                "as_of_iso": as_of_iso,
            },
        )

    @fastmcp.tool()
    async def search_vector_chunks(
        query: str,
        tenant_id: str = "enterprise_default",
        top_k: int = 6,
    ) -> dict[str, Any]:
        """Granular Multi-KB pgvector + BM25 RRF chunk search."""
        return await mcp_engine.call_tool(
            "search_vector_chunks", {"tenant_id": tenant_id, "query": query, "top_k": top_k}
        )

    @fastmcp.tool()
    async def search_kg_entities(
        query: str,
        tenant_id: str = "enterprise_default",
        ontology_class_filter: str | None = None,
        top_k: int = 8,
    ) -> dict[str, Any]:
        """Granular Knowledge Graph entity resolution & cross-KB provenance search."""
        return await mcp_engine.call_tool(
            "search_kg_entities",
            {
                "tenant_id": tenant_id,
                "query": query,
                "ontology_class_filter": ontology_class_filter,
                "top_k": top_k,
            },
        )

    @fastmcp.tool()
    async def traverse_entity_subgraph(
        seed_entities: list[str],
        tenant_id: str = "enterprise_default",
        max_hops: int = 2,
        as_of_iso: str | None = None,
    ) -> dict[str, Any]:
        """Granular K-hop bi-temporal subgraph expansion around specific entities."""
        return await mcp_engine.call_tool(
            "traverse_entity_subgraph",
            {
                "tenant_id": tenant_id,
                "seed_entities": seed_entities,
                "max_hops": max_hops,
                "as_of_iso": as_of_iso,
            },
        )

    @fastmcp.tool()
    async def execute_readonly_opencypher(
        cypher_query: str,
        tenant_id: str = "enterprise_default",
    ) -> dict[str, Any]:
        """Guarded read-only OpenCypher execution against Neo4j or Amazon Neptune."""
        return await mcp_engine.call_tool(
            "execute_readonly_opencypher",
            {"tenant_id": tenant_id, "cypher_query": cypher_query},
        )

    @fastmcp.tool()
    async def search_ppr_subgraph(
        seed_entities: list[str],
        tenant_id: str = "enterprise_default",
        top_k_facts: int = 15,
        alpha: float = 0.85,
        as_of_iso: str | None = None,
    ) -> dict[str, Any]:
        """HippoRAG 2 Personalized PageRank hub-damped subgraph retrieval."""
        return await mcp_engine.call_tool(
            "search_ppr_subgraph",
            {
                "tenant_id": tenant_id,
                "seed_entities": seed_entities,
                "top_k_facts": top_k_facts,
                "alpha": alpha,
                "as_of_iso": as_of_iso,
            },
        )

    @fastmcp.tool()
    async def query_global_communities(
        query: str,
        tenant_id: str = "enterprise_default",
        top_k_communities: int = 4,
    ) -> dict[str, Any]:
        """Microsoft GraphRAG Leiden/Louvain Global Sensemaking over persisted communities."""
        return await mcp_engine.call_tool(
            "query_global_communities",
            {
                "tenant_id": tenant_id,
                "query": query,
                "top_k_communities": top_k_communities,
            },
        )

    @fastmcp.tool()
    async def check_answerable(
        query: str,
        tenant_id: str = "enterprise_default",
    ) -> dict[str, Any]:
        """Pre-flight ontology & multi-KB coverage check (ANSWERABLE | PARTIAL | UNANSWERABLE)."""
        return await mcp_engine.call_tool(
            "check_answerable", {"tenant_id": tenant_id, "query": query}
        )

    @fastmcp.tool()
    async def audit_kg_quality(
        tenant_id: str = "enterprise_default",
        sample_size_faithfulness: int = 25,
        use_llm_judge: bool = False,
    ) -> dict[str, Any]:
        """Runs the 4-dimension Ontology & Knowledge Graph Quality Scorecard."""
        return await mcp_engine.call_tool(
            "audit_kg_quality",
            {
                "tenant_id": tenant_id,
                "sample_size_faithfulness": sample_size_faithfulness,
                "use_llm_judge": use_llm_judge,
            },
        )

    @fastmcp.tool()
    async def register_vector_kbs(registrations: list[dict[str, Any]]) -> dict[str, Any]:
        """Registers one or more vector KBs in the tenant federation registry."""
        return await mcp_engine.call_tool(
            "register_vector_kbs", {"registrations": registrations}
        )

    @fastmcp.tool()
    async def ingest_raw_documents(
        documents: list[dict[str, Any]],
        native_kb_dim: int = 768,
    ) -> dict[str, Any]:
        """Ingests raw enterprise documents with PII scrubbing and hierarchical chunking."""
        return await mcp_engine.call_tool(
            "ingest_raw_documents",
            {"documents": documents, "native_kb_dim": native_kb_dim},
        )

    @fastmcp.tool()
    async def induce_and_align_ontology(
        tenant_id: str = "enterprise_default",
        tbox_version: str = "1.0.0",
    ) -> dict[str, Any]:
        """Induces and aligns the tenant's unified TBox ontology across registered KBs."""
        return await mcp_engine.call_tool(
            "induce_and_align_ontology",
            {"tenant_id": tenant_id, "tbox_version": tbox_version},
        )

    @fastmcp.tool()
    async def build_or_update_knowledge_graph(
        tenant_id: str = "enterprise_default",
        incremental: bool = True,
        validate_shacl: bool = True,
    ) -> dict[str, Any]:
        """Builds or incrementally updates the tenant's Knowledge Graph and community cache."""
        return await mcp_engine.call_tool(
            "build_or_update_knowledge_graph",
            {
                "tenant_id": tenant_id,
                "incremental": incremental,
                "validate_shacl": validate_shacl,
            },
        )

    @fastmcp.tool()
    async def promote_quarantined_ontology(
        tenant_id: str = "enterprise_default",
        min_confidence: float = 0.80,
    ) -> dict[str, Any]:
        """Promotes high-confidence quarantined ontology concepts into APPROVED status."""
        return await mcp_engine.call_tool(
            "promote_quarantined_ontology",
            {"tenant_id": tenant_id, "min_confidence": min_confidence},
        )

    @fastmcp.tool()
    async def review_ontology_concept(
        concept_kind: str,
        concept_name: str,
        action: str,
        tenant_id: str = "enterprise_default",
        target_name: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Per-concept ontology review: approve, reject, or merge a specific class/relation with audit log."""
        return await mcp_engine.call_tool(
            "review_ontology_concept",
            {
                "tenant_id": tenant_id,
                "concept_kind": concept_kind,
                "concept_name": concept_name,
                "action": action,
                "target_name": target_name,
                "reason": reason,
            },
        )

    return fastmcp
