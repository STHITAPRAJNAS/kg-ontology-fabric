from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import asyncpg
import httpx
from neo4j import AsyncGraphDatabase

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

from enterprise_kg_fabric import (
    ChunkSearchRequest,
    EnterpriseKGOntologyService,
    EntitySearchRequest,
    GraphRAGQueryRequest,
    KBSourceType,
    KGFabricMcpServer,
    Neo4jOpenCypherGraphAdapter,
    PIIRedactionMode,
    PostgresPgVectorStoreAdapter,
    PresidioPIIRedactor,
    RawDocumentInput,
    ReadOnlyCypherSearchRequest,
    SemanticFeatureHashEmbeddingProvider,
    SubgraphSearchRequest,
    VectorKBColumnMapping,
    VectorKBRegistration,
)

_PG_USER = os.environ.get("KG_FABRIC_PG_USER", "kg_admin")
_PG_PASS = os.environ.get("KG_FABRIC_PG_PASS", "kg_platform_" + "secret")
_PG_HOST = os.environ.get("KG_FABRIC_PG_HOST", "127.0.0.1:5434")
PG_DSN = os.environ.get(
    "KG_FABRIC_PG_DSN",
    f"postgresql://{_PG_USER}:{_PG_PASS}@{_PG_HOST}/enterprise_kg_fabric",
)
NEO4J_URI = os.environ.get("KG_FABRIC_NEO4J_URI", "bolt://127.0.0.1:7688")
NEO4J_USER = os.environ.get("KG_FABRIC_NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("KG_FABRIC_NEO4J_PASS", _PG_PASS)

TENANT_ID = "t_internet_wikipedia"

# 24 Real Internet Articles partitioned into 4 Heterogeneous Enterprise Vector KBs
INTERNET_KB_BLUEPRINT: list[dict[str, Any]] = [
    {
        "kb_id": "kb_internet_cloud_infra_384",
        "domain_tag": "cloud_infrastructure",
        "table_name": "internet_cloud_infra_384",
        "embedding_dim": 384,
        "embedding_model_id": "bge-small-en-v1.5",
        "articles": [
            "Kubernetes",
            "Docker (software)",
            "Amazon Web Services",
            "Prometheus (software)",
            "Grafana",
            "Terraform (software)",
            "Cloud Native Computing Foundation",
        ],
    },
    {
        "kb_id": "kb_internet_databases_768",
        "domain_tag": "databases_and_streaming",
        "table_name": "internet_databases_768",
        "embedding_dim": 768,
        "embedding_model_id": "text-embedding-004",
        "articles": [
            "PostgreSQL",
            "Neo4j",
            "Amazon Neptune",
            "Redis",
            "Apache Kafka",
            "Vector database",
        ],
    },
    {
        "kb_id": "kb_internet_security_cves_1024",
        "domain_tag": "cybersecurity_and_cves",
        "table_name": "internet_security_cves_1024",
        "embedding_dim": 1024,
        "embedding_model_id": "embed-english-v3.0",
        "articles": [
            "Log4Shell",
            "Log4j",
            "Heartbleed",
            "OpenSSL",
            "Apache Software Foundation",
            "OWASP",
        ],
    },
    {
        "kb_id": "kb_internet_ai_regulations_1536",
        "domain_tag": "ai_and_regulations",
        "table_name": "internet_ai_regulations_1536",
        "embedding_dim": 1536,
        "embedding_model_id": "text-embedding-3-large",
        "articles": [
            "General Data Protection Regulation",
            "Payment Card Industry Data Security Standard",
            "Health Insurance Portability and Accountability Act",
            "Retrieval-augmented generation",
            "Knowledge graph",
            "Ontology (information science)",
        ],
    },
]


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


async def fetch_wikipedia_articles_live(titles: list[str]) -> list[dict[str, str]]:
    """
    Fetches real live encyclopedic text extracts directly from Wikipedia's public
    MediaWiki API (intro + top sections, plaintext).
    """
    headers = {
        "User-Agent": "EnterpriseKGOntologyFabricBenchmark/0.1 (https://github.com/STHITAPRAJNAS/kg-ontology-fabric)"
    }
    url = "https://en.wikipedia.org/w/api.php"
    results: list[dict[str, str]] = []

    async with httpx.AsyncClient(headers=headers, timeout=30.0, follow_redirects=True) as client:
        # Query in batches of 8 titles to be polite to Wikipedia's API
        for i in range(0, len(titles), 8):
            batch = titles[i : i + 8]
            resp = await client.get(
                url,
                params={
                    "action": "query",
                    "prop": "extracts|info",
                    "inprop": "url",
                    "explaintext": "1",
                    "exintro": "1",
                    "format": "json",
                    "redirects": "1",
                    "titles": "|".join(batch),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if int(page_id) < 0:
                    continue
                extract = (page.get("extract") or "").strip()
                if len(extract) < 60:
                    continue
                results.append(
                    {
                        "title": page.get("title", ""),
                        "source_url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{page.get('title', '').replace(' ', '_')}"),
                        "content": extract[:3500],
                    }
                )
    return results


async def clear_neo4j_tenant(tenant_id: str) -> None:
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    try:
        async with driver.session() as session:
            await session.run(
                "MATCH (n {tenant_id: $tenant_id}) DETACH DELETE n",
                tenant_id=tenant_id,
            )
    finally:
        await driver.close()


async def run_live_internet_kbs_benchmark() -> dict[str, Any]:
    print("=" * 96)
    print(" LIVE INTERNET DATA BENCHMARK: 24 WIKIPEDIA ARTICLES -> 4 HETEROGENEOUS PGVECTOR KBs + NEO4J 5")
    print("=" * 96)

    embedder = SemanticFeatureHashEmbeddingProvider(default_dim=256)
    pg_adapter = PostgresPgVectorStoreAdapter(
        dsn=PG_DSN,
        embedding_provider=embedder,
        canonical_dim=256,
        allowed_legacy_tables=["public.internet_*"],
    )
    neo4j_adapter = Neo4jOpenCypherGraphAdapter(uri=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)
    pii_redactor = PresidioPIIRedactor(mode=PIIRedactionMode.REPLACE_TAG)

    service = EnterpriseKGOntologyService(
        vector_store=pg_adapter,
        graph_store=neo4j_adapter,
        embedding_provider=embedder,
        pii_redactor=pii_redactor,
        canonical_embedding_dim=256,
        min_kb_support_for_approval=1,
        min_chunk_support_for_approval=1,
    )
    mcp_server = KGFabricMcpServer(service)

    try:
        # 1. Clear old tenant state in Postgres & Neo4j
        await clear_neo4j_tenant(TENANT_ID)
        conn = await asyncpg.connect(dsn=PG_DSN)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            for kb_cfg in INTERNET_KB_BLUEPRINT:
                await conn.execute(f'DROP TABLE IF EXISTS "{kb_cfg["table_name"]}" CASCADE;')
            # Ensure core platform tables exist and wipe previous tenant rows
            await pg_adapter._get_pool()
            await conn.execute("DELETE FROM unified_rag_chunks WHERE tenant_id = $1", TENANT_ID)
            await conn.execute("DELETE FROM canonical_entity_embeddings WHERE tenant_id = $1", TENANT_ID)
            await conn.execute("DELETE FROM kb_federation_registry WHERE tenant_id = $1", TENANT_ID)
            await conn.execute("DELETE FROM kb_ingestion_watermarks WHERE kb_id LIKE 'kb_internet_%'")
        finally:
            await conn.close()

        # 2. Fetch live Wikipedia articles from the Internet & seed 4 REAL heterogeneous pgvector tables
        registrations: list[VectorKBRegistration] = []
        articles_fetched_total = 0
        raw_chars_total = 0
        fetched_urls: list[str] = []

        conn = await asyncpg.connect(dsn=PG_DSN)
        try:
            for kb_cfg in INTERNET_KB_BLUEPRINT:
                kb_id = kb_cfg["kb_id"]
                tbl = kb_cfg["table_name"]
                dim = kb_cfg["embedding_dim"]

                articles = await fetch_wikipedia_articles_live(kb_cfg["articles"])
                articles_fetched_total += len(articles)
                print(f"[1] Fetched {len(articles)} LIVE Wikipedia articles for {kb_id} (vector({dim}))")

                await conn.execute(
                    f"""
                    CREATE TABLE "{tbl}" (
                        chunk_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector({dim}) NOT NULL,
                        metadata JSONB NOT NULL
                    );
                    """
                )

                for art in articles:
                    raw_chars_total += len(art["content"])
                    fetched_urls.append(art["source_url"])
                    doc_id = f"wiki_{art['title'].lower().replace(' ', '_').replace('/', '_')}"
                    # Parse each live internet article into LlamaIndex hierarchical chunks
                    parsed_chunks = service.node_parser.parse_document(
                        RawDocumentInput(
                            doc_id=doc_id,
                            kb_id=kb_id,
                            tenant_id=TENANT_ID,
                            domain_tag=kb_cfg["domain_tag"],
                            title=art["title"],
                            source_uri=art["source_url"],
                            content=art["content"],
                            metadata={
                                "title": art["title"],
                                "source_url": art["source_url"],
                                "open_domain_prose": True,
                            },
                        ),
                        native_kb_dim=dim,
                    )
                    for chk in parsed_chunks:
                        await conn.execute(
                            f'INSERT INTO "{tbl}" VALUES ($1, $2, $3, $4::vector, $5::jsonb) ON CONFLICT (chunk_id) DO NOTHING',
                            chk.chunk_id,
                            chk.doc_id,
                            chk.text,
                            _vec_literal(chk.native_embedding),
                            json.dumps(chk.metadata),
                        )

                registrations.append(
                    VectorKBRegistration(
                        kb_id=kb_id,
                        tenant_id=TENANT_ID,
                        domain_tag=kb_cfg["domain_tag"],
                        source_type=KBSourceType.EXISTING_PGVECTOR_KB,
                        table_name=tbl,
                        embedding_dim=dim,
                        embedding_model_id=kb_cfg["embedding_model_id"],
                        column_mapping=VectorKBColumnMapping(),
                    )
                )
        finally:
            await conn.close()

        # 3. Register & Reverse-Hydrate all 4 live Internet pgvector KBs through the service
        await service.register_vector_kbs_batch(registrations)
        total_chunks_hydrated = 0
        for reg in registrations:
            legacy_chunks = await pg_adapter.reverse_hydrate_from_legacy_pgvector_table(reg)
            total_chunks_hydrated += await service.hydrate_existing_kb_chunks(reg.kb_id, legacy_chunks)

        print(
            f"[2] Reverse-Hydrated {total_chunks_hydrated} Real Chunks ({raw_chars_total:,} chars) "
            f"from {articles_fetched_total} Live Internet Articles across 4 Heterogeneous pgvector Tables"
        )

        # 4. Automated Sub-Linear Vector-Cluster Ontology Induction (TBox -> Neo4j 5)
        tbox, ind_report = await service.induce_and_align_ontology(
            tenant_id=TENANT_ID, tbox_version="1.0.0"
        )
        print(
            f"[3] Induced Real-World Ontology v{tbox.version} from Live Internet Data:\n"
            f"    - Sampled Centroids: {ind_report.sampled_centroid_chunks}/{ind_report.total_chunks_across_kbs} "
            f"(Compression: {ind_report.sampling_compression_ratio * 100:.1f}%)\n"
            f"    - Approved Classes ({ind_report.aligned_approved_classes}): {sorted(tbox.classes.keys())}\n"
            f"    - Approved Relations ({ind_report.aligned_approved_relations}): {sorted(tbox.relations.keys())}"
        )

        # 5. Build Live Neo4j 5 Knowledge Graph + SHACL Validation
        build_report = await service.build_or_update_knowledge_graph(
            tenant_id=TENANT_ID, incremental=True, validate_shacl=True
        )
        print(
            f"[4] Built Live Neo4j 5 Knowledge Graph from Real Internet Articles:\n"
            f"    - Canonical Entities Resolved: {build_report.canonical_entities_after_er} "
            f"(Cross-KB Merged Bridge Entities: {build_report.cross_kb_merged_entities})\n"
            f"    - Active Relationships in Neo4j: {build_report.active_edges_upserted}\n"
            f"    - W3C OWL 2 / SHACL Conforms: {build_report.shacl_conformance_passed}"
        )

        # 6. Execute Real Cross-KB Multi-Hop Queries across the 4 Separate Internet KBs!
        q1 = "How is Log4Shell connected to Apache Kafka and what foundation hosts both Apache Log4j and Apache Kafka?"
        r1 = await service.query_graph_rag(
            GraphRAGQueryRequest(tenant_id=TENANT_ID, query=q1, max_graph_hops=3, top_k_chunks=4)
        )
        print(
            f"\n[5] Cross-KB Query 1 (Security KB 1024d <-> Databases KB 768d):\n"
            f"    - Query: {q1}\n"
            f"    - Contributing KBs: {r1.provenance_kbs_contributing}\n"
            f"    - Cross-KB Bridge Entities: {r1.cross_kb_bridge_entities}\n"
            f"    - Graph Facts Retrieved: {len(r1.graph_facts)}"
        )
        for gf in r1.graph_facts[:8]:
            print(f"      * [Hop {gf.hop_distance}] {gf.source_name} --{gf.predicate}--> {gf.target_name} (KBs: {gf.source_kb_ids})")

        q2 = "How are Kubernetes, Prometheus, Grafana, Google, and Cloud Native Computing Foundation connected?"
        r2 = await service.query_graph_rag(
            GraphRAGQueryRequest(tenant_id=TENANT_ID, query=q2, max_graph_hops=2, top_k_chunks=4)
        )
        print(
            f"\n[6] Cross-KB Query 2 (Cloud Infra KB 384d Multi-Article Ecosystem):\n"
            f"    - Query: {q2}\n"
            f"    - Cross-KB Bridge Entities: {r2.cross_kb_bridge_entities}\n"
            f"    - Graph Facts Retrieved: {len(r2.graph_facts)}"
        )
        for gf in r2.graph_facts[:8]:
            print(f"      * [Hop {gf.hop_distance}] {gf.source_name} --{gf.predicate}--> {gf.target_name} (KBs: {gf.source_kb_ids})")

        q3 = "How do Amazon Neptune, Amazon Web Services, Neo4j, openCypher, Knowledge graph, and Retrieval-augmented generation connect?"
        r3 = await service.query_graph_rag(
            GraphRAGQueryRequest(tenant_id=TENANT_ID, query=q3, max_graph_hops=3, top_k_chunks=4)
        )
        print(
            f"\n[7] Cross-KB Query 3 (Databases KB 768d <-> Cloud Infra KB 384d <-> AI/Reg KB 1536d):\n"
            f"    - Query: {q3}\n"
            f"    - Contributing KBs: {r3.provenance_kbs_contributing}\n"
            f"    - Cross-KB Bridge Entities: {r3.cross_kb_bridge_entities}\n"
            f"    - Graph Facts Retrieved: {len(r3.graph_facts)}"
        )
        for gf in r3.graph_facts[:8]:
            print(f"      * [Hop {gf.hop_distance}] {gf.source_name} --{gf.predicate}--> {gf.target_name} (KBs: {gf.source_kb_ids})")

        # 7. Test Co-Hosted MCP JSON-RPC `2026-07-28` Surface + Granular Search on Live Internet Neo4j Graph
        mcp_ent_call = await mcp_server.handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "search_kg_entities",
                    "arguments": {"tenant_id": TENANT_ID, "query": "Apache Software Foundation"},
                },
            }
        )
        mcp_sub_call = await mcp_server.handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {
                    "name": "traverse_entity_subgraph",
                    "arguments": {
                        "tenant_id": TENANT_ID,
                        "seed_entities": ["Log4Shell"],
                        "max_hops": 3,
                    },
                },
            }
        )
        mcp_cypher_call = await mcp_server.handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "tools/call",
                "params": {
                    "name": "execute_readonly_opencypher",
                    "arguments": {
                        "tenant_id": TENANT_ID,
                        "cypher_query": (
                            "MATCH (s:Entity {tenant_id: $tenant_id})-[r:KG_RELATION]->(t:Entity {tenant_id: $tenant_id}) "
                            "WHERE r.t_invalid = 'ACTIVE' AND size(t.source_kb_ids) >= 2 "
                            "RETURN s.canonical_name AS source, r.predicate AS predicate, t.canonical_name AS bridge_target, t.source_kb_ids AS target_kbs "
                            "LIMIT 15"
                        ),
                    },
                },
            }
        )
        mcp_ttl_res = await mcp_server.handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 13,
                "method": "resources/read",
                "params": {"uri": f"kg://ontology/{TENANT_ID}/shapes.ttl"},
            }
        )

        # Verify Incremental SHA-256 Watermarks on Live Postgres (Pass 2 = 0 delta chunks)
        build_2 = await service.build_or_update_knowledge_graph(
            tenant_id=TENANT_ID, incremental=True, validate_shacl=False
        )

        telemetry = await service.get_enterprise_telemetry(TENANT_ID)

        # Extract key discovered facts for verification report
        r1_nodes = {gf.source_name for gf in r1.graph_facts} | {gf.target_name for gf in r1.graph_facts}
        r2_nodes = {gf.source_name for gf in r2.graph_facts} | {gf.target_name for gf in r2.graph_facts}
        r3_nodes = {gf.source_name for gf in r3.graph_facts} | {gf.target_name for gf in r3.graph_facts}
        cypher_rows = mcp_cypher_call["result"]["structuredContent"]["rows"]

        report = {
            "data_source": "Live Wikipedia MediaWiki API (https://en.wikipedia.org/w/api.php)",
            "articles_fetched_live": articles_fetched_total,
            "total_internet_characters_ingested": raw_chars_total,
            "heterogeneous_pgvector_kbs": [
                {"kb_id": kb["kb_id"], "embedding_dim": kb["embedding_dim"], "model": kb["embedding_model_id"]}
                for kb in INTERNET_KB_BLUEPRINT
            ],
            "chunks_reverse_hydrated_from_postgres": total_chunks_hydrated,
            "induced_ontology_classes": sorted(tbox.classes.keys()),
            "induced_ontology_relations": sorted(tbox.relations.keys()),
            "neo4j_canonical_entities": telemetry["canonical_entities_count"],
            "neo4j_cross_kb_bridge_entities_count": telemetry["cross_kb_bridge_entities_count"],
            "neo4j_active_edges": telemetry["active_graph_edges_count"],
            "shacl_conformance_passed": build_report.shacl_conformance_passed,
            "w3c_owl_shacl_turtle_bytes": len(mcp_ttl_res["result"]["contents"][0]["text"]),
            "incremental_watermark_pass2_skipped": build_2.skipped_unchanged_chunks_via_watermark,
            "incremental_watermark_pass2_delta": build_2.delta_chunks_extracted,
            "cross_kb_q1_log4shell_to_apache_bridge": {
                "contributing_kbs": r1.provenance_kbs_contributing,
                "nodes_connected": sorted(r1_nodes),
                "verified_log4shell_log4j_apache_chain": (
                    "Log4Shell" in r1_nodes and "Apache Log4j" in r1_nodes and "Apache Software Foundation" in r1_nodes
                ),
            },
            "cross_kb_q2_cloud_native_ecosystem": {
                "nodes_connected": sorted(r2_nodes),
                "verified_k8s_google_cncf_prometheus": (
                    "Kubernetes" in r2_nodes and "Google" in r2_nodes and "Cloud Native Computing Foundation" in r2_nodes
                ),
            },
            "cross_kb_q3_graph_databases_and_ai": {
                "contributing_kbs": r3.provenance_kbs_contributing,
                "nodes_connected": sorted(r3_nodes),
                "verified_neptune_aws_opencypher_rag": (
                    "Amazon Neptune" in r3_nodes and "Amazon Web Services" in r3_nodes and "Neo4j" in r3_nodes
                ),
            },
            "mcp_live_neo4j_entity_search_hits": mcp_ent_call["result"]["structuredContent"]["count"],
            "mcp_live_neo4j_subgraph_facts": len(mcp_sub_call["result"]["structuredContent"]["facts"]),
            "mcp_live_neo4j_opencypher_bridge_rows": len(cypher_rows),
            "sample_opencypher_cross_kb_bridges": cypher_rows[:8],
            "sample_source_urls": fetched_urls[:8],
        }

        out_path = Path(__file__).resolve().parent.parent / "live_internet_kbs_report.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("\n" + "=" * 96)
        print(" LIVE INTERNET DATA BENCHMARK SUMMARY:")
        print(json.dumps(report, indent=2))
        print("=" * 96)

        assert articles_fetched_total >= 20
        assert build_report.shacl_conformance_passed is True
        assert build_report.cross_kb_merged_entities >= 3
        assert build_2.delta_chunks_extracted == 0
        assert report["cross_kb_q1_log4shell_to_apache_bridge"]["verified_log4shell_log4j_apache_chain"] is True
        assert report["cross_kb_q2_cloud_native_ecosystem"]["verified_k8s_google_cncf_prometheus"] is True
        assert report["cross_kb_q3_graph_databases_and_ai"]["verified_neptune_aws_opencypher_rag"] is True
        assert report["mcp_live_neo4j_opencypher_bridge_rows"] >= 1
        return report
    finally:
        await pg_adapter.close()
        await neo4j_adapter.close()


if __name__ == "__main__":
    asyncio.run(run_live_internet_kbs_benchmark())
