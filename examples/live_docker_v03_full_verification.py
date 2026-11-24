"""
Thorough Realistic & Adversarial Live-Docker E2E Verification (`v0.3.0`)

Tests EVERY `v0.2.0` + `v0.3.0` capability against LIVE infrastructure (zero mocks):
  1. Live PostgreSQL 16 (`pgvector`, 127.0.0.1:5434) with real fail-closed RLS + `NOBYPASSRLS` role (`kg_app_user`)
  2. Live Neo4j 5 (`bolt://127.0.0.1:7688`) with composite indexes, batched `UNWIND`, ABAC `acl_groups` & N-ary `qualifiers`
  3. Live Uvicorn HTTP Server (real TCP sockets + real `httpx.AsyncClient`) with 32+ byte JWT auth,
     anti-smuggling (`403`), rate limiting (`429`), REST `/v1/*`, `/mcp`, `/metrics`, `/ui`
  4. Multi-Pod Cold-Start Hydration of `enterprise_tbox_registry`, `kg_sync_jobs`, `kg_dlq_chunks`, `acl_groups`, `qualifiers`
  5. Live PostgreSQL `pg_try_advisory_lock` Multi-Replica Sync Worker coordination

Usage:
  docker compose up -d --wait
  python examples/live_docker_v03_full_verification.py
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any

import asyncpg
import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enterprise_kg_fabric import (  # noqa: E402
    AuthConfig,
    AuthMode,
    EnterpriseKGOntologyService,
    MultimodalDocumentIngestor,
    Neo4jOpenCypherGraphAdapter,
    PostgresPgVectorStoreAdapter,
    RawDocumentInput,
    SemanticFeatureHashEmbeddingProvider,
    create_hs256_jwt,
    create_kg_fabric_app,
)

PG_HOST = os.getenv("PGHOST", "127.0.0.1")
PG_PORT = int(os.getenv("PGPORT", "5434"))
PG_DB = os.getenv("PGDATABASE", "enterprise_kg_fabric")
PG_ADMIN_USER = os.getenv("KG_FABRIC_PG_USER", "kg_admin")
PG_ADMIN_PASS = os.getenv("KG_FABRIC_PG_PASS", "local_dev_only")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7688")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("KG_FABRIC_NEO4J_PASS", "local_dev_only")

APP_ROLE = "kg_app_user"
APP_ROLE_PASS = os.getenv("KG_FABRIC_APP_ROLE_PASSWORD") or f"rls_test_{secrets.token_hex(8)}"


def _pg_dsn(user: str, password: str) -> str:
    return f"postgresql://{user}:{password}@{PG_HOST}:{PG_PORT}/{PG_DB}"


async def _reset_live_stores(admin_dsn: str) -> None:
    conn = await asyncpg.connect(admin_dsn)
    try:
        for tbl in (
            "kg_dlq_chunks",
            "kg_sync_jobs",
            "enterprise_tbox_registry",
            "canonical_entity_embeddings",
            "unified_rag_chunks",
            "kb_ingestion_watermarks",
            "kb_federation_registry",
        ):
            await conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE;")
    finally:
        await conn.close()

    neo = Neo4jOpenCypherGraphAdapter(uri=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)
    try:
        driver = neo._get_driver()
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
    finally:
        await neo.close()


async def main() -> None:
    print("=" * 94)
    print(" LIVE DOCKER v0.3.0 FULL ADVERSARIAL & SOTA VERIFICATION (PG16 RLS + NEO4J 5 + UVICORN HTTP)")
    print("=" * 94)

    admin_dsn = _pg_dsn(PG_ADMIN_USER, PG_ADMIN_PASS)
    app_dsn = _pg_dsn(APP_ROLE, APP_ROLE_PASS)
    emb = SemanticFeatureHashEmbeddingProvider(default_dim=256)

    # 1. Reset + Migrate Schema + Provision real NOBYPASSRLS non-superuser role
    await _reset_live_stores(admin_dsn)
    admin_store = PostgresPgVectorStoreAdapter(dsn=admin_dsn, embedding_provider=emb, canonical_dim=256)
    await admin_store.migrate_schema_and_rls(app_role_name=APP_ROLE, app_role_password=APP_ROLE_PASS)
    await admin_store.close()
    print(f"[1] Migrated Postgres 16 schema + RLS policies + provisioned NOBYPASSRLS role '{APP_ROLE}'")

    # 2. Connect runtime service strictly as NON-SUPERUSER `kg_app_user` + live Neo4j 5
    pg_store = PostgresPgVectorStoreAdapter(dsn=app_dsn, embedding_provider=emb, canonical_dim=256)
    neo_store = Neo4jOpenCypherGraphAdapter(uri=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)
    svc = EnterpriseKGOntologyService(
        vector_store=pg_store,
        graph_store=neo_store,
        embedding_provider=emb,
        min_kb_support_for_approval=1,
        min_chunk_support_for_approval=1,
    )

    # 3. Seed realistic multi-tenant + multi-ACL + N-ary qualifier + Markdown table corpus
    table_md = (
        "| Service | Database | Compliance |\n"
        "| --- | --- | --- |\n"
        "| settlement-svc-01 | aurora-settlement-db | PCI-DSS-4.0 |\n"
    )
    table_triples = MultimodalDocumentIngestor.extract_triples_from_markdown_tables(table_md)
    assert len(table_triples) >= 1, "Expected structured triples extracted from Markdown table"
    table_doc = RawDocumentInput(
        doc_id="doc_table_arch",
        kb_id="kb_alpha_tables",
        tenant_id="tenant_alpha",
        domain_tag="architecture",
        title="Settlement Architecture Matrix",
        content=(
            f"{table_md}\n\n"
            "Microservice settlement-svc-01 depends on aurora-settlement-db. "
            "Database cluster aurora-settlement-db is governed by PCI-DSS-4.0."
        ),
        acl_groups=["public"],
        metadata={"parser_engine": "markdown_table_ingestor", "structured_triples": table_triples},
    )

    # Seed noisy hub leaves + true target + confidential merger DB + N-ary conditional qualifier + Tenant Beta data
    alpha_docs = [
        table_doc,
        RawDocumentInput(
            doc_id="doc_alpha_pub",
            kb_id="kb_alpha_sre",
            tenant_id="tenant_alpha",
            domain_tag="sre",
            title="Payment Router SRE Runbook",
            content=(
                "Microservice payment-router-01 depends on aurora-ledger-01. "
                "Database cluster aurora-ledger-01 is governed by SOC2-Type-II. "
                "Microservice payment-router-01 failed over to eu-central-1 only if primary latency exceeds 250ms."
            ),
            acl_groups=["public"],
        ),
        RawDocumentInput(
            doc_id="doc_alpha_conf",
            kb_id="kb_alpha_legal",
            tenant_id="tenant_alpha",
            domain_tag="legal",
            title="Project Helios Board Acquisition Memo",
            content=(
                "Microservice payment-router-01 depends on aurora-helios-merger-db. "
                "Database cluster aurora-helios-merger-db is governed by GDPR-Article-32."
            ),
            acl_groups=["legal_confidential"],
        ),
    ]
    # Add 10 noisy leaf services connected to SOC2-Type-II to test HippoRAG PPR hub damping on live Neo4j
    for i in range(1, 11):
        alpha_docs.append(
            RawDocumentInput(
                doc_id=f"doc_leaf_{i}",
                kb_id="kb_alpha_compliance",
                tenant_id="tenant_alpha",
                domain_tag="compliance",
                title=f"Leaf Compliance {i}",
                content=f"Database cluster db-leaf-{i:02d} is governed by SOC2-Type-II.",
                acl_groups=["public"],
            )
        )

    beta_docs = [
        RawDocumentInput(
            doc_id="doc_beta_1",
            kb_id="kb_beta_core",
            tenant_id="tenant_beta",
            domain_tag="sre",
            title="Tenant Beta Private System",
            content="Microservice beta-secret-api depends on aurora-beta-vault.",
            acl_groups=["public"],
        )
    ]

    await svc.ingest_raw_documents(alpha_docs)
    await svc.ingest_raw_documents(beta_docs)
    await svc.induce_and_align_ontology(tenant_id="tenant_alpha", tbox_version="3.0.0")
    await svc.induce_and_align_ontology(tenant_id="tenant_beta", tbox_version="3.0.0")

    # Inject 1 transient poison-pill failure to test live Postgres DLQ + Sync Job persistence
    orig_extract = svc.kg_extractor.extract_from_chunk
    tripped = {"done": False}

    def flaky_once(chunk, tbox, allow_dynamic_emergent=True):
        if chunk.tenant_id == "tenant_alpha" and not tripped["done"]:
            tripped["done"] = True
            raise RuntimeError("Live DLQ simulated extraction timeout")
        return orig_extract(chunk, tbox, allow_dynamic_emergent=allow_dynamic_emergent)

    svc.kg_extractor.extract_from_chunk = flaky_once  # type: ignore[method-assign]
    job_1 = await svc.execute_tracked_build_job(tenant_id="tenant_alpha", incremental=True)
    assert job_1.dlq_count == 1, f"Expected 1 DLQ failure, got {job_1.dlq_count}"

    # Restore extractor and complete delta build (previously failed chunk was NOT watermarked)
    svc.kg_extractor.extract_from_chunk = orig_extract  # type: ignore[method-assign]
    job_2 = await svc.execute_tracked_build_job(tenant_id="tenant_alpha", incremental=True)
    assert job_2.report is not None and job_2.report.delta_chunks_extracted >= 1
    await svc.build_or_update_knowledge_graph(tenant_id="tenant_beta", incremental=False)
    print(
        f"[2] Built live Neo4j 5 + Postgres 16 graphs as NOBYPASSRLS '{APP_ROLE}' | "
        f"Job1 DLQ={job_1.dlq_count} | Job2 Delta={job_2.report.delta_chunks_extracted}"
    )

    # 4. Direct PostgreSQL Kernel RLS Enforcement Proof (Raw asyncpg as `kg_app_user`) + Superuser Cross-Tenant Overwrite Guard
    raw_conn = await asyncpg.connect(app_dsn)
    try:
        # 4a. Without app.current_tenant_id -> FAIL-CLOSED (0 rows visible!)
        zero_rows = await raw_conn.fetchval("SELECT count(*) FROM unified_rag_chunks")
        assert zero_rows == 0, f"RLS fail-closed violated! Saw {zero_rows} rows without tenant context"

        # 4b. With app.current_tenant_id = 'tenant_alpha' -> NEVER sees tenant_beta rows!
        async with raw_conn.transaction():
            await raw_conn.execute("SELECT set_config('app.current_tenant_id', 'tenant_alpha', true)")
            alpha_count = await raw_conn.fetchval("SELECT count(*) FROM unified_rag_chunks")
            beta_leak = await raw_conn.fetchval(
                "SELECT count(*) FROM unified_rag_chunks WHERE tenant_id = 'tenant_beta'"
            )
            assert alpha_count > 0 and beta_leak == 0
    finally:
        await raw_conn.close()

    # 4c. Critical #3 & High #7: Even as Postgres SUPERUSER (`kg_admin`), Tenant Beta CANNOT overwrite Tenant Alpha's KB or chunk!
    from enterprise_kg_fabric import KBSourceType, VectorChunkRecord, VectorKBRegistration

    su_store = PostgresPgVectorStoreAdapter(dsn=admin_dsn, embedding_provider=emb, canonical_dim=256)
    try:
        alpha_chunks_live = await su_store.fetch_kb_chunks("kb_alpha_legal", tenant_id="tenant_alpha")
        assert len(alpha_chunks_live) >= 1
        kb_blocked = False
        try:
            await su_store.register_kb(
                VectorKBRegistration(
                    kb_id="kb_alpha_legal", tenant_id="tenant_beta", domain_tag="legal", table_name="hijack"
                )
            )
        except PermissionError:
            kb_blocked = True
        assert kb_blocked, "Superuser KB takeover was NOT blocked!"

        chunk_blocked = False
        try:
            await su_store.upsert_kb_chunks(
                "kb_alpha_legal",
                [
                    VectorChunkRecord(
                        chunk_id=alpha_chunks_live[0].chunk_id,
                        kb_id="kb_alpha_legal",
                        tenant_id="tenant_beta",
                        doc_id="hijack",
                        text="OVERWRITTEN",
                        acl_groups=["public"],
                    )
                ],
            )
        except PermissionError:
            chunk_blocked = True
        assert chunk_blocked, "Superuser cross-tenant chunk overwrite was NOT blocked!"

        # 4d. High #4: Registering a real legacy `pgvector` table auto-hydrates on `build_or_update_knowledge_graph`!
        su_conn = await asyncpg.connect(admin_dsn)
        try:
            await su_conn.execute("DROP TABLE IF EXISTS legacy_finops_vectors;")
            await su_conn.execute(
                """
                CREATE TABLE legacy_finops_vectors (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding vector(384) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                """
            )
            v384 = "[" + ",".join(f"{x:.5f}" for x in emb.embed_text("finops", dim=384)) + "]"
            await su_conn.execute(
                """
                INSERT INTO legacy_finops_vectors (id, doc_id, chunk_text, embedding, metadata)
                VALUES ('leg_1', 'fin_doc_1', 'Microservice payment-router-01 is allocated to cost center CC-505-Live.', $1::vector, '{}'::jsonb)
                """,
                v384,
            )
            await su_conn.execute(f"GRANT SELECT ON legacy_finops_vectors TO {APP_ROLE};")
        finally:
            await su_conn.close()

        from enterprise_kg_fabric import VectorKBColumnMapping

        await svc.register_vector_kb(
            VectorKBRegistration(
                kb_id="kb_legacy_auto_finops",
                tenant_id="tenant_alpha",
                domain_tag="finops",
                source_type=KBSourceType.EXISTING_PGVECTOR_KB,
                table_name="legacy_finops_vectors",
                embedding_dim=384,
                column_mapping=VectorKBColumnMapping(id_column="id", text_column="chunk_text"),
            )
        )
        legacy_build = await svc.build_or_update_knowledge_graph(tenant_id="tenant_alpha", incremental=True)
        assert legacy_build.delta_chunks_extracted >= 1, (
            f"Expected registered legacy pgvector table to auto-hydrate on build, got {legacy_build}"
        )
    finally:
        await su_store.close()
    print(f"[3] Verified PostgreSQL 16 Kernel RLS (`NOBYPASSRLS`) + Superuser Overwrite Guard + Auto Legacy Table Hydration")

    # 5. Live PostgreSQL `pg_try_advisory_lock` Multi-Replica Sync Worker Proof
    lock_key = svc._sync_lock_key("tenant_alpha")
    holder_store = PostgresPgVectorStoreAdapter(dsn=app_dsn, embedding_provider=emb, canonical_dim=256)
    try:
        assert await holder_store.try_acquire_advisory_lock(lock_key) is True
        locked_cycle = await svc.run_scheduled_sync_cycle(tenant_ids=["tenant_alpha"])
        assert "tenant_alpha" in locked_cycle["tenants_skipped_locked"], f"Expected skip: {locked_cycle}"
        await holder_store.release_advisory_lock(lock_key)
        ok_cycle = await svc.run_scheduled_sync_cycle(tenant_ids=["tenant_alpha"])
        assert "tenant_alpha" in ok_cycle["tenants_synced"]
    finally:
        await holder_store.close()
    print("[4] Verified live PostgreSQL `pg_try_advisory_lock` multi-replica sync coordination")

    # 6. Boot LIVE Uvicorn HTTP Server (real TCP socket) with 32+ byte JWT auth + rate limiter
    jwt_secret = secrets.token_hex(32)  # 64 ASCII bytes (> 32 byte minimum)
    auth_cfg = AuthConfig(
        mode=AuthMode.JWT,
        jwt_secret=jwt_secret,
        api_rate_limit_rpm=50,
    )
    app = create_kg_fabric_app(service=svc, auth_config=auth_cfg)

    port = 18089
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    base_url = f"http://127.0.0.1:{port}"
    report: dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=15.0, trust_env=False) as client:
            # 6a. Public /health & /ui
            r_health = await client.get("/health")
            assert r_health.status_code == 200, f"/health failed: {r_health.status_code} {r_health.text}"
            assert r_health.json()["auth_mode"] == "jwt", f"Unexpected /health: {r_health.json()}"
            r_ui = await client.get("/ui")
            assert r_ui.status_code == 200 and "Governance Console" in r_ui.text

            # 6b. Unauthenticated & Expired JWT -> 401
            r_unauth = await client.post("/v1/rag/query", json={"tenant_id": "tenant_alpha", "query": "test"})
            assert r_unauth.status_code == 401
            exp_jwt = create_hs256_jwt(
                {"sub": "u1", "tenant_id": "tenant_alpha", "acl_groups": ["public"], "roles": ["reader"], "exp": int(time.time()) - 30},
                jwt_secret,
            )
            r_exp = await client.get("/v1/telemetry", headers={"Authorization": f"Bearer {exp_jwt}"})
            assert r_exp.status_code == 401

            pub_jwt = create_hs256_jwt(
                {"sub": "analyst", "tenant_id": "tenant_alpha", "acl_groups": ["public"], "roles": ["reader"], "exp": int(time.time()) + 3600},
                jwt_secret,
            )
            conf_jwt = create_hs256_jwt(
                {"sub": "general_counsel", "tenant_id": "tenant_alpha", "acl_groups": ["public", "legal_confidential"], "roles": ["reader"], "exp": int(time.time()) + 3600},
                jwt_secret,
            )
            pub_hdr = {"Authorization": f"Bearer {pub_jwt}"}
            conf_hdr = {"Authorization": f"Bearer {conf_jwt}"}

            # 6c. Cross-tenant smuggling, query-vs-body smuggling, reader write block, raw Cypher block,
            #     MCP plural write block, cross-tenant KB/chunk overwrite block, MCP cross-tenant resource -> 403
            r_cross = await client.get("/v1/telemetry", params={"tenant_id": "tenant_beta"}, headers=pub_hdr)
            assert r_cross.status_code == 403
            r_smuggle = await client.post(
                "/v1/rag/check-answerable",
                params={"tenant_id": "tenant_alpha"},
                json={"tenant_id": "tenant_beta", "query": "payment-router-01"},
                headers=pub_hdr,
            )
            assert r_smuggle.status_code == 403
            r_write = await client.post("/v1/graph/build", json={"tenant_id": "tenant_alpha"}, headers=pub_hdr)
            assert r_write.status_code == 403
            r_cypher_rb = await client.post(
                "/v1/search/cypher",
                json={"tenant_id": "tenant_alpha", "cypher_query": "MATCH (n) RETURN n.canonical_name"},
                headers=pub_hdr,
            )
            assert r_cypher_rb.status_code == 403
            r_mcp_write = await client.post(
                "/v1/mcp/call",
                json={
                    "tool_name": "ingest_raw_documents",
                    "arguments": {
                        "documents": [
                            {"doc_id": "pwn", "kb_id": "kb_alpha_sre", "tenant_id": "tenant_beta", "title": "X", "content": "Evil"}
                        ]
                    },
                },
                headers=pub_hdr,
            )
            assert r_mcp_write.status_code == 403
            r_mcp_res = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "resources/read", "params": {"uri": "kg://ontology/tenant_beta/tbox"}},
                headers=pub_hdr,
            )
            assert r_mcp_res.status_code == 403 or "error" in r_mcp_res.json()

            # 6d. Live ABAC Zero-Leakage over LIVE Postgres + LIVE Neo4j + Path 4 OpenCypher + LIVE /mcp!
            r_pub_rag = await client.post(
                "/v1/rag/query",
                json={
                    "tenant_id": "tenant_alpha",
                    "query": "What databases, regions, and compliance standards are connected to payment-router-01?",
                    "caller_acl_groups": ["public", "legal_confidential"],  # Spoof attempt!
                },
                headers=pub_hdr,
            )
            assert r_pub_rag.status_code == 200
            pub_rag_json = r_pub_rag.json()
            pub_rag_str = json.dumps(pub_rag_json)
            assert "aurora-ledger-01" in pub_rag_str
            assert "aurora-helios-merger-db" not in pub_rag_str, "ABAC LEAK to public caller!"
            assert "GDPR-Article-32" not in pub_rag_str, "ABAC LEAK of GDPR-Article-32!"
            assert "aurora-beta-vault" not in pub_rag_str, "CROSS-TENANT LEAK!"

            # Live N-ary qualifier persisted in Neo4j 5 and returned!
            failover_facts = [
                f for f in pub_rag_json["graph_facts"]
                if f["predicate"] == "DEPLOYED_IN_PRIMARY" and f["target_name"] == "eu-central-1"
            ]
            assert failover_facts and "condition" in failover_facts[0]["qualifiers"], (
                f"Expected N-ary condition qualifier persisted in Neo4j 5, got {failover_facts}"
            )

            # Confidential caller DOES see aurora-helios-merger-db & GDPR-Article-32 from live Postgres + Neo4j
            r_conf_rag = await client.post(
                "/v1/rag/query",
                json={
                    "tenant_id": "tenant_alpha",
                    "query": "What databases and GDPR standards are connected to payment-router-01?",
                },
                headers=conf_hdr,
            )
            assert r_conf_rag.status_code == 200
            conf_rag_str = json.dumps(r_conf_rag.json())
            assert "aurora-helios-merger-db" in conf_rag_str
            assert "GDPR-Article-32" in conf_rag_str

            # 6e. Live HippoRAG 2 PPR Hub Damping on Live Neo4j 5
            r_ppr = await client.post(
                "/v1/search/ppr",
                json={"tenant_id": "tenant_alpha", "seed_entities": ["payment-router-01"], "top_k_facts": 5},
                headers=pub_hdr,
            )
            assert r_ppr.status_code == 200
            ppr_data = r_ppr.json()
            assert ppr_data["ranked_facts"][0]["target_name"] != "SOC2-Type-II", (
                f"Super-hub SOC2-Type-II should be damped below direct target! {ppr_data['ranked_facts'][0]}"
            )

            # 6f. Live Leiden/Louvain Global Sensemaking on Live Neo4j 5
            r_global = await client.post(
                "/v1/search/global",
                json={"tenant_id": "tenant_alpha", "query": "Summarize overall architecture and compliance clusters across the estate"},
                headers=pub_hdr,
            )
            assert r_global.status_code == 200
            global_data = r_global.json()
            assert len(global_data["communities_matched"]) >= 1
            assert "aurora-helios-merger-db" not in json.dumps(global_data)

            # 6g. Live Pre-flight Answerability Check + Markdown Table Triples + Prometheus Metrics
            r_ans_ok = await client.post(
                "/v1/rag/check-answerable",
                json={"tenant_id": "tenant_alpha", "query": "What database does settlement-svc-01 use?"},
                headers=pub_hdr,
            )
            assert r_ans_ok.status_code == 200 and r_ans_ok.json()["status"] == "ANSWERABLE"

            r_ans_no = await client.post(
                "/v1/rag/check-answerable",
                json={"tenant_id": "tenant_alpha", "query": "Quantum fusion reactor coolant telemetry"},
                headers=pub_hdr,
            )
            assert r_ans_no.status_code == 200 and r_ans_no.json()["status"] == "UNANSWERABLE"

            r_prom = await client.get("/metrics", headers=pub_hdr)
            assert r_prom.status_code == 200 and "kg_fabric_queries_total" in r_prom.text

            # 6h. Live MCP HTTP tools/list (14 tools) + ABAC-protected MCP call + Quality Scorecard
            r_mcp_list = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                headers=pub_hdr,
            )
            assert r_mcp_list.status_code == 200
            mcp_tools = r_mcp_list.json()["result"]["tools"]
            assert len(mcp_tools) == 14

            r_qual = await client.get(
                "/v1/quality/report",
                params={"tenant_id": "tenant_alpha"},
                headers=pub_hdr,
            )
            assert r_qual.status_code == 200
            qual_data = r_qual.json()
            assert qual_data["overall_score"] >= 70.0
            assert qual_data["tbox_metrics"]["shacl_conforms"] is True
            assert qual_data["extraction_er_metrics"]["provenance_grounded_edge_rate"] == 1.0

            r_prom_after = await client.get("/metrics", headers=pub_hdr)
            assert r_prom_after.status_code == 200 and 'kg_fabric_quality_score{tenant_id="tenant_alpha"}' in r_prom_after.text

            r_mcp_call = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "query_hybrid_graph_rag",
                        "arguments": {
                            "tenant_id": "tenant_alpha",
                            "query": "What databases are connected to payment-router-01?",
                            "caller_acl_groups": ["public", "legal_confidential"],
                        },
                    },
                },
                headers=pub_hdr,
            )
            assert r_mcp_call.status_code == 200
            assert "aurora-helios-merger-db" not in json.dumps(r_mcp_call.json())

            print("[5] Verified LIVE Uvicorn HTTP + JWT + Anti-Smuggling + ABAC + PPR + Leiden + Answerability + 13 MCP Tools")

        # 7. Cold-Start Multi-Pod Restart Verification (Brand-New Service, Zero RAM Cache -> Live Postgres + Neo4j)
        pod2_pg = PostgresPgVectorStoreAdapter(dsn=app_dsn, embedding_provider=emb, canonical_dim=256)
        pod2_neo = Neo4jOpenCypherGraphAdapter(uri=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)
        pod2 = EnterpriseKGOntologyService(vector_store=pod2_pg, graph_store=pod2_neo, embedding_provider=emb)
        try:
            hydrated_tbox = await pod2.load_or_get_tbox("tenant_alpha")
            assert hydrated_tbox.version == "3.0.0"
            hydrated_job = await pod2.get_sync_job_async(job_1.job_id)
            assert hydrated_job is not None and hydrated_job.dlq_count == 1
            hydrated_dlq = await pod2.list_dlq_records_async(tenant_id="tenant_alpha")
            assert len(hydrated_dlq) >= 1 and "Live DLQ simulated" in hydrated_dlq[0].error_message

            # Verify Neo4j + Postgres cold-hydrated acl_groups & qualifiers directly
            hydrated_chunks = await pod2_pg.fetch_kb_chunks("kb_alpha_legal")
            assert hydrated_chunks and "legal_confidential" in hydrated_chunks[0].acl_groups
            hydrated_edges = await pod2_neo.get_all_edges("tenant_alpha", active_only=True)
            assert any("legal_confidential" in e.acl_groups for e in hydrated_edges)
            assert any("condition" in e.qualifiers for e in hydrated_edges)
        finally:
            await pod2.close()
        print("[6] Verified Cold-Start Multi-Pod Disk Hydration (TBox, Sync Jobs, DLQ, Postgres acl_groups, Neo4j acl_groups + qualifiers)")

        report = {
            "postgres_rls_nobypassrls_role": APP_ROLE,
            "postgres_rls_fail_closed_verified": True,
            "postgres_rls_cross_tenant_zero_leak": True,
            "postgres_advisory_lock_sync_coordination": True,
            "live_http_jwt_32b_and_anti_smuggling": True,
            "live_abac_zero_leakage_rest_mcp_neo4j_pgvector": True,
            "live_neo4j_nary_qualifiers_persisted": True,
            "live_hipporag_ppr_hub_damping_verified": True,
            "live_leiden_global_sensemaking_communities": len(global_data["communities_matched"]),
            "live_answerability_check_verified": True,
            "live_mcp_tools_count": len(mcp_tools),
            "live_quality_scorecard_overall": qual_data["overall_score"],
            "live_quality_scorecard_grade": qual_data["grade"],
            "live_quality_dimension_scores": qual_data["dimension_scores"],
            "live_quality_findings_count": len(qual_data["findings"]),
            "live_cold_start_pod_hydration_tbox_jobs_dlq_acls": True,
        }
        print("=" * 94)
        print(" LIVE DOCKER v0.3.0 VERIFICATION REPORT:")
        print(json.dumps(report, indent=2))
        print("=" * 94)
    finally:
        server.should_exit = True
        await server_task
        await svc.close()


if __name__ == "__main__":
    asyncio.run(main())
