from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Sequence

from ..models import (
    DLQFailedChunkRecord,
    EnterpriseCompiledTBox,
    KBWatermarkState,
    KGEntityNode,
    RetrievedChunkContext,
    SyncJobRecord,
    VectorChunkRecord,
    VectorKBRegistration,
    utc_now_iso,
)
from ..vector_clustering import (
    EmbeddingProvider,
    FederatedEmbeddingModelRouter,
    cosine_similarity,
)


def get_production_pgvector_ddl(canonical_dim: int = 256) -> str:
    """
    Production PostgreSQL 16+ & pgvector 0.7+ DDL for Multi-KB Federation,
    Incremental SHA-256 CDC Watermarks, Hybrid HNSW + GIN BM25 Search, and Row-Level Security (RLS).
    """
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_federation_registry (
    kb_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    domain_tag TEXT NOT NULL,
    source_type TEXT NOT NULL,
    postgres_dsn_alias TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    table_name TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    embedding_model_id TEXT NOT NULL,
    column_mapping JSONB NOT NULL,
    owner_team TEXT NOT NULL,
    security_classification TEXT NOT NULL,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kb_ingestion_watermarks (
    kb_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    tbox_version TEXT NOT NULL,
    last_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (kb_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS unified_rag_chunks (
    chunk_id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    domain_tag TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    text TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    canonical_embedding vector({canonical_dim}) NOT NULL,
    section_path TEXT NOT NULL DEFAULT '',
    acl_groups JSONB NOT NULL DEFAULT '["public"]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE unified_rag_chunks
    ADD COLUMN IF NOT EXISTS acl_groups JSONB NOT NULL DEFAULT '["public"]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_unified_chunks_hnsw
    ON unified_rag_chunks USING hnsw (canonical_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_unified_chunks_tsv
    ON unified_rag_chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS idx_unified_chunks_tenant_kb
    ON unified_rag_chunks (tenant_id, kb_id, domain_tag);

CREATE INDEX IF NOT EXISTS idx_unified_chunks_acl_gin
    ON unified_rag_chunks USING gin (acl_groups);

CREATE TABLE IF NOT EXISTS canonical_entity_embeddings (
    entity_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    ontology_class TEXT NOT NULL,
    upper_category TEXT NOT NULL,
    source_kb_ids JSONB NOT NULL,
    embedding vector({canonical_dim}) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_canonical_entities_hnsw
    ON canonical_entity_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS enterprise_tbox_registry (
    tenant_id TEXT PRIMARY KEY,
    tbox_version TEXT NOT NULL,
    tbox_json JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kg_sync_jobs (
    job_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    status TEXT NOT NULL,
    job_json JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kg_sync_jobs_tenant
    ON kg_sync_jobs (tenant_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS kg_dlq_chunks (
    dlq_id SERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    kb_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    record_json JSONB NOT NULL,
    failed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kg_dlq_chunks_tenant
    ON kg_dlq_chunks (tenant_id, failed_at DESC);

-- Fail-Closed Row-Level Security (RLS) across all multi-tenant tables
-- Enforced automatically for non-superuser app roles (`NOBYPASSRLS`) via `SET LOCAL app.current_tenant_id`
ALTER TABLE kb_federation_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE unified_rag_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE canonical_entity_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE enterprise_tbox_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE kg_sync_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE kg_dlq_chunks ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS rls_kb_federation_tenant ON kb_federation_registry;
CREATE POLICY rls_kb_federation_tenant ON kb_federation_registry
    USING (
        current_setting('app.current_tenant_id', true) = '*'
        OR tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')
    )
    WITH CHECK (
        current_setting('app.current_tenant_id', true) = '*'
        OR tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')
    );

DROP POLICY IF EXISTS rls_unified_chunks_tenant ON unified_rag_chunks;
CREATE POLICY rls_unified_chunks_tenant ON unified_rag_chunks
    USING (
        current_setting('app.current_tenant_id', true) = '*'
        OR tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')
    )
    WITH CHECK (
        current_setting('app.current_tenant_id', true) = '*'
        OR tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')
    );

DROP POLICY IF EXISTS rls_canonical_entities_tenant ON canonical_entity_embeddings;
CREATE POLICY rls_canonical_entities_tenant ON canonical_entity_embeddings
    USING (
        current_setting('app.current_tenant_id', true) = '*'
        OR tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')
    )
    WITH CHECK (
        current_setting('app.current_tenant_id', true) = '*'
        OR tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')
    );

DROP POLICY IF EXISTS rls_tbox_registry_tenant ON enterprise_tbox_registry;
CREATE POLICY rls_tbox_registry_tenant ON enterprise_tbox_registry
    USING (
        current_setting('app.current_tenant_id', true) = '*'
        OR tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')
    )
    WITH CHECK (
        current_setting('app.current_tenant_id', true) = '*'
        OR tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')
    );

DROP POLICY IF EXISTS rls_sync_jobs_tenant ON kg_sync_jobs;
CREATE POLICY rls_sync_jobs_tenant ON kg_sync_jobs
    USING (
        current_setting('app.current_tenant_id', true) = '*'
        OR tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')
    )
    WITH CHECK (
        current_setting('app.current_tenant_id', true) = '*'
        OR tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')
    );

DROP POLICY IF EXISTS rls_dlq_chunks_tenant ON kg_dlq_chunks;
CREATE POLICY rls_dlq_chunks_tenant ON kg_dlq_chunks
    USING (
        current_setting('app.current_tenant_id', true) = '*'
        OR tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')
    )
    WITH CHECK (
        current_setting('app.current_tenant_id', true) = '*'
        OR tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')
    );
""".strip()


def get_rls_app_role_ddl(role_name: str = "kg_app_user", password: str = "") -> str:
    """
    Idempotently provisions a least-privilege PostgreSQL runtime role (`NOBYPASSRLS`)
    so PostgreSQL Row-Level Security policies are strictly enforced in production.
    """
    if not re.match(r"^[a-z_][a-z0-9_]{0,62}$", role_name):
        raise ValueError(f"Invalid PostgreSQL role name: {role_name}")
    escaped_pw = password.replace("'", "''")
    pw_clause = f"PASSWORD '{escaped_pw}'" if escaped_pw else ""
    return f"""
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '{role_name}') THEN
        CREATE ROLE {role_name} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS {pw_clause};
    ELSE
        ALTER ROLE {role_name} WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS {pw_clause};
    END IF;
END
$$;
GRANT USAGE ON SCHEMA public TO {role_name};
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {role_name};
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role_name};
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {role_name};
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO {role_name};
""".strip()


class PgVectorStorePort(ABC):
    """Port interface for multi-KB PostgreSQL + pgvector operations."""

    async def save_tbox(self, tbox: EnterpriseCompiledTBox) -> None:
        return None

    async def load_tbox(self, tenant_id: str) -> EnterpriseCompiledTBox | None:
        return None

    async def save_sync_job(self, job: SyncJobRecord) -> None:
        return None

    async def get_sync_job(self, job_id: str) -> SyncJobRecord | None:
        return None

    async def append_dlq_record(self, record: DLQFailedChunkRecord) -> None:
        return None

    async def list_dlq_records(self, tenant_id: str | None = None) -> list[DLQFailedChunkRecord]:
        return []

    async def try_acquire_advisory_lock(self, lock_key: int) -> bool:
        return True

    async def release_advisory_lock(self, lock_key: int) -> None:
        return None

    async def unregister_kb(self, tenant_id: str, kb_id: str) -> int:
        return 0

    async def delete_document(self, tenant_id: str, kb_id: str, doc_id: str) -> int:
        return 0

    async def purge_tenant(self, tenant_id: str) -> dict[str, int]:
        return {}

    async def close(self) -> None:
        return None

    @abstractmethod
    async def register_kb(self, registration: VectorKBRegistration) -> None:
        raise NotImplementedError

    @abstractmethod
    async def list_registered_kbs(self, tenant_id: str | None = None) -> list[VectorKBRegistration]:
        raise NotImplementedError

    @abstractmethod
    async def upsert_kb_chunks(self, kb_id: str, chunks: Sequence[VectorChunkRecord]) -> int:
        raise NotImplementedError

    @abstractmethod
    async def fetch_kb_chunks(
        self,
        kb_id: str,
        only_unindexed_or_changed: bool = False,
        target_tbox_version: str | None = None,
        tenant_id: str | None = None,
    ) -> list[VectorChunkRecord]:
        raise NotImplementedError

    @abstractmethod
    async def mark_chunks_watermarked(
        self,
        watermarks: Sequence[KBWatermarkState],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def upsert_canonical_entity_embeddings(
        self,
        entities: Sequence[KGEntityNode],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def hybrid_search_chunks(
        self,
        tenant_id: str,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 6,
        kb_filter: list[str] | None = None,
        domain_filter: list[str] | None = None,
        caller_acl_groups: list[str] | None = None,
    ) -> list[RetrievedChunkContext]:
        raise NotImplementedError

    @abstractmethod
    async def search_similar_entities(
        self,
        tenant_id: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[KGEntityNode, float]]:
        raise NotImplementedError


class InMemoryPgVectorStoreEmulator(PgVectorStorePort):
    """
    High-fidelity in-process emulator of a multi-database PostgreSQL + `pgvector` cluster.
    Simulates 100s of heterogeneous vector KBs (with native 384/768/1536-dim vectors),
    SHA-256 incremental CDC watermarks, HNSW dense cosine search, BM25 lexical search,
    and Reciprocal Rank Fusion (RRF) + tenant Row-Level Security (RLS).
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        canonical_dim: int = 256,
        model_router: FederatedEmbeddingModelRouter | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.canonical_dim = canonical_dim
        self.model_router = model_router or FederatedEmbeddingModelRouter(embedding_provider)
        self._kb_registry: dict[str, VectorKBRegistration] = {}
        self._kb_tables: dict[str, dict[str, VectorChunkRecord]] = {}
        self._watermarks: dict[tuple[str, str], KBWatermarkState] = {}
        self._entity_index: dict[str, KGEntityNode] = {}
        self._tboxes: dict[str, EnterpriseCompiledTBox] = {}
        self._sync_jobs: dict[str, SyncJobRecord] = {}
        self._dlq_records: list[DLQFailedChunkRecord] = []
        self._advisory_locks: set[int] = set()

    async def try_acquire_advisory_lock(self, lock_key: int) -> bool:
        if lock_key in self._advisory_locks:
            return False
        self._advisory_locks.add(lock_key)
        return True

    async def release_advisory_lock(self, lock_key: int) -> None:
        self._advisory_locks.discard(lock_key)

    async def save_tbox(self, tbox: EnterpriseCompiledTBox) -> None:
        self._tboxes[tbox.tenant_id] = tbox

    async def load_tbox(self, tenant_id: str) -> EnterpriseCompiledTBox | None:
        return self._tboxes.get(tenant_id)

    async def save_sync_job(self, job: SyncJobRecord) -> None:
        self._sync_jobs[job.job_id] = job

    async def get_sync_job(self, job_id: str) -> SyncJobRecord | None:
        return self._sync_jobs.get(job_id)

    async def append_dlq_record(self, record: DLQFailedChunkRecord) -> None:
        self._dlq_records.append(record)

    async def list_dlq_records(self, tenant_id: str | None = None) -> list[DLQFailedChunkRecord]:
        if tenant_id is None:
            return list(self._dlq_records)
        return [r for r in self._dlq_records if r.tenant_id == tenant_id]

    async def register_kb(self, registration: VectorKBRegistration) -> None:
        existing = self._kb_registry.get(registration.kb_id)
        if existing is not None and existing.tenant_id != registration.tenant_id:
            raise PermissionError(
                f"KB '{registration.kb_id}' is owned by another tenant and cannot be overwritten."
            )
        self._kb_registry[registration.kb_id] = registration
        self._kb_tables.setdefault(registration.kb_id, {})

    async def unregister_kb(self, tenant_id: str, kb_id: str) -> int:
        existing = self._kb_registry.get(kb_id)
        if existing is None:
            return 0
        if existing.tenant_id != tenant_id:
            raise PermissionError(f"KB '{kb_id}' belongs to another tenant.")
        del self._kb_registry[kb_id]
        removed_chunks = len(self._kb_tables.pop(kb_id, {}))
        for key in [k for k in self._watermarks if k[0] == kb_id]:
            del self._watermarks[key]
        return removed_chunks

    async def delete_document(self, tenant_id: str, kb_id: str, doc_id: str) -> int:
        existing = self._kb_registry.get(kb_id)
        if existing is not None and existing.tenant_id != tenant_id:
            raise PermissionError(f"KB '{kb_id}' belongs to another tenant.")
        table = self._kb_tables.get(kb_id, {})
        to_delete = [
            cid for cid, chk in table.items() if chk.tenant_id == tenant_id and chk.doc_id == doc_id
        ]
        for cid in to_delete:
            del table[cid]
            self._watermarks.pop((kb_id, cid), None)
        return len(to_delete)

    async def purge_tenant(self, tenant_id: str) -> dict[str, int]:
        kb_ids = [kid for kid, reg in self._kb_registry.items() if reg.tenant_id == tenant_id]
        chunks_removed = 0
        for kid in kb_ids:
            chunks_removed += await self.unregister_kb(tenant_id, kid)
        ent_removed = 0
        for eid in [k for k, v in self._entity_index.items() if v.tenant_id == tenant_id]:
            del self._entity_index[eid]
            ent_removed += 1
        tbox_removed = 1 if self._tboxes.pop(tenant_id, None) is not None else 0
        jobs_removed = 0
        for jid in [k for k, v in self._sync_jobs.items() if v.tenant_id == tenant_id]:
            del self._sync_jobs[jid]
            jobs_removed += 1
        before_dlq = len(self._dlq_records)
        self._dlq_records = [r for r in self._dlq_records if r.tenant_id != tenant_id]
        return {
            "kbs_removed": len(kb_ids),
            "chunks_removed": chunks_removed,
            "entity_embeddings_removed": ent_removed,
            "tboxes_removed": tbox_removed,
            "sync_jobs_removed": jobs_removed,
            "dlq_records_removed": before_dlq - len(self._dlq_records),
        }

    async def list_registered_kbs(self, tenant_id: str | None = None) -> list[VectorKBRegistration]:
        all_kbs = list(self._kb_registry.values())
        if tenant_id is not None:
            return [kb for kb in all_kbs if kb.tenant_id == tenant_id]
        return all_kbs

    async def upsert_kb_chunks(self, kb_id: str, chunks: Sequence[VectorChunkRecord]) -> int:
        if not chunks:
            return 0
        if kb_id not in self._kb_registry:
            # Auto-register default KB entry if needed
            first_tenant = chunks[0].tenant_id
            first_domain = chunks[0].domain_tag
            first_dim = chunks[0].native_embedding_dim if chunks[0].native_embedding_dim > 0 else 768
            await self.register_kb(
                VectorKBRegistration(
                    kb_id=kb_id,
                    tenant_id=first_tenant,
                    domain_tag=first_domain,
                    table_name=f"pgvec_{kb_id}",
                    embedding_dim=first_dim,
                )
            )

        reg = self._kb_registry[kb_id]
        for chk in chunks:
            if chk.tenant_id != reg.tenant_id:
                raise PermissionError(
                    f"Cannot write chunk for tenant '{chk.tenant_id}' into KB '{kb_id}' owned by another tenant."
                )

        table = self._kb_tables.setdefault(kb_id, {})
        count = 0
        for chk in chunks:
            prev = table.get(chk.chunk_id)
            if prev is not None and prev.tenant_id != chk.tenant_id:
                raise PermissionError(
                    f"Chunk '{chk.chunk_id}' belongs to another tenant and cannot be overwritten."
                )
            if not chk.native_embedding:
                chk.native_embedding = self.embedding_provider.embed_text(
                    chk.text, dim=reg.embedding_dim
                )
                chk.native_embedding_dim = reg.embedding_dim
            if not chk.canonical_embedding:
                chk.canonical_embedding = self.embedding_provider.embed_text(
                    chk.text, dim=self.canonical_dim
                )
            table[chk.chunk_id] = chk
            count += 1
        return count

    async def fetch_kb_chunks(
        self,
        kb_id: str,
        only_unindexed_or_changed: bool = False,
        target_tbox_version: str | None = None,
        tenant_id: str | None = None,
    ) -> list[VectorChunkRecord]:
        table = self._kb_tables.get(kb_id, {})
        values = [
            chk for chk in table.values() if tenant_id is None or chk.tenant_id == tenant_id
        ]
        if not only_unindexed_or_changed:
            return values

        delta_chunks: list[VectorChunkRecord] = []
        for chk in values:
            wm = self._watermarks.get((kb_id, chk.chunk_id))
            if wm is None:
                delta_chunks.append(chk)
            elif wm.content_sha256 != chk.content_sha256:
                delta_chunks.append(chk)
            elif target_tbox_version is not None and wm.tbox_version != target_tbox_version:
                delta_chunks.append(chk)
        return delta_chunks

    async def mark_chunks_watermarked(
        self,
        watermarks: Sequence[KBWatermarkState],
    ) -> None:
        for wm in watermarks:
            self._watermarks[(wm.kb_id, wm.chunk_id)] = wm

    async def upsert_canonical_entity_embeddings(
        self,
        entities: Sequence[KGEntityNode],
    ) -> None:
        for ent in entities:
            if not ent.embedding:
                ent.embedding = self.embedding_provider.embed_text(
                    f"{ent.ontology_class} {ent.canonical_name} {ent.summary}",
                    dim=self.canonical_dim,
                )
            self._entity_index[ent.entity_id] = ent

    @staticmethod
    def _tokenize_bm25(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_\-]+", text.lower())

    async def hybrid_search_chunks(
        self,
        tenant_id: str,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 6,
        kb_filter: list[str] | None = None,
        domain_filter: list[str] | None = None,
        caller_acl_groups: list[str] | None = None,
    ) -> list[RetrievedChunkContext]:
        candidate_chunks: list[VectorChunkRecord] = []
        kb_allow = set(kb_filter) if kb_filter else None
        dom_allow = set(domain_filter) if domain_filter else None
        caller_acls = {g.lower() for g in caller_acl_groups} if caller_acl_groups is not None else None

        for kb_id, table in self._kb_tables.items():
            if kb_allow is not None and kb_id not in kb_allow:
                continue
            for chk in table.values():
                if chk.tenant_id != tenant_id:
                    continue
                if dom_allow is not None and chk.domain_tag not in dom_allow:
                    continue
                if caller_acls is not None:
                    chk_acls = {g.lower() for g in chk.acl_groups}
                    if "public" not in chk_acls and not caller_acls.intersection(chk_acls):
                        continue
                candidate_chunks.append(chk)

        if not candidate_chunks:
            return []

        # 1. Multi-Model Federated Dense Search:
        # Group KBs by (embedding_model_id, embedding_dim) so 100s of KBs with different
        # models (e.g. 384d BGE, 768d Gecko, 1024d Cohere, 1536d OpenAI) are searched natively
        # without re-embedding legacy chunks, then fused via scale-invariant rank-based RRF.
        chunks_by_model_group: dict[tuple[str, int], list[VectorChunkRecord]] = {}
        for chk in candidate_chunks:
            reg = self._kb_registry.get(chk.kb_id)
            model_id = reg.embedding_model_id if reg else "canonical"
            dim = chk.native_embedding_dim or (reg.embedding_dim if reg else self.canonical_dim)
            chunks_by_model_group.setdefault((model_id, dim), []).append(chk)

        query_vecs_by_model = self.model_router.embed_query_per_model_group(
            query_text, set(chunks_by_model_group.keys())
        )

        # 2. Lexical BM25-style ranking
        q_tokens = set(self._tokenize_bm25(query_text))
        lexical_scored: list[tuple[VectorChunkRecord, float]] = []
        for chk in candidate_chunks:
            c_tokens = self._tokenize_bm25(chk.text)
            c_set = set(c_tokens)
            overlap = q_tokens & c_set
            # Boost exact hyphenated identifiers (e.g. payment-router-01, PCI-DSS-4.0, CVE-2026-9981)
            score = sum(2.5 if "-" in tok or any(ch.isdigit() for ch in tok) else 1.0 for tok in overlap)
            lexical_scored.append((chk, score))
        lexical_scored.sort(key=lambda x: x[1], reverse=True)

        # 3. Scale-Invariant Reciprocal Rank Fusion (RRF, k=60) across all Model Groups + BM25
        rrf_k = 60.0
        rrf_scores: dict[str, float] = {}
        channels: dict[str, list[str]] = {}
        chunk_lookup: dict[str, VectorChunkRecord] = {}

        for (model_id, dim), group_chunks in chunks_by_model_group.items():
            q_vec_native = query_vecs_by_model[(model_id, dim)]
            group_scored = [
                (
                    chk,
                    cosine_similarity(
                        q_vec_native,
                        chk.native_embedding if len(chk.native_embedding) == dim else chk.canonical_embedding,
                    ),
                )
                for chk in group_chunks
            ]
            group_scored.sort(key=lambda x: x[1], reverse=True)
            channel_tag = f"pgvector_native_{model_id}_{dim}d"
            for rank, (chk, _sim) in enumerate(group_scored[: max(top_k * 3, 15)], start=1):
                chunk_lookup[chk.chunk_id] = chk
                rrf_scores[chk.chunk_id] = rrf_scores.get(chk.chunk_id, 0.0) + (1.0 / (rrf_k + rank))
                if channel_tag not in channels.setdefault(chk.chunk_id, []):
                    channels[chk.chunk_id].append(channel_tag)

        for rank, (chk, lex_score) in enumerate(lexical_scored[: max(top_k * 4, 25)], start=1):
            if lex_score <= 0:
                continue
            chunk_lookup[chk.chunk_id] = chk
            rrf_scores[chk.chunk_id] = rrf_scores.get(chk.chunk_id, 0.0) + (1.2 / (rrf_k + rank))
            if "pg_tsvector_bm25" not in channels.setdefault(chk.chunk_id, []):
                channels[chk.chunk_id].append("pg_tsvector_bm25")

        ranked_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        return [
            RetrievedChunkContext(
                chunk_id=cid,
                kb_id=chunk_lookup[cid].kb_id,
                domain_tag=chunk_lookup[cid].domain_tag,
                doc_id=chunk_lookup[cid].doc_id,
                text=chunk_lookup[cid].text,
                rrf_score=round(rrf_scores[cid], 5),
                retrieval_channels=channels.get(cid, []),
            )
            for cid in ranked_ids
        ]

    async def search_similar_entities(
        self,
        tenant_id: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[KGEntityNode, float]]:
        scored: list[tuple[KGEntityNode, float]] = []
        for ent in self._entity_index.values():
            if ent.tenant_id != tenant_id:
                continue
            sim = cosine_similarity(query_embedding, ent.embedding)
            scored.append((ent, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class PostgresPgVectorStoreAdapter(PgVectorStorePort):
    """
    Production `asyncpg` PostgreSQL + `pgvector` Multi-KB Adapter.
    Reads heterogeneous legacy pgvector tables via `VectorKBColumnMapping`,
    tracks SHA-256 incremental watermarks in `kb_ingestion_watermarks`,
    and executes hybrid HNSW (`<=>`) + `ts_rank_cd` RRF queries.
    """

    def __init__(
        self,
        dsn: str,
        embedding_provider: EmbeddingProvider,
        canonical_dim: int = 256,
        auto_init_schema: bool = True,
    ):
        self.dsn = dsn
        self.embedding_provider = embedding_provider
        self.canonical_dim = canonical_dim
        self.auto_init_schema = auto_init_schema
        self._pool: Any = None
        self._advisory_lock_conns: dict[int, Any] = {}

    async def _get_pool(self) -> Any:
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(dsn=self.dsn, min_size=1, max_size=10)
            if self.auto_init_schema:
                async with self._pool.acquire() as conn:
                    try:
                        await conn.execute(get_production_pgvector_ddl(self.canonical_dim))
                    except Exception as exc:
                        # Non-superuser runtime roles (`NOBYPASSRLS`) rely on pre-migrated schema
                        if "InsufficientPrivilege" not in type(exc).__name__ and "permission denied" not in str(exc).lower():
                            raise
        return self._pool

    @asynccontextmanager
    async def _scoped_conn(self, tenant_id: str | None = "*") -> AsyncIterator[Any]:
        """
        Acquires a pooled connection inside a transaction and sets transaction-local
        `app.current_tenant_id` so PostgreSQL Row-Level Security policies enforce tenant
        boundaries directly in the database kernel for `NOBYPASSRLS` roles.
        """
        pool = await self._get_pool()
        scope_val = tenant_id if tenant_id else "*"
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.current_tenant_id', $1, true)",
                    scope_val,
                )
                yield conn

    async def migrate_schema_and_rls(
        self,
        app_role_name: str | None = None,
        app_role_password: str = "",
    ) -> None:
        """Idempotently applies schema tables, `acl_groups` column, RLS policies, and optional `NOBYPASSRLS` role."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(get_production_pgvector_ddl(self.canonical_dim))
            if app_role_name:
                await conn.execute(get_rls_app_role_ddl(app_role_name, app_role_password))

    async def try_acquire_advisory_lock(self, lock_key: int) -> bool:
        pool = await self._get_pool()
        conn = await pool.acquire()
        acquired = bool(await conn.fetchval("SELECT pg_try_advisory_lock($1)", int(lock_key)))
        if acquired:
            self._advisory_lock_conns[int(lock_key)] = conn
            return True
        await pool.release(conn)
        return False

    async def release_advisory_lock(self, lock_key: int) -> None:
        conn = self._advisory_lock_conns.pop(int(lock_key), None)
        if conn is not None and self._pool is not None:
            try:
                await conn.execute("SELECT pg_advisory_unlock($1)", int(lock_key))
            finally:
                await self._pool.release(conn)

    async def register_kb(self, registration: VectorKBRegistration) -> None:
        async with self._scoped_conn("*") as conn:
            existing_owner = await conn.fetchval(
                "SELECT tenant_id FROM kb_federation_registry WHERE kb_id = $1",
                registration.kb_id,
            )
            if existing_owner is not None and existing_owner != registration.tenant_id:
                raise PermissionError(
                    f"KB '{registration.kb_id}' is owned by another tenant and cannot be overwritten."
                )
        async with self._scoped_conn(registration.tenant_id) as conn:
            await conn.execute(
                """
                INSERT INTO kb_federation_registry (
                    kb_id, tenant_id, domain_tag, source_type, postgres_dsn_alias,
                    schema_name, table_name, embedding_dim, embedding_model_id,
                    column_mapping, owner_team, security_classification
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11,$12)
                ON CONFLICT (kb_id) DO UPDATE SET
                    domain_tag = EXCLUDED.domain_tag,
                    table_name = EXCLUDED.table_name,
                    embedding_dim = EXCLUDED.embedding_dim
                WHERE kb_federation_registry.tenant_id = EXCLUDED.tenant_id
                """,
                registration.kb_id,
                registration.tenant_id,
                registration.domain_tag,
                registration.source_type.value,
                registration.postgres_dsn_alias,
                registration.schema_name,
                registration.table_name,
                registration.embedding_dim,
                registration.embedding_model_id,
                json.dumps(registration.column_mapping.model_dump()),
                registration.owner_team,
                registration.security_classification,
            )

    async def unregister_kb(self, tenant_id: str, kb_id: str) -> int:
        async with self._scoped_conn("*") as conn:
            owner = await conn.fetchval(
                "SELECT tenant_id FROM kb_federation_registry WHERE kb_id = $1", kb_id
            )
            if owner is not None and owner != tenant_id:
                raise PermissionError(f"KB '{kb_id}' belongs to another tenant.")
        async with self._scoped_conn(tenant_id) as conn:
            deleted_chunks = await conn.fetchval(
                "WITH d AS (DELETE FROM unified_rag_chunks WHERE tenant_id = $1 AND kb_id = $2 RETURNING 1) SELECT count(*) FROM d",
                tenant_id,
                kb_id,
            )
            await conn.execute("DELETE FROM kb_ingestion_watermarks WHERE kb_id = $1", kb_id)
            await conn.execute(
                "DELETE FROM kb_federation_registry WHERE tenant_id = $1 AND kb_id = $2",
                tenant_id,
                kb_id,
            )
            return int(deleted_chunks or 0)

    async def delete_document(self, tenant_id: str, kb_id: str, doc_id: str) -> int:
        async with self._scoped_conn(tenant_id) as conn:
            rows = await conn.fetch(
                "DELETE FROM unified_rag_chunks WHERE tenant_id = $1 AND kb_id = $2 AND doc_id = $3 RETURNING chunk_id",
                tenant_id,
                kb_id,
                doc_id,
            )
            chunk_ids = [r["chunk_id"] for r in rows]
            if chunk_ids:
                await conn.execute(
                    "DELETE FROM kb_ingestion_watermarks WHERE kb_id = $1 AND chunk_id = ANY($2::text[])",
                    kb_id,
                    chunk_ids,
                )
            return len(chunk_ids)

    async def purge_tenant(self, tenant_id: str) -> dict[str, int]:
        async with self._scoped_conn(tenant_id) as conn:
            kbs = await conn.fetch(
                "SELECT kb_id FROM kb_federation_registry WHERE tenant_id = $1", tenant_id
            )
            kb_ids = [r["kb_id"] for r in kbs]
            chunks_removed = int(
                await conn.fetchval(
                    "WITH d AS (DELETE FROM unified_rag_chunks WHERE tenant_id = $1 RETURNING 1) SELECT count(*) FROM d",
                    tenant_id,
                )
                or 0
            )
            if kb_ids:
                await conn.execute(
                    "DELETE FROM kb_ingestion_watermarks WHERE kb_id = ANY($1::text[])", kb_ids
                )
            kbs_removed = int(
                await conn.fetchval(
                    "WITH d AS (DELETE FROM kb_federation_registry WHERE tenant_id = $1 RETURNING 1) SELECT count(*) FROM d",
                    tenant_id,
                )
                or 0
            )
            ents_removed = int(
                await conn.fetchval(
                    "WITH d AS (DELETE FROM canonical_entity_embeddings WHERE tenant_id = $1 RETURNING 1) SELECT count(*) FROM d",
                    tenant_id,
                )
                or 0
            )
            tbox_removed = int(
                await conn.fetchval(
                    "WITH d AS (DELETE FROM enterprise_tbox_registry WHERE tenant_id = $1 RETURNING 1) SELECT count(*) FROM d",
                    tenant_id,
                )
                or 0
            )
            jobs_removed = int(
                await conn.fetchval(
                    "WITH d AS (DELETE FROM kg_sync_jobs WHERE tenant_id = $1 RETURNING 1) SELECT count(*) FROM d",
                    tenant_id,
                )
                or 0
            )
            dlq_removed = int(
                await conn.fetchval(
                    "WITH d AS (DELETE FROM kg_dlq_chunks WHERE tenant_id = $1 RETURNING 1) SELECT count(*) FROM d",
                    tenant_id,
                )
                or 0
            )
            return {
                "kbs_removed": kbs_removed,
                "chunks_removed": chunks_removed,
                "entity_embeddings_removed": ents_removed,
                "tboxes_removed": tbox_removed,
                "sync_jobs_removed": jobs_removed,
                "dlq_records_removed": dlq_removed,
            }

    async def list_registered_kbs(self, tenant_id: str | None = None) -> list[VectorKBRegistration]:
        async with self._scoped_conn(tenant_id or "*") as conn:
            if tenant_id:
                rows = await conn.fetch(
                    "SELECT * FROM kb_federation_registry WHERE tenant_id = $1 ORDER BY kb_id",
                    tenant_id,
                )
            else:
                rows = await conn.fetch("SELECT * FROM kb_federation_registry ORDER BY kb_id")
            return [
                VectorKBRegistration(
                    kb_id=r["kb_id"],
                    tenant_id=r["tenant_id"],
                    domain_tag=r["domain_tag"],
                    source_type=r["source_type"],
                    postgres_dsn_alias=r["postgres_dsn_alias"],
                    schema_name=r["schema_name"],
                    table_name=r["table_name"],
                    embedding_dim=r["embedding_dim"],
                    embedding_model_id=r["embedding_model_id"],
                    column_mapping=json.loads(r["column_mapping"]),
                    owner_team=r["owner_team"],
                    security_classification=r["security_classification"],
                )
                for r in rows
            ]

    @staticmethod
    def _parse_json_safe(val: Any) -> Any:
        if isinstance(val, str):
            return json.loads(val)
        return val

    async def close(self) -> None:
        for lock_key in list(self._advisory_lock_conns.keys()):
            await self.release_advisory_lock(lock_key)
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def save_tbox(self, tbox: EnterpriseCompiledTBox) -> None:
        async with self._scoped_conn(tbox.tenant_id) as conn:
            await conn.execute(
                """
                INSERT INTO enterprise_tbox_registry (tenant_id, tbox_version, tbox_json)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (tenant_id) DO UPDATE SET
                    tbox_version = EXCLUDED.tbox_version,
                    tbox_json = EXCLUDED.tbox_json,
                    updated_at = NOW()
                """,
                tbox.tenant_id,
                tbox.version,
                tbox.model_dump_json(),
            )

    async def load_tbox(self, tenant_id: str) -> EnterpriseCompiledTBox | None:
        async with self._scoped_conn(tenant_id) as conn:
            row = await conn.fetchrow(
                "SELECT tbox_json FROM enterprise_tbox_registry WHERE tenant_id = $1",
                tenant_id,
            )
            if not row:
                return None
            raw = self._parse_json_safe(row["tbox_json"])
            return EnterpriseCompiledTBox.model_validate(raw)

    async def initialize_schema(self) -> None:
        await self.migrate_schema_and_rls()

    async def save_sync_job(self, job: SyncJobRecord) -> None:
        async with self._scoped_conn(job.tenant_id) as conn:
            await conn.execute(
                """
                INSERT INTO kg_sync_jobs (job_id, tenant_id, status, job_json)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    job_json = EXCLUDED.job_json,
                    updated_at = NOW()
                """,
                job.job_id,
                job.tenant_id,
                job.status.value,
                job.model_dump_json(),
            )

    async def get_sync_job(self, job_id: str) -> SyncJobRecord | None:
        async with self._scoped_conn("*") as conn:
            row = await conn.fetchrow(
                "SELECT job_json FROM kg_sync_jobs WHERE job_id = $1",
                job_id,
            )
            if not row:
                return None
            raw = self._parse_json_safe(row["job_json"])
            return SyncJobRecord.model_validate(raw)

    async def append_dlq_record(self, record: DLQFailedChunkRecord) -> None:
        async with self._scoped_conn(record.tenant_id) as conn:
            await conn.execute(
                """
                INSERT INTO kg_dlq_chunks (tenant_id, kb_id, chunk_id, record_json)
                VALUES ($1, $2, $3, $4::jsonb)
                """,
                record.tenant_id,
                record.kb_id,
                record.chunk_id,
                record.model_dump_json(),
            )

    async def list_dlq_records(self, tenant_id: str | None = None) -> list[DLQFailedChunkRecord]:
        async with self._scoped_conn(tenant_id or "*") as conn:
            if tenant_id is not None:
                rows = await conn.fetch(
                    "SELECT record_json FROM kg_dlq_chunks WHERE tenant_id = $1 ORDER BY dlq_id ASC",
                    tenant_id,
                )
            else:
                rows = await conn.fetch(
                    "SELECT record_json FROM kg_dlq_chunks ORDER BY dlq_id ASC"
                )
            return [
                DLQFailedChunkRecord.model_validate(self._parse_json_safe(r["record_json"]))
                for r in rows
            ]

    async def reverse_hydrate_from_legacy_pgvector_table(
        self,
        registration: VectorKBRegistration,
    ) -> list[VectorChunkRecord]:
        """
        Directly reads an existing, heterogeneous legacy PostgreSQL `pgvector` table
        (with its own schema, table name, column mapping, and native vector dimension)
        and reverse-hydrates `VectorChunkRecord`s without re-embedding native vectors.
        """
        pool = await self._get_pool()
        cm = registration.column_mapping
        # Safe identifier quoting
        schema_q = f'"{registration.schema_name}"'
        table_q = f'"{registration.table_name}"'
        sql = (
            f'SELECT "{cm.id_column}" AS cid, "{cm.doc_id_column}" AS did, '
            f'"{cm.text_column}" AS txt, "{cm.embedding_column}"::text AS emb, '
            f'"{cm.metadata_column}" AS meta '
            f"FROM {schema_q}.{table_q}"
        )
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql)

        hydrated: list[VectorChunkRecord] = []
        for r in rows:
            native_vec = self._parse_json_safe(r["emb"]) if r["emb"] else []
            meta = self._parse_json_safe(r["meta"]) if r["meta"] else {}
            text_val = str(r["txt"])
            canon_vec = self.embedding_provider.embed_text(text_val, dim=self.canonical_dim)
            hydrated.append(
                VectorChunkRecord(
                    chunk_id=f"{registration.tenant_id}::{registration.kb_id}::{r['cid']}",
                    kb_id=registration.kb_id,
                    tenant_id=registration.tenant_id,
                    domain_tag=registration.domain_tag,
                    doc_id=str(r["did"]),
                    text=text_val,
                    native_embedding=[float(x) for x in native_vec],
                    native_embedding_dim=len(native_vec) or registration.embedding_dim,
                    canonical_embedding=canon_vec,
                    metadata=meta if isinstance(meta, dict) else {},
                )
            )
        return hydrated

    async def upsert_kb_chunks(self, kb_id: str, chunks: Sequence[VectorChunkRecord]) -> int:
        if not chunks:
            return 0
        # Enforce KB ownership & cross-tenant chunk_id collision protection (even under DB superuser)
        incoming_tenants = {chk.tenant_id for chk in chunks}
        incoming_cids = [chk.chunk_id for chk in chunks]
        async with self._scoped_conn("*") as conn:
            kb_owner = await conn.fetchval(
                "SELECT tenant_id FROM kb_federation_registry WHERE kb_id = $1", kb_id
            )
            if kb_owner is not None and any(t != kb_owner for t in incoming_tenants):
                raise PermissionError(
                    f"Cannot write chunks into KB '{kb_id}' owned by another tenant."
                )
            existing_rows = await conn.fetch(
                "SELECT chunk_id, tenant_id FROM unified_rag_chunks WHERE chunk_id = ANY($1::text[])",
                incoming_cids,
            )
            existing_owners = {r["chunk_id"]: r["tenant_id"] for r in existing_rows}
            for chk in chunks:
                owner = existing_owners.get(chk.chunk_id)
                if owner is not None and owner != chk.tenant_id:
                    raise PermissionError(
                        f"Chunk '{chk.chunk_id}' belongs to another tenant and cannot be overwritten."
                    )

        # Group by tenant_id so each batch executes under its strict RLS tenant context
        by_tenant: dict[str, list[tuple[Any, ...]]] = {}
        for chk in chunks:
            if not chk.canonical_embedding:
                chk.canonical_embedding = self.embedding_provider.embed_text(
                    chk.text, dim=self.canonical_dim
                )
            vec_str = "[" + ",".join(f"{v:.6f}" for v in chk.canonical_embedding) + "]"
            meta_copy = dict(chk.metadata)
            acls = list(chk.acl_groups or ["public"])
            meta_copy["acl_groups"] = acls
            by_tenant.setdefault(chk.tenant_id, []).append(
                (
                    chk.chunk_id,
                    kb_id,
                    chk.tenant_id,
                    chk.domain_tag,
                    chk.doc_id,
                    chk.text,
                    chk.content_sha256,
                    vec_str,
                    chk.section_path,
                    json.dumps(acls),
                    json.dumps(meta_copy),
                )
            )
        for t_id, rows_to_insert in by_tenant.items():
            async with self._scoped_conn(t_id) as conn:
                await conn.executemany(
                    """
                    INSERT INTO unified_rag_chunks (
                        chunk_id, kb_id, tenant_id, domain_tag, doc_id, text,
                        content_sha256, canonical_embedding, section_path, acl_groups, metadata
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8::vector,$9,$10::jsonb,$11::jsonb)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        text = EXCLUDED.text,
                        content_sha256 = EXCLUDED.content_sha256,
                        canonical_embedding = EXCLUDED.canonical_embedding,
                        acl_groups = EXCLUDED.acl_groups,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    WHERE unified_rag_chunks.tenant_id = EXCLUDED.tenant_id
                    """,
                    rows_to_insert,
                )
        return len(chunks)

    async def fetch_kb_chunks(
        self,
        kb_id: str,
        only_unindexed_or_changed: bool = False,
        target_tbox_version: str | None = None,
        tenant_id: str | None = None,
    ) -> list[VectorChunkRecord]:
        async with self._scoped_conn(tenant_id or "*") as conn:
            if only_unindexed_or_changed:
                rows = await conn.fetch(
                    """
                    SELECT c.chunk_id, c.kb_id, c.tenant_id, c.domain_tag, c.doc_id,
                           c.text, c.content_sha256, c.canonical_embedding::text AS emb_txt,
                           c.section_path, c.acl_groups, c.metadata, c.updated_at
                    FROM unified_rag_chunks c
                    LEFT JOIN kb_ingestion_watermarks w
                      ON c.kb_id = w.kb_id AND c.chunk_id = w.chunk_id
                    WHERE c.kb_id = $1
                      AND ($3::text IS NULL OR c.tenant_id = $3)
                      AND (w.chunk_id IS NULL OR w.content_sha256 != c.content_sha256
                           OR ($2::text IS NOT NULL AND w.tbox_version != $2))
                    """,
                    kb_id,
                    target_tbox_version,
                    tenant_id,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT chunk_id, kb_id, tenant_id, domain_tag, doc_id,
                           text, content_sha256, canonical_embedding::text AS emb_txt,
                           section_path, acl_groups, metadata, updated_at
                    FROM unified_rag_chunks
                    WHERE kb_id = $1
                      AND ($2::text IS NULL OR tenant_id = $2)
                    """,
                    kb_id,
                    tenant_id,
                )
            out: list[VectorChunkRecord] = []
            for r in rows:
                meta = self._parse_json_safe(r["metadata"]) if r["metadata"] else {}
                raw_acls = self._parse_json_safe(r["acl_groups"]) if r["acl_groups"] else None
                if not isinstance(raw_acls, list) or not raw_acls:
                    raw_acls = meta.get("acl_groups", ["public"]) if isinstance(meta, dict) else ["public"]
                out.append(
                    VectorChunkRecord(
                        chunk_id=r["chunk_id"],
                        kb_id=r["kb_id"],
                        tenant_id=r["tenant_id"],
                        domain_tag=r["domain_tag"],
                        doc_id=r["doc_id"],
                        text=r["text"],
                        content_sha256=r["content_sha256"],
                        canonical_embedding=self._parse_json_safe(r["emb_txt"]),
                        section_path=r["section_path"],
                        acl_groups=[str(g) for g in raw_acls],
                        metadata=meta if isinstance(meta, dict) else {},
                        updated_at=r["updated_at"].isoformat() if r["updated_at"] else utc_now_iso(),
                    )
                )
            return out

    async def mark_chunks_watermarked(self, watermarks: Sequence[KBWatermarkState]) -> None:
        pool = await self._get_pool()
        rows = [(w.kb_id, w.chunk_id, w.content_sha256, w.tbox_version) for w in watermarks]
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO kb_ingestion_watermarks (kb_id, chunk_id, content_sha256, tbox_version)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (kb_id, chunk_id) DO UPDATE SET
                    content_sha256 = EXCLUDED.content_sha256,
                    tbox_version = EXCLUDED.tbox_version,
                    last_indexed_at = NOW()
                """,
                rows,
            )

    async def upsert_canonical_entity_embeddings(self, entities: Sequence[KGEntityNode]) -> None:
        if not entities:
            return
        by_tenant: dict[str, list[tuple[Any, ...]]] = {}
        for ent in entities:
            if not ent.embedding:
                ent.embedding = self.embedding_provider.embed_text(
                    f"{ent.ontology_class} {ent.canonical_name} {ent.summary}",
                    dim=self.canonical_dim,
                )
            vec_str = "[" + ",".join(f"{v:.6f}" for v in ent.embedding) + "]"
            by_tenant.setdefault(ent.tenant_id, []).append(
                (
                    ent.entity_id,
                    ent.tenant_id,
                    ent.canonical_name,
                    ent.ontology_class,
                    ent.upper_category.value,
                    json.dumps(ent.source_kb_ids),
                    vec_str,
                )
            )
        for t_id, rows in by_tenant.items():
            async with self._scoped_conn(t_id) as conn:
                await conn.executemany(
                    """
                    INSERT INTO canonical_entity_embeddings (
                        entity_id, tenant_id, canonical_name, ontology_class,
                        upper_category, source_kb_ids, embedding
                    ) VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7::vector)
                    ON CONFLICT (entity_id) DO UPDATE SET
                        source_kb_ids = EXCLUDED.source_kb_ids,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW()
                    """,
                    rows,
                )

    async def hybrid_search_chunks(
        self,
        tenant_id: str,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 6,
        kb_filter: list[str] | None = None,
        domain_filter: list[str] | None = None,
        caller_acl_groups: list[str] | None = None,
    ) -> list[RetrievedChunkContext]:
        vec_str = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"
        async with self._scoped_conn(tenant_id) as conn:
            rows = await conn.fetch(
                """
                WITH dense AS (
                    SELECT chunk_id, kb_id, domain_tag, doc_id, text, acl_groups, metadata,
                           ROW_NUMBER() OVER (ORDER BY canonical_embedding <=> $2::vector) AS dense_rank
                    FROM unified_rag_chunks
                    WHERE tenant_id = $1
                      AND ($5::text[] IS NULL OR kb_id = ANY($5::text[]))
                      AND ($6::text[] IS NULL OR domain_tag = ANY($6::text[]))
                      AND ($8::text[] IS NULL OR acl_groups ? 'public' OR acl_groups ?| $8::text[])
                    ORDER BY canonical_embedding <=> $2::vector
                    LIMIT $4
                ),
                lexical AS (
                    SELECT chunk_id, kb_id, domain_tag, doc_id, text, acl_groups, metadata,
                           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $3)) DESC) AS lex_rank
                    FROM unified_rag_chunks
                    WHERE tenant_id = $1
                      AND ($5::text[] IS NULL OR kb_id = ANY($5::text[]))
                      AND ($6::text[] IS NULL OR domain_tag = ANY($6::text[]))
                      AND ($8::text[] IS NULL OR acl_groups ? 'public' OR acl_groups ?| $8::text[])
                      AND tsv @@ plainto_tsquery('english', $3)
                    ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $3)) DESC
                    LIMIT $4
                ),
                fused AS (
                    SELECT COALESCE(d.chunk_id, l.chunk_id) AS chunk_id,
                           COALESCE(d.kb_id, l.kb_id) AS kb_id,
                           COALESCE(d.domain_tag, l.domain_tag) AS domain_tag,
                           COALESCE(d.doc_id, l.doc_id) AS doc_id,
                           COALESCE(d.text, l.text) AS text,
                           COALESCE(d.acl_groups, l.acl_groups) AS acl_groups,
                           COALESCE(d.metadata, l.metadata) AS metadata,
                           COALESCE(1.0 / (60.0 + d.dense_rank), 0.0)
                             + COALESCE(1.2 / (60.0 + l.lex_rank), 0.0) AS rrf_score,
                           (d.dense_rank IS NOT NULL) AS hit_dense,
                           (l.lex_rank IS NOT NULL) AS hit_lex
                    FROM dense d
                    FULL OUTER JOIN lexical l ON d.chunk_id = l.chunk_id
                )
                SELECT * FROM fused
                ORDER BY rrf_score DESC
                LIMIT $7
                """,
                tenant_id,
                vec_str,
                query_text,
                max(top_k * 4, 25),
                kb_filter,
                domain_filter,
                top_k,
                list(caller_acl_groups) if caller_acl_groups is not None else None,
            )
            caller_acls = {g.lower() for g in caller_acl_groups} if caller_acl_groups is not None else None
            results: list[RetrievedChunkContext] = []
            for r in rows:
                raw_acls = self._parse_json_safe(r["acl_groups"]) if r["acl_groups"] else None
                meta = self._parse_json_safe(r["metadata"]) if r["metadata"] else {}
                chk_acls = (
                    raw_acls
                    if isinstance(raw_acls, list) and raw_acls
                    else (meta.get("acl_groups", ["public"]) if isinstance(meta, dict) else ["public"])
                )
                if caller_acls is not None:
                    chk_set = {str(g).lower() for g in chk_acls}
                    if "public" not in chk_set and not caller_acls.intersection(chk_set):
                        continue
                chans = []
                if r["hit_dense"]:
                    chans.append("pgvector_hnsw_dense")
                if r["hit_lex"]:
                    chans.append("pg_tsvector_bm25")
                results.append(
                    RetrievedChunkContext(
                        chunk_id=r["chunk_id"],
                        kb_id=r["kb_id"],
                        domain_tag=r["domain_tag"],
                        doc_id=r["doc_id"],
                        text=r["text"],
                        rrf_score=round(float(r["rrf_score"]), 5),
                        retrieval_channels=chans,
                        acl_groups=[str(g) for g in chk_acls],
                    )
                )
                if len(results) >= top_k:
                    break
            return results

    async def search_similar_entities(
        self,
        tenant_id: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[KGEntityNode, float]]:
        from ..models import UpperOntologyCategory

        vec_str = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"
        async with self._scoped_conn(tenant_id) as conn:
            rows = await conn.fetch(
                """
                SELECT entity_id, tenant_id, canonical_name, ontology_class,
                       upper_category, source_kb_ids,
                       (1.0 - (embedding <=> $2::vector)) AS cosine_sim
                FROM canonical_entity_embeddings
                WHERE tenant_id = $1
                ORDER BY embedding <=> $2::vector
                LIMIT $3
                """,
                tenant_id,
                vec_str,
                top_k,
            )
            out: list[tuple[KGEntityNode, float]] = []
            for r in rows:
                node = KGEntityNode(
                    entity_id=r["entity_id"],
                    tenant_id=r["tenant_id"],
                    canonical_name=r["canonical_name"],
                    ontology_class=r["ontology_class"],
                    upper_category=UpperOntologyCategory(r["upper_category"]),
                    source_kb_ids=self._parse_json_safe(r["source_kb_ids"]),
                )
                out.append((node, float(r["cosine_sim"])))
            return out
