from __future__ import annotations

import fnmatch
import json
import os
import re
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Sequence

from ..models import (
    CommunitySummaryRecord,
    DLQFailedChunkRecord,
    EnterpriseCompiledTBox,
    KBSourceType,
    KBWatermarkState,
    KGEntityNode,
    RetrievedChunkContext,
    SyncJobRecord,
    SyncJobStatus,
    VectorChunkRecord,
    VectorKBRegistration,
    utc_now_iso,
)
from ..vector_clustering import (
    EmbeddingProvider,
    FederatedEmbeddingModelRouter,
    cosine_similarity,
)


_VALID_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_FORBIDDEN_LEGACY_SCHEMAS = {"pg_catalog", "information_schema", "pg_toast"}
_FORBIDDEN_LEGACY_TABLES = {
    "unified_rag_chunks",
    "kb_federation_registry",
    "kb_ingestion_watermarks",
    "canonical_entity_embeddings",
    "enterprise_tbox_registry",
    "kg_sync_jobs",
    "kg_dlq_chunks",
    "kg_community_summaries",
}


def is_job_heartbeat_stale(job: SyncJobRecord, stale_timeout_sec: float = 45.0) -> bool:
    """Returns True only when a QUEUED/RUNNING job's last heartbeat (or created_at) exceeds `stale_timeout_sec`."""
    ref_iso = job.heartbeat_at or job.created_at
    if not ref_iso:
        return False
    try:
        dt = datetime.fromisoformat(ref_iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        return age > stale_timeout_sec
    except Exception:
        return False


def validate_legacy_table_access(
    schema_name: str,
    table_name: str,
    column_names: Sequence[str],
    allowed_patterns: Sequence[str] | None = None,
    enforce_allowlist: bool = True,
) -> None:
    """
    Fail-closed guard preventing arbitrary SQL table exfiltration via legacy KB registration:
      1. Rejects invalid SQL identifiers (prevents quoting/injection).
      2. Unconditionally blocks PostgreSQL system catalogs & all internal KG Fabric tables.
      3. When `enforce_allowlist=True`, requires `schema.table` (or `table`) to match an explicit
         operator-configured allowlist (`KG_FABRIC_ALLOWED_LEGACY_TABLES` / `allowed_legacy_tables`).
    """
    for ident in [schema_name, table_name, *column_names]:
        if not _VALID_SQL_IDENT_RE.match(str(ident)):
            raise PermissionError(f"Invalid SQL identifier in legacy KB mapping: '{ident}'")

    s_low = schema_name.lower()
    t_low = table_name.lower()
    if s_low in _FORBIDDEN_LEGACY_SCHEMAS or s_low.startswith("pg_temp"):
        raise PermissionError(
            f"Access to PostgreSQL system schema '{schema_name}' is forbidden for legacy KB hydration."
        )
    if t_low in _FORBIDDEN_LEGACY_TABLES:
        raise PermissionError(
            f"Access to internal KG Fabric table '{table_name}' is forbidden for legacy KB hydration."
        )

    if enforce_allowlist:
        patterns = [p.strip().lower() for p in (allowed_patterns or []) if p.strip()]
        if not patterns:
            raise PermissionError(
                "Legacy PostgreSQL table hydration is disabled by default. Configure "
                "KG_FABRIC_ALLOWED_LEGACY_TABLES (or PostgresPgVectorStoreAdapter(allowed_legacy_tables=...)) "
                f"to explicitly allow '{schema_name}.{table_name}'."
            )
        qualified = f"{s_low}.{t_low}"
        if not any(
            fnmatch.fnmatch(qualified, pat) or fnmatch.fnmatch(t_low, pat)
            for pat in patterns
        ):
            raise PermissionError(
                f"Legacy table '{schema_name}.{table_name}' is not in KG_FABRIC_ALLOWED_LEGACY_TABLES."
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
    entity_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    ontology_class TEXT NOT NULL,
    upper_category TEXT NOT NULL,
    source_kb_ids JSONB NOT NULL,
    embedding vector({canonical_dim}) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, entity_id)
);

-- Idempotently upgrade legacy single-column PK (entity_id) -> composite PK (tenant_id, entity_id)
DO $$
DECLARE
    pk_cols INTEGER;
BEGIN
    SELECT array_length(conkey, 1) INTO pk_cols
    FROM pg_constraint
    WHERE conrelid = 'canonical_entity_embeddings'::regclass AND contype = 'p';
    IF pk_cols = 1 THEN
        ALTER TABLE canonical_entity_embeddings DROP CONSTRAINT canonical_entity_embeddings_pkey;
        ALTER TABLE canonical_entity_embeddings ADD PRIMARY KEY (tenant_id, entity_id);
    END IF;
END
$$;

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
    heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE kg_sync_jobs
    ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE INDEX IF NOT EXISTS idx_kg_sync_jobs_tenant
    ON kg_sync_jobs (tenant_id, updated_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS uq_kg_sync_jobs_active_tenant
    ON kg_sync_jobs (tenant_id)
    WHERE status IN ('QUEUED', 'RUNNING');

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

CREATE TABLE IF NOT EXISTS kg_community_summaries (
    tenant_id TEXT NOT NULL,
    community_id TEXT NOT NULL,
    summary_json JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, community_id)
);

-- Fail-Closed Row-Level Security (RLS) across all multi-tenant tables
-- Enforced automatically for non-superuser app roles (`NOBYPASSRLS`) via `SET LOCAL app.current_tenant_id`
ALTER TABLE kb_federation_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE unified_rag_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE canonical_entity_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE enterprise_tbox_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE kg_sync_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE kg_dlq_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE kg_community_summaries ENABLE ROW LEVEL SECURITY;

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

DROP POLICY IF EXISTS rls_community_summaries_tenant ON kg_community_summaries;
CREATE POLICY rls_community_summaries_tenant ON kg_community_summaries
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

    async def try_enqueue_sync_job(
        self, job: SyncJobRecord, stale_timeout_sec: float = 45.0
    ) -> bool:
        await self.save_sync_job(job)
        return True

    async def get_sync_job(self, job_id: str) -> SyncJobRecord | None:
        return None

    async def append_dlq_record(self, record: DLQFailedChunkRecord) -> None:
        return None

    async def list_dlq_records(self, tenant_id: str | None = None) -> list[DLQFailedChunkRecord]:
        return []

    async def list_sync_jobs(
        self, tenant_id: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[SyncJobRecord]:
        return []

    async def list_kb_documents(
        self,
        tenant_id: str,
        kb_id: str,
        limit: int = 50,
        offset: int = 0,
        caller_acl_groups: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        return []

    async def save_community_summaries(
        self, tenant_id: str, summaries: Sequence[CommunitySummaryRecord]
    ) -> None:
        return None

    async def load_community_summaries(self, tenant_id: str) -> list[CommunitySummaryRecord]:
        return []

    async def get_legacy_sync_cursor(self, kb_id: str) -> str | None:
        return None

    async def set_legacy_sync_cursor(self, kb_id: str, cursor_iso: str) -> None:
        return None

    async def try_acquire_advisory_lock(self, lock_key: int) -> bool:
        return True

    async def release_advisory_lock(self, lock_key: int) -> None:
        return None

    async def unregister_kb(self, tenant_id: str, kb_id: str) -> int:
        return len(await self.unregister_kb_chunks(tenant_id, kb_id))

    async def unregister_kb_chunks(self, tenant_id: str, kb_id: str) -> list[str]:
        return []

    async def delete_document(self, tenant_id: str, kb_id: str, doc_id: str) -> int:
        return len(await self.delete_document_chunks(tenant_id, kb_id, doc_id))

    async def delete_document_chunks(self, tenant_id: str, kb_id: str, doc_id: str) -> list[str]:
        return []

    async def delete_canonical_entity_embeddings(
        self, tenant_id: str, entity_ids: Sequence[str]
    ) -> int:
        return 0

    async def fetch_tenant_chunk_acl_map(self, tenant_id: str) -> dict[str, list[str]]:
        return {}

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
        self._entity_index: dict[tuple[str, str], KGEntityNode] = {}
        self._tboxes: dict[str, EnterpriseCompiledTBox] = {}
        self._sync_jobs: dict[str, SyncJobRecord] = {}
        self._dlq_records: list[DLQFailedChunkRecord] = []
        self._communities: dict[str, list[CommunitySummaryRecord]] = {}
        self._legacy_cursors: dict[str, str] = {}
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
        self._sync_jobs[job.job_id] = job.model_copy(deep=True)

    async def try_enqueue_sync_job(
        self, job: SyncJobRecord, stale_timeout_sec: float = 45.0
    ) -> bool:
        for existing in self._sync_jobs.values():
            if existing.tenant_id != job.tenant_id:
                continue
            if existing.status in (SyncJobStatus.QUEUED, SyncJobStatus.RUNNING):
                if is_job_heartbeat_stale(existing, stale_timeout_sec):
                    existing.status = SyncJobStatus.FAILED
                    existing.error = "Worker heartbeat expired / interrupted by restart"
                    existing.completed_at = utc_now_iso()
                else:
                    return False
        self._sync_jobs[job.job_id] = job.model_copy(deep=True)
        return True

    async def get_sync_job(self, job_id: str) -> SyncJobRecord | None:
        rec = self._sync_jobs.get(job_id)
        return rec.model_copy(deep=True) if rec is not None else None

    async def append_dlq_record(self, record: DLQFailedChunkRecord) -> None:
        self._dlq_records.append(record)

    async def list_dlq_records(self, tenant_id: str | None = None) -> list[DLQFailedChunkRecord]:
        if tenant_id is None:
            return list(self._dlq_records)
        return [r for r in self._dlq_records if r.tenant_id == tenant_id]

    async def list_sync_jobs(
        self, tenant_id: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[SyncJobRecord]:
        jobs = list(self._sync_jobs.values())
        if tenant_id is not None:
            jobs = [j for j in jobs if j.tenant_id == tenant_id]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[offset : offset + limit]

    async def list_kb_documents(
        self,
        tenant_id: str,
        kb_id: str,
        limit: int = 50,
        offset: int = 0,
        caller_acl_groups: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        caller_set = {g.lower() for g in caller_acl_groups} if caller_acl_groups is not None else None
        chunks = []
        for c in self._kb_tables.get(kb_id, {}).values():
            if c.tenant_id != tenant_id:
                continue
            if caller_set is not None:
                c_acls = {g.lower() for g in (c.acl_groups or ["public"])}
                if "public" not in c_acls and not caller_set.intersection(c_acls):
                    continue
            chunks.append(c)
        by_doc: dict[str, dict[str, Any]] = {}
        for c in chunks:
            entry = by_doc.setdefault(
                c.doc_id,
                {
                    "doc_id": c.doc_id,
                    "kb_id": kb_id,
                    "tenant_id": tenant_id,
                    "chunk_count": 0,
                    "acl_groups": set(),
                    "title": c.metadata.get("title", c.doc_id),
                    "updated_at": c.updated_at,
                },
            )
            entry["chunk_count"] += 1
            entry["acl_groups"].update(c.acl_groups)
            if c.updated_at > entry["updated_at"]:
                entry["updated_at"] = c.updated_at
        docs = [
            {**d, "acl_groups": sorted(d["acl_groups"])}
            for d in sorted(by_doc.values(), key=lambda x: x["doc_id"])
        ]
        return docs[offset : offset + limit]

    async def save_community_summaries(
        self, tenant_id: str, summaries: Sequence[CommunitySummaryRecord]
    ) -> None:
        self._communities[tenant_id] = list(summaries)

    async def load_community_summaries(self, tenant_id: str) -> list[CommunitySummaryRecord]:
        return list(self._communities.get(tenant_id, []))

    async def get_legacy_sync_cursor(self, kb_id: str) -> str | None:
        return self._legacy_cursors.get(kb_id)

    async def set_legacy_sync_cursor(self, kb_id: str, cursor_iso: str) -> None:
        self._legacy_cursors[kb_id] = cursor_iso

    async def register_kb(self, registration: VectorKBRegistration) -> None:
        existing = self._kb_registry.get(registration.kb_id)
        if existing is not None and existing.tenant_id != registration.tenant_id:
            raise PermissionError(
                f"KB '{registration.kb_id}' is owned by another tenant and cannot be overwritten."
            )
        self._kb_registry[registration.kb_id] = registration
        self._kb_tables.setdefault(registration.kb_id, {})

    async def unregister_kb_chunks(self, tenant_id: str, kb_id: str) -> list[str]:
        existing = self._kb_registry.get(kb_id)
        if existing is None:
            return []
        if existing.tenant_id != tenant_id:
            raise PermissionError(f"KB '{kb_id}' belongs to another tenant.")
        del self._kb_registry[kb_id]
        removed_ids = list(self._kb_tables.pop(kb_id, {}).keys())
        for key in [k for k in self._watermarks if k[0] == kb_id]:
            del self._watermarks[key]
        return removed_ids

    async def delete_document_chunks(self, tenant_id: str, kb_id: str, doc_id: str) -> list[str]:
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
        return to_delete

    async def delete_canonical_entity_embeddings(
        self, tenant_id: str, entity_ids: Sequence[str]
    ) -> int:
        removed = 0
        for eid in entity_ids:
            if self._entity_index.pop((tenant_id, eid), None) is not None:
                removed += 1
        return removed

    async def fetch_tenant_chunk_acl_map(self, tenant_id: str) -> dict[str, list[str]]:
        acl_map: dict[str, list[str]] = {}
        for kb_chunks in self._kb_tables.values():
            for cid, chk in kb_chunks.items():
                if chk.tenant_id == tenant_id:
                    acl_map[cid] = list(chk.acl_groups or ["public"])
        return acl_map

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
            self._entity_index[(ent.tenant_id, ent.entity_id)] = ent

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
        allowed_legacy_tables: Sequence[str] | None = None,
        model_router: FederatedEmbeddingModelRouter | None = None,
        dsn_aliases: dict[str, str] | None = None,
    ):
        self.dsn = dsn
        self.embedding_provider = embedding_provider
        self.canonical_dim = canonical_dim
        self.auto_init_schema = auto_init_schema
        self.model_router = model_router or FederatedEmbeddingModelRouter(embedding_provider)
        if allowed_legacy_tables is not None:
            self.allowed_legacy_tables = [p.strip() for p in allowed_legacy_tables if p.strip()]
        else:
            raw_env = os.getenv("KG_FABRIC_ALLOWED_LEGACY_TABLES", "").strip()
            self.allowed_legacy_tables = [p.strip() for p in raw_env.split(",") if p.strip()]
        if dsn_aliases is not None:
            self.dsn_aliases = dict(dsn_aliases)
        else:
            raw_aliases = os.getenv("KG_FABRIC_PG_DSN_ALIASES", "").strip()
            if raw_aliases.startswith("{"):
                parsed = json.loads(raw_aliases)
                self.dsn_aliases = {str(k): str(v) for k, v in parsed.items()} if isinstance(parsed, dict) else {}
            else:
                self.dsn_aliases = {}
        self._pool: Any = None
        self._alias_pools: dict[str, Any] = {}
        self._advisory_lock_conns: dict[int, Any] = {}

    async def _get_pool_for_alias(self, alias: str | None = None) -> Any:
        key = (alias or "primary").strip()
        if not key or key == "primary" or key not in self.dsn_aliases:
            return await self._get_pool()
        pool = self._alias_pools.get(key)
        if pool is None:
            import asyncpg

            pool = await asyncpg.create_pool(dsn=self.dsn_aliases[key], min_size=1, max_size=6)
            self._alias_pools[key] = pool
        return pool

    @asynccontextmanager
    async def _scoped_alias_conn(
        self, alias: str | None, tenant_id: str | None = "*"
    ) -> AsyncIterator[Any]:
        pool = await self._get_pool_for_alias(alias)
        scope_val = tenant_id if tenant_id else "*"
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.current_tenant_id', $1, true)",
                    scope_val,
                )
                yield conn

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
                    source_type = EXCLUDED.source_type,
                    schema_name = EXCLUDED.schema_name,
                    table_name = EXCLUDED.table_name,
                    embedding_dim = EXCLUDED.embedding_dim,
                    embedding_model_id = EXCLUDED.embedding_model_id,
                    column_mapping = EXCLUDED.column_mapping
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

    async def unregister_kb_chunks(self, tenant_id: str, kb_id: str) -> list[str]:
        async with self._scoped_conn("*") as conn:
            owner = await conn.fetchval(
                "SELECT tenant_id FROM kb_federation_registry WHERE kb_id = $1", kb_id
            )
            if owner is not None and owner != tenant_id:
                raise PermissionError(f"KB '{kb_id}' belongs to another tenant.")
        async with self._scoped_conn(tenant_id) as conn:
            rows = await conn.fetch(
                "DELETE FROM unified_rag_chunks WHERE tenant_id = $1 AND kb_id = $2 RETURNING chunk_id",
                tenant_id,
                kb_id,
            )
            chunk_ids = [r["chunk_id"] for r in rows]
            await conn.execute("DELETE FROM kb_ingestion_watermarks WHERE kb_id = $1", kb_id)
            await conn.execute(
                "DELETE FROM kb_federation_registry WHERE tenant_id = $1 AND kb_id = $2",
                tenant_id,
                kb_id,
            )
            return chunk_ids

    async def delete_document_chunks(self, tenant_id: str, kb_id: str, doc_id: str) -> list[str]:
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
            return chunk_ids

    async def delete_canonical_entity_embeddings(
        self, tenant_id: str, entity_ids: Sequence[str]
    ) -> int:
        if not entity_ids:
            return 0
        async with self._scoped_conn(tenant_id) as conn:
            removed = await conn.fetchval(
                "WITH d AS (DELETE FROM canonical_entity_embeddings WHERE tenant_id = $1 AND entity_id = ANY($2::text[]) RETURNING 1) SELECT count(*) FROM d",
                tenant_id,
                list(entity_ids),
            )
            return int(removed or 0)

    async def fetch_tenant_chunk_acl_map(self, tenant_id: str) -> dict[str, list[str]]:
        async with self._scoped_conn(tenant_id) as conn:
            rows = await conn.fetch(
                "SELECT chunk_id, acl_groups FROM unified_rag_chunks WHERE tenant_id = $1",
                tenant_id,
            )
            acl_map: dict[str, list[str]] = {}
            for r in rows:
                raw_acls = r["acl_groups"]
                if isinstance(raw_acls, str):
                    parsed = json.loads(raw_acls)
                elif isinstance(raw_acls, list):
                    parsed = raw_acls
                else:
                    parsed = ["public"]
                acl_map[str(r["chunk_id"])] = [str(g) for g in parsed] if parsed else ["public"]
            return acl_map

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
            await conn.execute(
                "DELETE FROM kg_community_summaries WHERE tenant_id = $1",
                tenant_id,
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
        for a_pool in list(self._alias_pools.values()):
            try:
                await a_pool.close()
            except Exception:
                pass
        self._alias_pools.clear()

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

    @staticmethod
    def _parse_iso_dt(raw_iso: str | None) -> datetime:
        if raw_iso:
            try:
                dt = datetime.fromisoformat(raw_iso.replace("Z", "+00:00"))
                return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
            except Exception:
                pass
        return datetime.now(timezone.utc)

    async def save_sync_job(self, job: SyncJobRecord) -> None:
        hb_dt = self._parse_iso_dt(job.heartbeat_at or job.created_at)
        async with self._scoped_conn(job.tenant_id) as conn:
            await conn.execute(
                """
                INSERT INTO kg_sync_jobs (job_id, tenant_id, status, job_json, heartbeat_at, updated_at)
                VALUES ($1, $2, $3, $4::jsonb, $5, NOW())
                ON CONFLICT (job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    job_json = EXCLUDED.job_json,
                    heartbeat_at = EXCLUDED.heartbeat_at,
                    updated_at = NOW()
                """,
                job.job_id,
                job.tenant_id,
                job.status.value,
                job.model_dump_json(),
                hb_dt,
            )

    async def try_enqueue_sync_job(
        self, job: SyncJobRecord, stale_timeout_sec: float = 45.0
    ) -> bool:
        """
        Cross-replica & concurrent-request safe atomic enqueue:
          1. Expires any genuinely dead QUEUED/RUNNING job for `tenant_id` whose `heartbeat_at`
             exceeds `stale_timeout_sec`.
          2. Atomically inserts `job` relying on `uq_kg_sync_jobs_active_tenant` partial unique
             index (`WHERE status IN ('QUEUED', 'RUNNING')`) so concurrent requests across pods
             cannot both enqueue an active job for the same tenant!
        """
        hb_dt = self._parse_iso_dt(job.heartbeat_at or job.created_at)
        async with self._scoped_conn(job.tenant_id) as conn:
            await conn.execute(
                """
                UPDATE kg_sync_jobs
                SET status = 'FAILED',
                    job_json = jsonb_set(
                        jsonb_set(job_json, '{status}', '"FAILED"'),
                        '{error}', '"Worker heartbeat expired / interrupted by restart"'
                    ),
                    updated_at = NOW()
                WHERE tenant_id = $1
                  AND status IN ('QUEUED', 'RUNNING')
                  AND heartbeat_at < NOW() - ($2 * INTERVAL '1 second')
                """,
                job.tenant_id,
                float(stale_timeout_sec),
            )
            inserted = await conn.fetchval(
                """
                INSERT INTO kg_sync_jobs (job_id, tenant_id, status, job_json, heartbeat_at, updated_at)
                VALUES ($1, $2, $3, $4::jsonb, $5, NOW())
                ON CONFLICT (tenant_id) WHERE status IN ('QUEUED', 'RUNNING') DO NOTHING
                RETURNING job_id
                """,
                job.job_id,
                job.tenant_id,
                job.status.value,
                job.model_dump_json(),
                hb_dt,
            )
            return inserted is not None

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

    async def list_sync_jobs(
        self, tenant_id: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[SyncJobRecord]:
        async with self._scoped_conn(tenant_id or "*") as conn:
            if tenant_id is not None:
                rows = await conn.fetch(
                    "SELECT job_json FROM kg_sync_jobs WHERE tenant_id = $1 ORDER BY updated_at DESC LIMIT $2 OFFSET $3",
                    tenant_id,
                    limit,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    "SELECT job_json FROM kg_sync_jobs ORDER BY updated_at DESC LIMIT $1 OFFSET $2",
                    limit,
                    offset,
                )
            return [
                SyncJobRecord.model_validate(self._parse_json_safe(r["job_json"]))
                for r in rows
            ]

    async def list_kb_documents(
        self,
        tenant_id: str,
        kb_id: str,
        limit: int = 50,
        offset: int = 0,
        caller_acl_groups: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        async with self._scoped_conn(tenant_id) as conn:
            rows = await conn.fetch(
                """
                SELECT doc_id,
                       count(*) AS chunk_count,
                       MAX(updated_at) AS last_updated,
                       MIN(metadata->>'title') AS title
                FROM unified_rag_chunks
                WHERE tenant_id = $1
                  AND kb_id = $2
                  AND ($5::text[] IS NULL OR acl_groups ? 'public' OR acl_groups ?| $5::text[])
                GROUP BY doc_id
                ORDER BY doc_id ASC
                LIMIT $3 OFFSET $4
                """,
                tenant_id,
                kb_id,
                limit,
                offset,
                list(caller_acl_groups) if caller_acl_groups is not None else None,
            )
            return [
                {
                    "doc_id": str(r["doc_id"]),
                    "kb_id": kb_id,
                    "tenant_id": tenant_id,
                    "chunk_count": int(r["chunk_count"]),
                    "title": r["title"] or str(r["doc_id"]),
                    "updated_at": r["last_updated"].isoformat() if r["last_updated"] else utc_now_iso(),
                }
                for r in rows
            ]

    async def save_community_summaries(
        self, tenant_id: str, summaries: Sequence[CommunitySummaryRecord]
    ) -> None:
        async with self._scoped_conn(tenant_id) as conn:
            await conn.execute(
                "DELETE FROM kg_community_summaries WHERE tenant_id = $1",
                tenant_id,
            )
            if summaries:
                rows = [
                    (tenant_id, s.community_id, s.model_dump_json())
                    for s in summaries
                ]
                await conn.executemany(
                    """
                    INSERT INTO kg_community_summaries (tenant_id, community_id, summary_json)
                    VALUES ($1, $2, $3::jsonb)
                    ON CONFLICT (tenant_id, community_id) DO UPDATE SET
                        summary_json = EXCLUDED.summary_json,
                        updated_at = NOW()
                    """,
                    rows,
                )

    async def load_community_summaries(self, tenant_id: str) -> list[CommunitySummaryRecord]:
        async with self._scoped_conn(tenant_id) as conn:
            rows = await conn.fetch(
                "SELECT summary_json FROM kg_community_summaries WHERE tenant_id = $1 ORDER BY community_id ASC",
                tenant_id,
            )
            return [
                CommunitySummaryRecord.model_validate(self._parse_json_safe(r["summary_json"]))
                for r in rows
            ]

    async def get_legacy_sync_cursor(self, kb_id: str) -> str | None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            val = await conn.fetchval(
                "SELECT content_sha256 FROM kb_ingestion_watermarks WHERE kb_id = $1 AND chunk_id = '__legacy_cursor__'",
                kb_id,
            )
            return str(val) if val else None

    async def set_legacy_sync_cursor(self, kb_id: str, cursor_iso: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO kb_ingestion_watermarks (kb_id, chunk_id, content_sha256, tbox_version)
                VALUES ($1, '__legacy_cursor__', $2, 'cursor')
                ON CONFLICT (kb_id, chunk_id) DO UPDATE SET
                    content_sha256 = EXCLUDED.content_sha256,
                    last_indexed_at = NOW()
                """,
                kb_id,
                cursor_iso,
            )

    async def reverse_hydrate_from_legacy_pgvector_table(
        self,
        registration: VectorKBRegistration,
        *,
        enforce_allowlist: bool = True,
        since_updated_at: str | None = None,
    ) -> list[VectorChunkRecord]:
        """
        Directly reads an explicitly allowlisted legacy PostgreSQL `pgvector` table
        (supporting multi-DB `postgres_dsn_alias` and incremental `updated_at_column` cursors)
        and reverse-hydrates `VectorChunkRecord`s inside a tenant-scoped transaction.
        """
        cm = registration.column_mapping
        col_list = [
            cm.id_column,
            cm.doc_id_column,
            cm.text_column,
            cm.embedding_column,
            cm.metadata_column,
        ]
        if cm.updated_at_column:
            col_list.append(cm.updated_at_column)
        validate_legacy_table_access(
            schema_name=registration.schema_name,
            table_name=registration.table_name,
            column_names=col_list,
            allowed_patterns=self.allowed_legacy_tables,
            enforce_allowlist=enforce_allowlist,
        )
        schema_q = f'"{registration.schema_name}"'
        table_q = f'"{registration.table_name}"'
        core_sql = (
            f'SELECT "{cm.id_column}" AS cid, "{cm.doc_id_column}" AS did, '
            f'"{cm.text_column}" AS txt, "{cm.embedding_column}"::text AS emb, '
            f'"{cm.metadata_column}" AS meta'
        )
        has_upd_col = False
        async with self._scoped_alias_conn(
            registration.postgres_dsn_alias, registration.tenant_id
        ) as conn:
            if cm.updated_at_column:
                has_upd_col = bool(
                    await conn.fetchval(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_schema = $1 AND table_name = $2 AND column_name = $3",
                        registration.schema_name,
                        registration.table_name,
                        cm.updated_at_column,
                    )
                )
            if has_upd_col:
                sql_with_ts = f'{core_sql}, "{cm.updated_at_column}"::text AS upd_ts FROM {schema_q}.{table_q}'
                if since_updated_at:
                    rows = await conn.fetch(
                        f'{sql_with_ts} WHERE "{cm.updated_at_column}"::text > $1 ORDER BY "{cm.updated_at_column}" ASC',
                        since_updated_at,
                    )
                else:
                    rows = await conn.fetch(sql_with_ts)
            else:
                rows = await conn.fetch(f"{core_sql} FROM {schema_q}.{table_q}")

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
                    updated_at=str(r["upd_ts"]) if has_upd_col and r["upd_ts"] else utc_now_iso(),
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
                    ON CONFLICT (tenant_id, entity_id) DO UPDATE SET
                        canonical_name = EXCLUDED.canonical_name,
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

            # Native-dimension vector fan-out over allowlisted legacy pgvector tables + RRF fusion
            if self.allowed_legacy_tables:
                try:
                    kbs = await self.list_registered_kbs(tenant_id=tenant_id)
                except Exception:
                    kbs = []
                legacy_kbs = [
                    kb
                    for kb in kbs
                    if kb.source_type == KBSourceType.EXISTING_PGVECTOR_KB
                    and (kb_filter is None or kb.kb_id in kb_filter)
                    and (domain_filter is None or kb.domain_tag in domain_filter)
                ]
                if legacy_kbs:
                    fused_map: dict[str, RetrievedChunkContext] = {r.chunk_id: r for r in results}
                    for kb in legacy_kbs:
                        cm = kb.column_mapping
                        try:
                            validate_legacy_table_access(
                                schema_name=kb.schema_name,
                                table_name=kb.table_name,
                                column_names=[
                                    cm.id_column,
                                    cm.doc_id_column,
                                    cm.text_column,
                                    cm.embedding_column,
                                    cm.metadata_column,
                                ],
                                allowed_patterns=self.allowed_legacy_tables,
                                enforce_allowlist=True,
                            )
                            native_q_vec = self.model_router.embed_query_for_kb(query_text, kb)
                            n_vec_str = "[" + ",".join(f"{v:.6f}" for v in native_q_vec) + "]"
                            async with self._scoped_alias_conn(
                                kb.postgres_dsn_alias, tenant_id
                            ) as lconn:
                                lrows = await lconn.fetch(
                                    f'SELECT "{cm.id_column}" AS cid, "{cm.doc_id_column}" AS did, '
                                    f'"{cm.text_column}" AS txt, "{cm.metadata_column}" AS meta '
                                    f'FROM "{kb.schema_name}"."{kb.table_name}" '
                                    f'ORDER BY "{cm.embedding_column}" <=> $1::vector LIMIT $2',
                                    n_vec_str,
                                    max(top_k * 2, 10),
                                )
                            chan_name = f"pgvector_native_{kb.embedding_model_id}({kb.embedding_dim}d)"
                            for rank, lr in enumerate(lrows, start=1):
                                cid = f"{tenant_id}::{kb.kb_id}::{lr['cid']}"
                                meta = self._parse_json_safe(lr["meta"]) if lr["meta"] else {}
                                l_acls = (
                                    meta.get("acl_groups", ["public"])
                                    if isinstance(meta, dict)
                                    else ["public"]
                                )
                                if caller_acls is not None:
                                    l_set = {str(g).lower() for g in l_acls}
                                    if "public" not in l_set and not caller_acls.intersection(l_set):
                                        continue
                                boost = 1.0 / (60.0 + rank)
                                if cid in fused_map:
                                    fused_map[cid].rrf_score = round(fused_map[cid].rrf_score + boost, 5)
                                    if chan_name not in fused_map[cid].retrieval_channels:
                                        fused_map[cid].retrieval_channels.append(chan_name)
                                else:
                                    fused_map[cid] = RetrievedChunkContext(
                                        chunk_id=cid,
                                        kb_id=kb.kb_id,
                                        domain_tag=kb.domain_tag,
                                        doc_id=str(lr["did"]),
                                        text=str(lr["txt"]),
                                        rrf_score=round(boost, 5),
                                        retrieval_channels=[chan_name],
                                        acl_groups=[str(g) for g in l_acls],
                                    )
                        except Exception:
                            continue
                    results = sorted(
                        fused_map.values(), key=lambda x: x.rrf_score, reverse=True
                    )[:top_k]

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
