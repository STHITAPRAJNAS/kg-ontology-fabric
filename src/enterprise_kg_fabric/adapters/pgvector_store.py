from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..models import (
    KBWatermarkState,
    KGEntityNode,
    RetrievedChunkContext,
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
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_unified_chunks_hnsw
    ON unified_rag_chunks USING hnsw (canonical_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_unified_chunks_tsv
    ON unified_rag_chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS idx_unified_chunks_tenant_kb
    ON unified_rag_chunks (tenant_id, kb_id, domain_tag);

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

ALTER TABLE unified_rag_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE canonical_entity_embeddings ENABLE ROW LEVEL SECURITY;
""".strip()


class PgVectorStorePort(ABC):
    """Port interface for multi-KB PostgreSQL + pgvector operations."""

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

    async def register_kb(self, registration: VectorKBRegistration) -> None:
        self._kb_registry[registration.kb_id] = registration
        self._kb_tables.setdefault(registration.kb_id, {})

    async def list_registered_kbs(self, tenant_id: str | None = None) -> list[VectorKBRegistration]:
        all_kbs = list(self._kb_registry.values())
        if tenant_id is not None:
            return [kb for kb in all_kbs if kb.tenant_id == tenant_id]
        return all_kbs

    async def upsert_kb_chunks(self, kb_id: str, chunks: Sequence[VectorChunkRecord]) -> int:
        if kb_id not in self._kb_registry:
            # Auto-register default KB entry if needed
            first_tenant = chunks[0].tenant_id if chunks else "enterprise_default"
            first_domain = chunks[0].domain_tag if chunks else "general"
            first_dim = (
                chunks[0].native_embedding_dim
                if (chunks and chunks[0].native_embedding_dim > 0)
                else 768
            )
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
        table = self._kb_tables.setdefault(kb_id, {})
        count = 0
        for chk in chunks:
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
    ) -> list[VectorChunkRecord]:
        table = self._kb_tables.get(kb_id, {})
        if not only_unindexed_or_changed:
            return list(table.values())

        delta_chunks: list[VectorChunkRecord] = []
        for chunk_id, chk in table.items():
            wm = self._watermarks.get((kb_id, chunk_id))
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
    ) -> list[RetrievedChunkContext]:
        candidate_chunks: list[VectorChunkRecord] = []
        kb_allow = set(kb_filter) if kb_filter else None
        dom_allow = set(domain_filter) if domain_filter else None

        for kb_id, table in self._kb_tables.items():
            if kb_allow is not None and kb_id not in kb_allow:
                continue
            for chk in table.values():
                if chk.tenant_id != tenant_id:
                    continue
                if dom_allow is not None and chk.domain_tag not in dom_allow:
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
    ):
        self.dsn = dsn
        self.embedding_provider = embedding_provider
        self.canonical_dim = canonical_dim
        self._pool: Any = None

    async def _get_pool(self) -> Any:
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(dsn=self.dsn, min_size=1, max_size=10)
            async with self._pool.acquire() as conn:
                await conn.execute(get_production_pgvector_ddl(self.canonical_dim))
        return self._pool

    async def register_kb(self, registration: VectorKBRegistration) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
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

    async def list_registered_kbs(self, tenant_id: str | None = None) -> list[VectorKBRegistration]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
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
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

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
                    chunk_id=str(r["cid"]),
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
        pool = await self._get_pool()
        rows_to_insert = []
        for chk in chunks:
            if not chk.canonical_embedding:
                chk.canonical_embedding = self.embedding_provider.embed_text(
                    chk.text, dim=self.canonical_dim
                )
            vec_str = "[" + ",".join(f"{v:.6f}" for v in chk.canonical_embedding) + "]"
            rows_to_insert.append(
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
                    json.dumps(chk.metadata),
                )
            )
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO unified_rag_chunks (
                    chunk_id, kb_id, tenant_id, domain_tag, doc_id, text,
                    content_sha256, canonical_embedding, section_path, metadata
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8::vector,$9,$10::jsonb)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    content_sha256 = EXCLUDED.content_sha256,
                    canonical_embedding = EXCLUDED.canonical_embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """,
                rows_to_insert,
            )
        return len(chunks)

    async def fetch_kb_chunks(
        self,
        kb_id: str,
        only_unindexed_or_changed: bool = False,
        target_tbox_version: str | None = None,
    ) -> list[VectorChunkRecord]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if only_unindexed_or_changed:
                rows = await conn.fetch(
                    """
                    SELECT c.chunk_id, c.kb_id, c.tenant_id, c.domain_tag, c.doc_id,
                           c.text, c.content_sha256, c.canonical_embedding::text AS emb_txt,
                           c.section_path, c.metadata, c.updated_at
                    FROM unified_rag_chunks c
                    LEFT JOIN kb_ingestion_watermarks w
                      ON c.kb_id = w.kb_id AND c.chunk_id = w.chunk_id
                    WHERE c.kb_id = $1
                      AND (w.chunk_id IS NULL OR w.content_sha256 != c.content_sha256
                           OR ($2::text IS NOT NULL AND w.tbox_version != $2))
                    """,
                    kb_id,
                    target_tbox_version,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT chunk_id, kb_id, tenant_id, domain_tag, doc_id,
                           text, content_sha256, canonical_embedding::text AS emb_txt,
                           section_path, metadata, updated_at
                    FROM unified_rag_chunks WHERE kb_id = $1
                    """,
                    kb_id,
                )
            return [
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
                    metadata=self._parse_json_safe(r["metadata"]),
                    updated_at=r["updated_at"].isoformat() if r["updated_at"] else utc_now_iso(),
                )
                for r in rows
            ]

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
        pool = await self._get_pool()
        rows = []
        for ent in entities:
            if not ent.embedding:
                ent.embedding = self.embedding_provider.embed_text(
                    f"{ent.ontology_class} {ent.canonical_name} {ent.summary}",
                    dim=self.canonical_dim,
                )
            vec_str = "[" + ",".join(f"{v:.6f}" for v in ent.embedding) + "]"
            rows.append(
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
        async with pool.acquire() as conn:
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
    ) -> list[RetrievedChunkContext]:
        pool = await self._get_pool()
        vec_str = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH dense AS (
                    SELECT chunk_id, kb_id, domain_tag, doc_id, text,
                           ROW_NUMBER() OVER (ORDER BY canonical_embedding <=> $2::vector) AS dense_rank
                    FROM unified_rag_chunks
                    WHERE tenant_id = $1
                      AND ($5::text[] IS NULL OR kb_id = ANY($5::text[]))
                      AND ($6::text[] IS NULL OR domain_tag = ANY($6::text[]))
                    ORDER BY canonical_embedding <=> $2::vector
                    LIMIT $4
                ),
                lexical AS (
                    SELECT chunk_id, kb_id, domain_tag, doc_id, text,
                           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $3)) DESC) AS lex_rank
                    FROM unified_rag_chunks
                    WHERE tenant_id = $1
                      AND ($5::text[] IS NULL OR kb_id = ANY($5::text[]))
                      AND ($6::text[] IS NULL OR domain_tag = ANY($6::text[]))
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
            )
            results: list[RetrievedChunkContext] = []
            for r in rows:
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
                    )
                )
            return results

    async def search_similar_entities(
        self,
        tenant_id: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[KGEntityNode, float]]:
        from ..models import UpperOntologyCategory

        pool = await self._get_pool()
        vec_str = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"
        async with pool.acquire() as conn:
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
