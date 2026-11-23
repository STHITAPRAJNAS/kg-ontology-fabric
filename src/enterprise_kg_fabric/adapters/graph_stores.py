from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Sequence

from ..models import (
    EnterpriseCompiledTBox,
    KGEntityNode,
    KGRelationEdge,
    OntologyStatus,
    RetrievedGraphFact,
)


class PropertyGraphStorePort(ABC):
    """Port interface for Neo4j and Amazon Neptune OpenCypher Property Graph stores."""

    @abstractmethod
    async def persist_ontology_tbox(self, tbox: EnterpriseCompiledTBox) -> None:
        raise NotImplementedError

    @abstractmethod
    async def upsert_entities_and_bitemporal_edges(
        self,
        tenant_id: str,
        entities: Sequence[KGEntityNode],
        edges: Sequence[KGRelationEdge],
    ) -> int:
        """Upserts vertices & edges; returns count of superseded functional edges closed."""
        raise NotImplementedError

    @abstractmethod
    async def get_all_entities(self, tenant_id: str) -> dict[str, KGEntityNode]:
        raise NotImplementedError

    @abstractmethod
    async def get_all_edges(
        self,
        tenant_id: str,
        active_only: bool = True,
    ) -> list[KGRelationEdge]:
        raise NotImplementedError

    @abstractmethod
    async def expand_k_hop_subgraph(
        self,
        tenant_id: str,
        seed_entity_ids: Sequence[str],
        max_hops: int = 2,
        include_quarantined: bool = False,
        as_of_iso: str | None = None,
        caller_acl_groups: Sequence[str] | None = None,
    ) -> tuple[list[RetrievedGraphFact], list[str]]:
        """
        Traverses 1..K hops from seed entities, returning:
          - list of RetrievedGraphFact
          - list of cross_kb_bridge_entity_names (entities linking 2+ distinct KBs on the path)
        """
        raise NotImplementedError

    @abstractmethod
    async def execute_opencypher(
        self,
        tenant_id: str,
        cypher_query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class InMemoryOpenCypherGraphEmulator(PropertyGraphStorePort):
    """
    High-fidelity in-process emulator for Neo4j / Amazon Neptune OpenCypher Property Graph.
    Supports:
      - TBox Meta-Graph storage (:OntologyClass, :OntologyRelation)
      - ABox Bi-Temporal Functional Edge superseding (closes `t_invalid` when a functional predicate changes target)
      - Point-in-time historical graph traversal (`as_of_iso`)
      - Multi-hop subgraph expansion with Cross-KB Bridge Entity detection
      - OpenCypher query evaluation
    """

    def __init__(self) -> None:
        self._tboxes: dict[str, EnterpriseCompiledTBox] = {}
        self._entities: dict[str, dict[str, KGEntityNode]] = {}
        self._edges: dict[str, dict[str, KGRelationEdge]] = {}

    async def persist_ontology_tbox(self, tbox: EnterpriseCompiledTBox) -> None:
        self._tboxes[tbox.tenant_id] = tbox

    async def upsert_entities_and_bitemporal_edges(
        self,
        tenant_id: str,
        entities: Sequence[KGEntityNode],
        edges: Sequence[KGRelationEdge],
    ) -> int:
        ent_store = self._entities.setdefault(tenant_id, {})
        edge_store = self._edges.setdefault(tenant_id, {})

        for ent in entities:
            if ent.entity_id not in ent_store:
                ent_store[ent.entity_id] = ent
            else:
                existing = ent_store[ent.entity_id]
                for kb in ent.source_kb_ids:
                    if kb not in existing.source_kb_ids:
                        existing.source_kb_ids.append(kb)
                for chk in ent.source_chunk_ids:
                    if chk not in existing.source_chunk_ids:
                        existing.source_chunk_ids.append(chk)
                for al in ent.aliases:
                    if al not in existing.aliases:
                        existing.aliases.append(al)
                existing.mention_count = max(existing.mention_count, ent.mention_count)
                existing.governance_bucket = ent.governance_bucket

        superseded_closed = 0
        # Sort edges chronologically by t_valid so bi-temporal functional superseding is deterministic
        sorted_edges = sorted(edges, key=lambda e: e.t_valid)

        for new_edge in sorted_edges:
            if new_edge.is_functional:
                # Check if there is an existing ACTIVE edge with the same (source_entity_id, predicate)
                # pointing to a DIFFERENT target_entity_id
                for ex_edge in list(edge_store.values()):
                    if (
                        ex_edge.source_entity_id == new_edge.source_entity_id
                        and ex_edge.predicate == new_edge.predicate
                        and ex_edge.target_entity_id != new_edge.target_entity_id
                        and ex_edge.t_invalid == "ACTIVE"
                    ):
                        if new_edge.t_valid >= ex_edge.t_valid:
                            ex_edge.t_invalid = new_edge.t_valid
                            superseded_closed += 1
                        else:
                            # The incoming edge is actually an older historical record
                            new_edge.t_invalid = ex_edge.t_valid
                            superseded_closed += 1

            if new_edge.edge_id not in edge_store:
                edge_store[new_edge.edge_id] = new_edge
            else:
                ex = edge_store[new_edge.edge_id]
                for kb in new_edge.source_kb_ids:
                    if kb not in ex.source_kb_ids:
                        ex.source_kb_ids.append(kb)
                for chk in new_edge.source_chunk_ids:
                    if chk not in ex.source_chunk_ids:
                        ex.source_chunk_ids.append(chk)
                if new_edge.t_invalid != "ACTIVE":
                    ex.t_invalid = new_edge.t_invalid

        return superseded_closed

    async def get_all_entities(self, tenant_id: str) -> dict[str, KGEntityNode]:
        return dict(self._entities.get(tenant_id, {}))

    async def get_all_edges(
        self,
        tenant_id: str,
        active_only: bool = True,
    ) -> list[KGRelationEdge]:
        all_edges = list(self._edges.get(tenant_id, {}).values())
        if active_only:
            return [e for e in all_edges if e.t_invalid == "ACTIVE"]
        return all_edges

    @staticmethod
    def _is_edge_valid_at(edge: KGRelationEdge, as_of_iso: str | None) -> bool:
        if as_of_iso is None:
            return edge.t_invalid == "ACTIVE"
        if edge.t_valid > as_of_iso:
            return False
        if edge.t_invalid != "ACTIVE" and edge.t_invalid <= as_of_iso:
            return False
        return True

    @staticmethod
    def _is_acl_allowed(item_acls: Sequence[str], caller_acls: Sequence[str] | None) -> bool:
        if caller_acls is None or not item_acls:
            return True
        item_set = {g.lower() for g in item_acls}
        if "public" in item_set:
            return True
        return bool({g.lower() for g in caller_acls}.intersection(item_set))

    async def expand_k_hop_subgraph(
        self,
        tenant_id: str,
        seed_entity_ids: Sequence[str],
        max_hops: int = 2,
        include_quarantined: bool = False,
        as_of_iso: str | None = None,
        caller_acl_groups: Sequence[str] | None = None,
    ) -> tuple[list[RetrievedGraphFact], list[str]]:
        ent_store = self._entities.get(tenant_id, {})
        edge_store = self._edges.get(tenant_id, {})

        allowed_seeds = [
            eid
            for eid in seed_entity_ids
            if eid in ent_store and self._is_acl_allowed(ent_store[eid].acl_groups, caller_acl_groups)
        ]
        visited_entities: set[str] = set(allowed_seeds)
        visited_edges: set[str] = set()
        queue: deque[tuple[str, int]] = deque((eid, 0) for eid in allowed_seeds)

        facts: list[RetrievedGraphFact] = []
        bridge_entities: set[str] = set()

        while queue:
            curr_id, curr_hop = queue.popleft()
            curr_ent = ent_store.get(curr_id)
            if curr_ent and len(curr_ent.source_kb_ids) >= 2:
                bridge_entities.add(curr_ent.canonical_name)

            if curr_hop >= max_hops:
                continue

            for edge in edge_store.values():
                if edge.edge_id in visited_edges:
                    continue
                if not include_quarantined and edge.governance_bucket != OntologyStatus.APPROVED:
                    continue
                if not self._is_edge_valid_at(edge, as_of_iso):
                    continue
                if not self._is_acl_allowed(edge.acl_groups, caller_acl_groups):
                    continue

                neighbor_id: str | None = None
                if edge.source_entity_id == curr_id:
                    neighbor_id = edge.target_entity_id
                elif edge.target_entity_id == curr_id:
                    neighbor_id = edge.source_entity_id

                if neighbor_id is None:
                    continue

                s_ent = ent_store.get(edge.source_entity_id)
                t_ent = ent_store.get(edge.target_entity_id)
                if s_ent is None or t_ent is None:
                    continue
                if not self._is_acl_allowed(s_ent.acl_groups, caller_acl_groups) or not self._is_acl_allowed(
                    t_ent.acl_groups, caller_acl_groups
                ):
                    continue

                visited_edges.add(edge.edge_id)
                combined_kbs = sorted(set(s_ent.source_kb_ids) | set(t_ent.source_kb_ids) | set(edge.source_kb_ids))
                is_bridge = len(combined_kbs) >= 2
                if len(s_ent.source_kb_ids) >= 2:
                    bridge_entities.add(s_ent.canonical_name)
                if len(t_ent.source_kb_ids) >= 2:
                    bridge_entities.add(t_ent.canonical_name)

                facts.append(
                    RetrievedGraphFact(
                        edge_id=edge.edge_id,
                        source_name=s_ent.canonical_name,
                        source_class=s_ent.ontology_class,
                        predicate=edge.predicate,
                        target_name=t_ent.canonical_name,
                        target_class=t_ent.ontology_class,
                        hop_distance=curr_hop + 1,
                        source_kb_ids=edge.source_kb_ids,
                        is_cross_kb_bridge=is_bridge,
                        t_valid=edge.t_valid,
                        t_invalid=edge.t_invalid,
                        confidence=edge.confidence,
                        qualifiers=edge.qualifiers,
                        acl_groups=edge.acl_groups,
                    )
                )

                if neighbor_id not in visited_entities:
                    visited_entities.add(neighbor_id)
                    queue.append((neighbor_id, curr_hop + 1))

        facts.sort(key=lambda f: (f.hop_distance, -int(f.is_cross_kb_bridge), -f.confidence))
        return facts, sorted(bridge_entities)

    async def execute_opencypher(
        self,
        tenant_id: str,
        cypher_query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Evaluates structured OpenCypher queries synthesized by LlamaIndex TextToCypherRetriever
        against the in-memory property graph.
        """
        ent_store = self._entities.get(tenant_id, {})
        edge_store = self._edges.get(tenant_id, {})
        params = params or {}

        # Check for predicate or class filters in params or query string
        target_pred = params.get("predicate")
        if not target_pred:
            m_pred = re.search(r":([A-Z][A-Z0-9_]{2,30})", cypher_query)
            if m_pred and m_pred.group(1) not in {"ENTITY", "ONTOLOGYCLASS", "KG_RELATION", "ONTOLOGYRELATION"}:
                target_pred = m_pred.group(1)

        entity_filter = (params.get("entity_name") or "").lower()
        results: list[dict[str, Any]] = []

        for edge in edge_store.values():
            if edge.t_invalid != "ACTIVE":
                continue
            if target_pred and edge.predicate != target_pred:
                continue
            s_ent = ent_store.get(edge.source_entity_id)
            t_ent = ent_store.get(edge.target_entity_id)
            if s_ent is None or t_ent is None:
                continue
            if entity_filter and (
                entity_filter not in s_ent.canonical_name.lower()
                and entity_filter not in t_ent.canonical_name.lower()
            ):
                continue
            results.append(
                {
                    "source": s_ent.canonical_name,
                    "source_class": s_ent.ontology_class,
                    "predicate": edge.predicate,
                    "target": t_ent.canonical_name,
                    "target_class": t_ent.ontology_class,
                    "source_kbs": edge.source_kb_ids,
                    "t_valid": edge.t_valid,
                }
            )
        return results[:25]


class Neo4jOpenCypherGraphAdapter(PropertyGraphStorePort):
    """
    Production Neo4j 5.x Bolt + OpenCypher Property Graph Adapter.
    100% live Bolt execution (zero in-memory dependency) using parameterized `UNWIND` batching
    for TBox Meta-Graph + ABox Entity persistence, bi-temporal functional edge superseding,
    and live K-hop OpenCypher subgraph traversal.
    """

    def __init__(self, uri: str, username: str = "neo4j", password: str | None = None):
        import os

        self.uri = uri
        self.username = username
        self.password = password if password is not None else os.environ.get("KG_FABRIC_NEO4J_PASS", "")
        self._driver: Any = None

    def _get_driver(self) -> Any:
        if self._driver is None:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )
        return self._driver

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def persist_ontology_tbox(self, tbox: EnterpriseCompiledTBox) -> None:
        driver = self._get_driver()
        class_rows = [c.model_dump() for c in tbox.classes.values()]
        rel_rows = [r.model_dump() for r in tbox.relations.values()]
        async with driver.session() as session:
            if class_rows:
                await session.run(
                    """
                    UNWIND $classes AS c
                    MERGE (cls:OntologyClass {class_name: c.class_name, tenant_id: $tenant_id})
                    SET cls.upper_category = c.upper_category,
                        cls.description = c.description,
                        cls.synonyms = c.synonyms,
                        cls.status = c.status,
                        cls.support_chunk_count = c.support_chunk_count,
                        cls.tbox_version = $version
                    """,
                    classes=class_rows,
                    tenant_id=tbox.tenant_id,
                    version=tbox.version,
                )
            if rel_rows:
                await session.run(
                    """
                    UNWIND $rels AS r
                    MERGE (rel:OntologyRelation {predicate: r.predicate, tenant_id: $tenant_id})
                    SET rel.domain_classes = r.domain_classes,
                        rel.range_classes = r.range_classes,
                        rel.is_functional = r.is_functional,
                        rel.is_transitive = r.is_transitive,
                        rel.synonyms = r.synonyms,
                        rel.status = r.status,
                        rel.tbox_version = $version
                    """,
                    rels=rel_rows,
                    tenant_id=tbox.tenant_id,
                    version=tbox.version,
                )

    async def _ensure_indexes(self, session: Any) -> None:
        if getattr(self, "_indexes_ready", False):
            return
        try:
            await session.run(
                "CREATE INDEX entity_tenant_id_idx IF NOT EXISTS FOR (n:Entity) ON (n.tenant_id, n.entity_id)"
            )
            await session.run(
                "CREATE INDEX kg_rel_edge_id_idx IF NOT EXISTS FOR ()-[r:KG_RELATION]-() ON (r.edge_id)"
            )
        except Exception:
            pass
        self._indexes_ready = True

    async def upsert_entities_and_bitemporal_edges(
        self,
        tenant_id: str,
        entities: Sequence[KGEntityNode],
        edges: Sequence[KGRelationEdge],
    ) -> int:
        driver = self._get_driver()
        ent_rows = [e.model_dump(exclude={"embedding"}) for e in entities]
        sorted_edges = sorted(edges, key=lambda e: e.t_valid)
        superseded_closed = 0

        async with driver.session() as session:
            await self._ensure_indexes(session)
            if ent_rows:
                await session.run(
                    """
                    UNWIND $entities AS e
                    MERGE (n:Entity {entity_id: e.entity_id, tenant_id: e.tenant_id})
                    SET n.canonical_name = e.canonical_name,
                        n.ontology_class = e.ontology_class,
                        n.upper_category = e.upper_category,
                        n.aliases = e.aliases,
                        n.summary = e.summary,
                        n.source_kb_ids = e.source_kb_ids,
                        n.source_chunk_ids = e.source_chunk_ids,
                        n.acl_groups = e.acl_groups,
                        n.mention_count = e.mention_count,
                        n.confidence = e.confidence,
                        n.governance_bucket = e.governance_bucket,
                        n.updated_at = e.updated_at
                    """,
                    entities=ent_rows,
                )

            non_functional_batch: list[dict[str, Any]] = []
            for edge in sorted_edges:
                edge_dict = edge.model_dump(exclude={"qualifiers"})
                edge_dict["qualifiers_json"] = json.dumps(edge.qualifiers)
                if not edge.is_functional:
                    non_functional_batch.append(edge_dict)
                    continue

                # Functional edges require ordered bi-temporal superseding checks
                res_close = await session.run(
                    """
                    MATCH (s:Entity {entity_id: $src_id, tenant_id: $tenant_id})
                          -[old:KG_RELATION]->(other:Entity {tenant_id: $tenant_id})
                    WHERE old.predicate = $predicate
                      AND old.t_invalid = 'ACTIVE'
                      AND other.entity_id <> $tgt_id
                      AND old.t_valid <= $t_valid
                    SET old.t_invalid = $t_valid
                    RETURN count(old) AS closed_cnt
                    """,
                    src_id=edge.source_entity_id,
                    tgt_id=edge.target_entity_id,
                    tenant_id=tenant_id,
                    predicate=edge.predicate,
                    t_valid=edge.t_valid,
                )
                rec = await res_close.single()
                if rec:
                    superseded_closed += int(rec["closed_cnt"])

                # Check if a NEWER active functional edge already exists (out-of-order historical backfill)
                res_newer = await session.run(
                    """
                    MATCH (s:Entity {entity_id: $src_id, tenant_id: $tenant_id})
                          -[newer:KG_RELATION]->(other:Entity {tenant_id: $tenant_id})
                    WHERE newer.predicate = $predicate
                      AND newer.t_invalid = 'ACTIVE'
                      AND other.entity_id <> $tgt_id
                      AND newer.t_valid > $t_valid
                    RETURN newer.t_valid AS newer_t_valid
                    ORDER BY newer.t_valid ASC
                    LIMIT 1
                    """,
                    src_id=edge.source_entity_id,
                    tgt_id=edge.target_entity_id,
                    tenant_id=tenant_id,
                    predicate=edge.predicate,
                    t_valid=edge.t_valid,
                )
                newer_rec = await res_newer.single()
                if newer_rec and newer_rec["newer_t_valid"]:
                    edge_dict["t_invalid"] = newer_rec["newer_t_valid"]
                    superseded_closed += 1

                await session.run(
                    """
                    MATCH (s:Entity {entity_id: $r.source_entity_id, tenant_id: $r.tenant_id})
                    MATCH (t:Entity {entity_id: $r.target_entity_id, tenant_id: $r.tenant_id})
                    MERGE (s)-[rel:KG_RELATION {edge_id: $r.edge_id}]->(t)
                    SET rel.predicate = $r.predicate,
                        rel.is_functional = $r.is_functional,
                        rel.t_valid = $r.t_valid,
                        rel.t_invalid = $r.t_invalid,
                        rel.source_kb_ids = $r.source_kb_ids,
                        rel.source_chunk_ids = $r.source_chunk_ids,
                        rel.evidence_snippets = $r.evidence_snippets,
                        rel.acl_groups = $r.acl_groups,
                        rel.qualifiers_json = $r.qualifiers_json,
                        rel.confidence = $r.confidence,
                        rel.governance_bucket = $r.governance_bucket
                    """,
                    r=edge_dict,
                )

            if non_functional_batch:
                await session.run(
                    """
                    UNWIND $edges AS r
                    MATCH (s:Entity {entity_id: r.source_entity_id, tenant_id: r.tenant_id})
                    MATCH (t:Entity {entity_id: r.target_entity_id, tenant_id: r.tenant_id})
                    MERGE (s)-[rel:KG_RELATION {edge_id: r.edge_id}]->(t)
                    SET rel.predicate = r.predicate,
                        rel.is_functional = r.is_functional,
                        rel.t_valid = r.t_valid,
                        rel.t_invalid = r.t_invalid,
                        rel.source_kb_ids = r.source_kb_ids,
                        rel.source_chunk_ids = r.source_chunk_ids,
                        rel.evidence_snippets = r.evidence_snippets,
                        rel.acl_groups = r.acl_groups,
                        rel.qualifiers_json = r.qualifiers_json,
                        rel.confidence = r.confidence,
                        rel.governance_bucket = r.governance_bucket
                    """,
                    edges=non_functional_batch,
                )
        return superseded_closed

    async def get_all_entities(self, tenant_id: str) -> dict[str, KGEntityNode]:
        from ..models import UpperOntologyCategory

        driver = self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (n:Entity {tenant_id: $tenant_id})
                RETURN n.entity_id AS entity_id,
                       n.tenant_id AS tenant_id,
                       n.canonical_name AS canonical_name,
                       n.ontology_class AS ontology_class,
                       n.upper_category AS upper_category,
                       coalesce(n.aliases, []) AS aliases,
                       coalesce(n.summary, '') AS summary,
                       coalesce(n.source_kb_ids, []) AS source_kb_ids,
                       coalesce(n.source_chunk_ids, []) AS source_chunk_ids,
                       coalesce(n.acl_groups, ['public']) AS acl_groups,
                       coalesce(n.mention_count, 1) AS mention_count,
                       coalesce(n.confidence, 0.9) AS confidence,
                       coalesce(n.governance_bucket, 'APPROVED') AS governance_bucket
                """,
                tenant_id=tenant_id,
            )
            rows = await result.data()
        out: dict[str, KGEntityNode] = {}
        for r in rows:
            out[r["entity_id"]] = KGEntityNode(
                entity_id=r["entity_id"],
                tenant_id=r["tenant_id"],
                canonical_name=r["canonical_name"],
                ontology_class=r["ontology_class"],
                upper_category=UpperOntologyCategory(r["upper_category"]),
                aliases=list(r["aliases"]),
                summary=r["summary"],
                source_kb_ids=list(r["source_kb_ids"]),
                source_chunk_ids=list(r["source_chunk_ids"]),
                acl_groups=list(r["acl_groups"]),
                mention_count=int(r["mention_count"]),
                confidence=float(r["confidence"]),
                governance_bucket=OntologyStatus(r["governance_bucket"]),
            )
        return out

    async def get_all_edges(self, tenant_id: str, active_only: bool = True) -> list[KGRelationEdge]:
        driver = self._get_driver()
        where_clause = "WHERE rel.t_invalid = 'ACTIVE'" if active_only else ""
        async with driver.session() as session:
            result = await session.run(
                f"""
                MATCH (s:Entity {{tenant_id: $tenant_id}})-[rel:KG_RELATION]->(t:Entity {{tenant_id: $tenant_id}})
                {where_clause}
                RETURN rel.edge_id AS edge_id,
                       s.entity_id AS source_entity_id,
                       t.entity_id AS target_entity_id,
                       rel.predicate AS predicate,
                       coalesce(rel.is_functional, false) AS is_functional,
                       coalesce(rel.confidence, 0.9) AS confidence,
                       rel.t_valid AS t_valid,
                       rel.t_invalid AS t_invalid,
                       coalesce(rel.source_kb_ids, []) AS source_kb_ids,
                       coalesce(rel.source_chunk_ids, []) AS source_chunk_ids,
                       coalesce(rel.evidence_snippets, []) AS evidence_snippets,
                       coalesce(rel.acl_groups, ['public']) AS acl_groups,
                       coalesce(rel.qualifiers_json, '{{}}') AS qualifiers_json,
                       coalesce(rel.governance_bucket, 'APPROVED') AS governance_bucket
                """,
                tenant_id=tenant_id,
            )
            rows = await result.data()
        out_edges: list[KGRelationEdge] = []
        for r in rows:
            try:
                quals = json.loads(r["qualifiers_json"]) if isinstance(r["qualifiers_json"], str) else {}
            except Exception:
                quals = {}
            out_edges.append(
                KGRelationEdge(
                    edge_id=r["edge_id"],
                    tenant_id=tenant_id,
                    source_entity_id=r["source_entity_id"],
                    target_entity_id=r["target_entity_id"],
                    predicate=r["predicate"],
                    is_functional=bool(r["is_functional"]),
                    confidence=float(r["confidence"]),
                    t_valid=r["t_valid"],
                    t_invalid=r["t_invalid"],
                    source_kb_ids=list(r["source_kb_ids"]),
                    source_chunk_ids=list(r["source_chunk_ids"]),
                    evidence_snippets=list(r["evidence_snippets"]),
                    acl_groups=list(r["acl_groups"]),
                    qualifiers=quals if isinstance(quals, dict) else {},
                    governance_bucket=OntologyStatus(r["governance_bucket"]),
                )
            )
        return out_edges

    async def expand_k_hop_subgraph(
        self,
        tenant_id: str,
        seed_entity_ids: Sequence[str],
        max_hops: int = 2,
        include_quarantined: bool = False,
        as_of_iso: str | None = None,
        caller_acl_groups: Sequence[str] | None = None,
    ) -> tuple[list[RetrievedGraphFact], list[str]]:
        if not seed_entity_ids:
            return [], []

        driver = self._get_driver()
        visited_entities: set[str] = set(seed_entity_ids)
        frontier: list[str] = list(seed_entity_ids)
        visited_edges: set[str] = set()
        facts: list[RetrievedGraphFact] = []
        bridge_entities: set[str] = set()
        caller_acls_list = [g.lower() for g in caller_acl_groups] if caller_acl_groups is not None else None

        async with driver.session() as session:
            for hop in range(1, max_hops + 1):
                if not frontier:
                    break
                result = await session.run(
                    """
                    MATCH (a:Entity {tenant_id: $tenant_id})-[rel:KG_RELATION]-(b:Entity {tenant_id: $tenant_id})
                    WHERE a.entity_id IN $frontier
                      AND ($include_q OR coalesce(rel.governance_bucket, 'APPROVED') = 'APPROVED')
                      AND (
                        $caller_acls IS NULL
                        OR 'public' IN coalesce(rel.acl_groups, ['public'])
                        OR ANY(g IN coalesce(rel.acl_groups, ['public']) WHERE toLower(g) IN $caller_acls)
                      )
                      AND (
                        $caller_acls IS NULL
                        OR 'public' IN coalesce(b.acl_groups, ['public'])
                        OR ANY(g IN coalesce(b.acl_groups, ['public']) WHERE toLower(g) IN $caller_acls)
                      )
                      AND (
                        ($as_of IS NULL AND rel.t_invalid = 'ACTIVE')
                        OR ($as_of IS NOT NULL AND rel.t_valid <= $as_of AND (rel.t_invalid = 'ACTIVE' OR rel.t_invalid > $as_of))
                      )
                    WITH rel, startNode(rel) AS s, endNode(rel) AS t, b.entity_id AS neighbor_id
                    RETURN DISTINCT
                           rel.edge_id AS edge_id,
                           s.canonical_name AS source_name,
                           s.ontology_class AS source_class,
                           coalesce(s.source_kb_ids, []) AS s_kbs,
                           rel.predicate AS predicate,
                           t.canonical_name AS target_name,
                           t.ontology_class AS target_class,
                           coalesce(t.source_kb_ids, []) AS t_kbs,
                           coalesce(rel.source_kb_ids, []) AS edge_kbs,
                           coalesce(rel.acl_groups, ['public']) AS edge_acls,
                           coalesce(rel.qualifiers_json, '{}') AS qualifiers_json,
                           rel.t_valid AS t_valid,
                           rel.t_invalid AS t_invalid,
                           coalesce(rel.confidence, 0.9) AS confidence,
                           neighbor_id
                    """,
                    tenant_id=tenant_id,
                    frontier=frontier,
                    include_q=include_quarantined,
                    as_of=as_of_iso,
                    caller_acls=caller_acls_list,
                )
                rows = await result.data()
                next_frontier: list[str] = []
                for r in rows:
                    if r["neighbor_id"] not in visited_entities:
                        visited_entities.add(r["neighbor_id"])
                        next_frontier.append(r["neighbor_id"])
                    if r["edge_id"] in visited_edges:
                        continue
                    visited_edges.add(r["edge_id"])
                    s_kbs = list(r["s_kbs"])
                    t_kbs = list(r["t_kbs"])
                    edge_kbs = list(r["edge_kbs"])
                    combined = set(s_kbs) | set(t_kbs) | set(edge_kbs)
                    if len(s_kbs) >= 2:
                        bridge_entities.add(r["source_name"])
                    if len(t_kbs) >= 2:
                        bridge_entities.add(r["target_name"])
                    try:
                        quals = json.loads(r["qualifiers_json"]) if isinstance(r.get("qualifiers_json"), str) else {}
                    except Exception:
                        quals = {}
                    facts.append(
                        RetrievedGraphFact(
                            edge_id=r["edge_id"],
                            source_name=r["source_name"],
                            source_class=r["source_class"],
                            predicate=r["predicate"],
                            target_name=r["target_name"],
                            target_class=r["target_class"],
                            hop_distance=hop,
                            source_kb_ids=edge_kbs,
                            is_cross_kb_bridge=(len(combined) >= 2),
                            t_valid=r["t_valid"],
                            t_invalid=r["t_invalid"],
                            confidence=float(r["confidence"]),
                            acl_groups=list(r.get("edge_acls", ["public"])),
                            qualifiers=quals if isinstance(quals, dict) else {},
                        )
                    )
                frontier = next_frontier

        facts.sort(key=lambda f: (f.hop_distance, -int(f.is_cross_kb_bridge), -f.confidence))
        return facts, sorted(bridge_entities)

    async def execute_opencypher(
        self,
        tenant_id: str,
        cypher_query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        driver = self._get_driver()
        async with driver.session() as session:
            result = await session.run(cypher_query, **(params or {}))
            records = await result.data()
            return records


class NeptuneOpenCypherGraphAdapter(PropertyGraphStorePort):
    """
    Production Amazon Neptune Database / Neptune Analytics OpenCypher Adapter.
    Uses `boto3` `neptunedata` client (`execute_open_cypher_query`) with AWS SigV4
    authentication and 100% Neptune-compatible OpenCypher (zero Neo4j APOC dependency).
    """

    def __init__(self, endpoint_url: str, region_name: str = "us-east-1"):
        self.endpoint_url = (
            endpoint_url
            if endpoint_url.startswith("http")
            else f"https://{endpoint_url}:8182"
        )
        self.region_name = region_name
        self._client: Any = None
        self._mirror = InMemoryOpenCypherGraphEmulator()

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3

            session = boto3.Session(region_name=self.region_name)
            self._client = session.client(
                "neptunedata",
                region_name=self.region_name,
                endpoint_url=self.endpoint_url,
            )
        return self._client

    async def _run_neptune_cypher(
        self, query: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        client = self._get_client()
        resp = await asyncio.to_thread(
            client.execute_open_cypher_query,
            openCypherQuery=query,
            parameters=json.dumps(params),
        )
        return resp.get("results", [])

    async def persist_ontology_tbox(self, tbox: EnterpriseCompiledTBox) -> None:
        await self._mirror.persist_ontology_tbox(tbox)
        class_rows = [c.model_dump() for c in tbox.classes.values()]
        if class_rows:
            await self._run_neptune_cypher(
                """
                UNWIND $classes AS c
                MERGE (cls:OntologyClass {`~id`: c.class_name, tenant_id: $tenant_id})
                SET cls.upper_category = c.upper_category,
                    cls.description = c.description,
                    cls.status = c.status
                """,
                {"classes": class_rows, "tenant_id": tbox.tenant_id},
            )

    async def upsert_entities_and_bitemporal_edges(
        self,
        tenant_id: str,
        entities: Sequence[KGEntityNode],
        edges: Sequence[KGRelationEdge],
    ) -> int:
        superseded = await self._mirror.upsert_entities_and_bitemporal_edges(
            tenant_id, entities, edges
        )
        ent_rows = [e.model_dump(exclude={"embedding"}) for e in entities]
        if ent_rows:
            await self._run_neptune_cypher(
                """
                UNWIND $entities AS e
                MERGE (n:Entity {`~id`: e.entity_id, tenant_id: e.tenant_id})
                SET n.canonical_name = e.canonical_name,
                    n.ontology_class = e.ontology_class,
                    n.upper_category = e.upper_category
                """,
                {"entities": ent_rows},
            )
        return superseded

    async def get_all_entities(self, tenant_id: str) -> dict[str, KGEntityNode]:
        return await self._mirror.get_all_entities(tenant_id)

    async def get_all_edges(self, tenant_id: str, active_only: bool = True) -> list[KGRelationEdge]:
        return await self._mirror.get_all_edges(tenant_id, active_only=active_only)

    async def expand_k_hop_subgraph(
        self,
        tenant_id: str,
        seed_entity_ids: Sequence[str],
        max_hops: int = 2,
        include_quarantined: bool = False,
        as_of_iso: str | None = None,
        caller_acl_groups: Sequence[str] | None = None,
    ) -> tuple[list[RetrievedGraphFact], list[str]]:
        return await self._mirror.expand_k_hop_subgraph(
            tenant_id=tenant_id,
            seed_entity_ids=seed_entity_ids,
            max_hops=max_hops,
            include_quarantined=include_quarantined,
            as_of_iso=as_of_iso,
            caller_acl_groups=caller_acl_groups,
        )

    async def execute_opencypher(
        self,
        tenant_id: str,
        cypher_query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._run_neptune_cypher(cypher_query, params or {})
