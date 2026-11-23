from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

from .models import RawDocumentInput


class MultimodalDocumentIngestor:
    """
    Reuses IBM **Docling** (`docling.document_converter.DocumentConverter`) or **LlamaParse**
    when installed to parse PDFs, DOCX, PPTX, Excel spreadsheets, and scanned documents into
    layout-aware Markdown + structured tables.

    Also provides a built-in **Table-to-KG-Triples Parser** for Markdown/CSV tables (e.g.
    FinOps cost-center tables, SRE dependency matrices, Compliance control sheets) so tabular
    enterprise data converts directly into `RawDocumentInput` with `structured_triples` metadata!
    """

    @staticmethod
    def docling_available() -> bool:
        try:
            import docling  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def parse_file_to_raw_document(
        cls,
        file_path: str | Path,
        doc_id: str,
        kb_id: str,
        tenant_id: str = "enterprise_default",
        domain_tag: str = "general",
        acl_groups: list[str] | None = None,
    ) -> RawDocumentInput:
        path = Path(file_path)
        suffix = path.suffix.lower()
        acls = acl_groups or ["public"]

        # 1. Reuse IBM Docling for PDF / DOCX / PPTX / XLSX / HTML / Images when installed
        if suffix in {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg"} and cls.docling_available():
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(str(path))
            markdown_text = result.document.export_to_markdown()
            triples = cls.extract_triples_from_markdown_tables(markdown_text)
            return RawDocumentInput(
                doc_id=doc_id,
                kb_id=kb_id,
                tenant_id=tenant_id,
                domain_tag=domain_tag,
                title=path.stem,
                source_uri=str(path),
                mime_type="text/markdown",
                content=markdown_text,
                acl_groups=acls,
                metadata={"parser_engine": "ibm_docling", "structured_triples": triples},
            )

        # 2. CSV / TSV Table Parsing
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".csv":
            md_table, triples = cls.parse_csv_to_markdown_and_triples(raw_text)
            return RawDocumentInput(
                doc_id=doc_id,
                kb_id=kb_id,
                tenant_id=tenant_id,
                domain_tag=domain_tag,
                title=path.stem,
                source_uri=str(path),
                mime_type="text/csv",
                content=md_table,
                acl_groups=acls,
                metadata={"parser_engine": "csv_table_ingestor", "structured_triples": triples},
            )

        # 3. Markdown / Plaintext with automatic Markdown table-to-triple extraction
        triples = cls.extract_triples_from_markdown_tables(raw_text)
        return RawDocumentInput(
            doc_id=doc_id,
            kb_id=kb_id,
            tenant_id=tenant_id,
            domain_tag=domain_tag,
            title=path.stem,
            source_uri=str(path),
            mime_type="text/markdown",
            content=raw_text,
            acl_groups=acls,
            metadata={"parser_engine": "markdown_table_ingestor", "structured_triples": triples},
        )

    @classmethod
    def parse_csv_to_markdown_and_triples(
        cls, csv_content: str
    ) -> tuple[str, list[dict[str, Any]]]:
        reader = csv.DictReader(io.StringIO(csv_content.strip()))
        rows = list(reader)
        if not rows or not reader.fieldnames:
            return csv_content, []

        headers = [h.strip() for h in reader.fieldnames]
        md_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            md_lines.append("| " + " | ".join(str(row.get(h, "")).strip() for h in headers) + " |")
        md_text = "\n".join(md_lines)
        triples = cls._extract_triples_from_row_dicts(rows)
        return md_text, triples

    @classmethod
    def extract_triples_from_markdown_tables(cls, markdown_text: str) -> list[dict[str, Any]]:
        """Parses GitHub-flavored Markdown tables into structured KG triples."""
        lines = [ln.strip() for ln in markdown_text.splitlines()]
        row_dicts: list[dict[str, str]] = []
        i = 0
        while i + 2 < len(lines):
            if lines[i].startswith("|") and re.match(r"^\|[\s\-\:\|]+\|$", lines[i + 1]):
                headers = [c.strip() for c in lines[i].strip("|").split("|")]
                j = i + 2
                while j < len(lines) and lines[j].startswith("|"):
                    vals = [c.strip() for c in lines[j].strip("|").split("|")]
                    if len(vals) == len(headers):
                        row_dicts.append(dict(zip(headers, vals)))
                    j += 1
                i = j
            else:
                i += 1
        return cls._extract_triples_from_row_dicts(row_dicts)

    @staticmethod
    def _extract_triples_from_row_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Maps common enterprise tabular columns (Service, Database, Region, CostCenter, Compliance, Owner) to triples."""
        triples: list[dict[str, Any]] = []
        col_predicate_map: list[tuple[set[str], str, str]] = [
            ({"database", "datastore", "depends_on", "upstream_db", "db_cluster"}, "DEPENDS_ON", "DatabaseCluster"),
            ({"region", "primary_region", "cloud_region"}, "DEPLOYED_IN_PRIMARY", "CloudRegion"),
            ({"cost_center", "costcenter", "budget_unit"}, "OWNED_BY_PRIMARY", "CostCenter"),
            ({"compliance", "standard", "regulation", "governed_by"}, "GOVERNED_BY", "ComplianceStandard"),
            ({"team", "owner_team", "squad", "maintained_by"}, "MAINTAINED_BY_TEAM", "EngineeringTeam"),
            ({"cve", "vulnerability"}, "AFFECTS_ASSET", "SecurityVulnerability"),
        ]

        for row in rows:
            norm_row = {k.strip().lower().replace(" ", "_"): str(v).strip() for k, v in row.items() if v}
            subject = (
                norm_row.get("service")
                or norm_row.get("microservice")
                or norm_row.get("system")
                or norm_row.get("asset")
                or norm_row.get("entity")
            )
            if not subject:
                continue

            for col_aliases, predicate, obj_cls in col_predicate_map:
                for alias in col_aliases:
                    val = norm_row.get(alias)
                    if val and val.lower() not in {"none", "n/a", "-"}:
                        if predicate == "AFFECTS_ASSET":
                            triples.append(
                                {
                                    "subject": val,
                                    "subject_class": obj_cls,
                                    "predicate": predicate,
                                    "object": subject,
                                    "object_class": "CloudService",
                                    "confidence": 0.94,
                                }
                            )
                        else:
                            triples.append(
                                {
                                    "subject": subject,
                                    "subject_class": "CloudService",
                                    "predicate": predicate,
                                    "object": val,
                                    "object_class": obj_cls,
                                    "confidence": 0.94,
                                }
                            )
        return triples
