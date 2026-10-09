"""docs/STAR_SCHEMA.md against the schema it describes.

Claim: the doc says it describes the warehouse "as built", and nothing held
it to that. Tests pin the DDL to itself (column-list assertions, the
natural-key and coverage drift tests, the fingerprint), so a column can be
added, renamed or dropped with every one of them green while the doc a
reader opens first still shows the old shape. Two agents used to be asked to
eyeball this; both were checking names that had stopped existing.

Three things are held, each in the direction a reader gets hurt:

- a column table in the doc names only columns that exist, with the type
  DuckDB reports (a ghost column is a query that fails);
- a column table lists every column of its object, apart from the lineage
  block the doc states once for all facts (a missing column is a capability
  the reader never learns about);
- every table and view in `main` is named somewhere in the doc.

It does NOT require a column table for every object. Where the doc gives
only prose, the reader's guide and DESCRIBE are the column list; where it
gives a table, the table is exact.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ccutils import create_star_schema

DOC = Path(__file__).resolve().parents[1] / "docs" / "STAR_SCHEMA.md"

# The lineage block, stated once in the doc under "Lineage convention" and
# carried by every fact. A column table may list these or leave them out.
LINEAGE_BLOCK = frozenset({
    "created_at", "created_by_version_key", "last_updated_at",
    "last_updated_by_version_key", "etl_run_id", "record_source",
    "hash_diff", "is_deleted", "deleted_at",
})

_HEADING = re.compile(r"^#### (.+)$")
_HEADER_ROW = re.compile(r"^\|\s*Column\s*\|\s*Type\s*\|")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")


def column_tables(markdown: str) -> dict[str, dict[str, str]]:
    """{object name: {column: documented type}} for each `####` section that
    carries a Column/Type table. A row naming several columns (`a / b`)
    documents each of them, with no type claimed unless there is one name."""
    tables: dict[str, dict[str, str]] = {}
    heading = None
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        m = _HEADING.match(lines[i])
        if m:
            heading = m.group(1)
        elif _HEADER_ROW.match(lines[i]) and heading is not None:
            name = heading.split()[0].strip("`")
            assert " / " not in heading, f"a column table under a shared heading: {heading!r}"
            columns = tables.setdefault(name, {})
            i += 2  # the header row and its separator
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in _CELL_SPLIT.split(lines[i])[1:-1]]
                names = [n.strip().strip("`") for n in cells[0].split(" / ")]
                for n in names:
                    assert _NAME.match(n), f"{name}: not a column name: {cells[0]!r}"
                    columns[n] = cells[1] if len(names) == 1 else ""
                i += 1
            continue
        i += 1
    return tables


def _normalise(sql_type: str) -> str:
    """Spellings DuckDB treats as one type, folded together."""
    t = sql_type.upper().strip()
    return {"TEXT": "VARCHAR", "INT": "INTEGER", "FLOAT8": "DOUBLE", "REAL": "FLOAT",
            "TIMESTAMP WITH TIME ZONE": "TIMESTAMPTZ"}.get(t, t)


@pytest.fixture(scope="module")
def schema(tmp_path_factory):
    conn = create_star_schema(tmp_path_factory.mktemp("doc") / "w.duckdb")
    objects = {
        name: {r[0]: r[1] for r in conn.execute(f'DESCRIBE "{name}"').fetchall()}
        for (name,) in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }
    conn.close()
    return objects


@pytest.fixture(scope="module")
def documented():
    return column_tables(DOC.read_text())


def test_the_doc_has_column_tables_to_check(documented):
    """Non-vacuity: a changed heading level or header row would leave nothing."""
    assert len(documented) >= 8, sorted(documented)
    assert "fact_messages" in documented and "dim_session" in documented


def test_every_documented_object_exists(documented, schema):
    assert sorted(set(documented) - set(schema)) == []


def test_documented_columns_exist(documented, schema):
    ghosts = {
        obj: sorted(set(cols) - set(schema[obj]))
        for obj, cols in documented.items() if obj in schema
    }
    assert {o: g for o, g in ghosts.items() if g} == {}


def test_column_tables_are_complete(documented, schema):
    missing = {
        obj: sorted(set(schema[obj]) - set(cols) - LINEAGE_BLOCK)
        for obj, cols in documented.items() if obj in schema
    }
    assert {o: m for o, m in missing.items() if m} == {}


def test_documented_types_match(documented, schema):
    wrong = {}
    for obj, cols in documented.items():
        for col, doc_type in cols.items():
            actual = schema.get(obj, {}).get(col)
            if doc_type and actual and _normalise(doc_type) != _normalise(actual):
                wrong[f"{obj}.{col}"] = f"doc {doc_type}, schema {actual}"
    assert wrong == {}


def test_every_object_is_named_in_the_doc(schema):
    text = DOC.read_text()
    unnamed = sorted(n for n in schema if not re.search(rf"\b{re.escape(n)}\b", text))
    assert unnamed == []


class TestTheParserCanFail:
    """The oracle, on a doc written to be wrong in each way."""

    DOC_TEXT = (
        "#### fact_x (view)\n"
        "Prose.\n\n"
        "| Column | Type | Description |\n"
        "|--------|------|-------------|\n"
        "| real_col | INTEGER | fine |\n"
        "| a_key / b_key | VARCHAR | two columns, one row, a \\| in the text |\n"
        "| ghost_col | TEXT | not in the schema |\n"
    )

    def test_it_reads_names_types_and_shared_rows(self):
        assert column_tables(self.DOC_TEXT) == {
            "fact_x": {"real_col": "INTEGER", "a_key": "", "b_key": "", "ghost_col": "TEXT"}
        }

    def test_a_ghost_and_a_gap_are_both_visible(self):
        documented = column_tables(self.DOC_TEXT)["fact_x"]
        actual = {"real_col": "INTEGER", "a_key": "VARCHAR", "b_key": "VARCHAR",
                  "undocumented": "VARCHAR", "is_deleted": "BOOLEAN"}
        assert sorted(set(documented) - set(actual)) == ["ghost_col"]
        assert sorted(set(actual) - set(documented) - LINEAGE_BLOCK) == ["undocumented"]
