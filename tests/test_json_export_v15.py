"""Tests for the JSON export's schema-relationships metadata.

``RELATIONSHIPS`` was a hardcoded literal that drifted: it referenced
``fact_tool_calls`` (removed from the DDL entirely), pointed at the
never-populated ``fact_turn_durations`` / ``fact_stop_events`` stubs,
and omitted most populated v0.15 facts. Relationships are now derived
from the live database at export time -- the same discipline
``_star_tables`` already applies to the table manifest.
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.schemas.star.json_export import (
    export_star_schema_to_json,
    star_relationships,
)


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _table_names(conn):
    return {
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_type = 'BASE TABLE'"
        ).fetchall()
    }


def _column_names(conn, table):
    return {
        r[0]
        for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ?",
            [table],
        ).fetchall()
    }


class TestStarRelationships:
    def test_every_relationship_binds_against_live_schema(self, conn):
        tables = _table_names(conn)
        rels = star_relationships(conn)
        assert rels, "expected a non-empty relationship list"
        for rel in rels:
            assert rel["from_table"] in tables, rel
            assert rel["to_table"] in tables, rel
            assert rel["from_column"] in _column_names(conn, rel["from_table"]), rel
            assert rel["to_column"] in _column_names(conn, rel["to_table"]), rel

    def test_no_legacy_fact_tool_calls_references(self, conn):
        rels = star_relationships(conn)
        assert not any(
            "fact_tool_calls" in (r["from_table"], r["to_table"]) for r in rels
        )

    def test_covers_populated_v15_facts(self, conn):
        triples = {
            (r["from_table"], r["from_column"], r["to_table"])
            for r in star_relationships(conn)
        }
        assert ("fact_tool_uses", "tool_key", "dim_tool") in triples
        assert ("fact_tool_results", "session_key", "dim_session") in triples
        assert ("fact_session_facets", "facet_type_key", "dim_facet_type") in triples
        assert ("fact_agent_delegations", "agent_session_key", "dim_session") in triples
        assert ("bridge_session_file", "file_key", "dim_file") in triples

    def test_no_self_pk_relationships(self, conn):
        # dim_session.parent_session_key -> dim_session.session_key is fine
        # (self-referencing hierarchy); dim_session.session_key -> itself is not.
        for r in star_relationships(conn):
            assert not (
                r["from_table"] == r["to_table"]
                and r["from_column"] == r["to_column"]
            ), r

    def test_meta_json_carries_derived_relationships(self, conn, tmp_path):
        out = tmp_path / "export"
        export_star_schema_to_json(conn, out)
        meta = json.loads((out / "meta.json").read_text())
        tables = _table_names(conn)
        assert meta["relationships"], "meta.json relationships must not be empty"
        for rel in meta["relationships"]:
            assert rel["from_table"] in tables, rel
            assert rel["to_table"] in tables, rel
