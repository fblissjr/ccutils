"""Tests for the v0.15 Tier 1 facet populator.

populate_tier1_facets(conn, *, run) reads from the v0.15 fact tables that
have already been populated for the current staging session, computes the
19 Tier 1 facets defined in docs/FACET_CLUSTER_PIPELINE.md §3, and writes
one row per (session, facet) into fact_session_facets via the shared
lineage_upsert helper.

Properties checked:

  * Every seeded Tier 1 facet emits exactly one row for the session.
  * Bool facets default to FALSE when source is empty (graceful absence),
    not a missing row.
  * Rows resolve cleanly through dim_facet_type via facet_type_key.
  * Re-running the populator is a no-op (idempotency via hash_diff).
  * Same facet_id, different prompt_version yields different
    facet_type_key, and both versions coexist + resolve correctly.

The populator is invoked directly rather than through run_v15_etl so the
contract is verifiable independent of orchestrator wiring.
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.fact_session_facets import populate_tier1_facets
from ccutils.etl.lineage import EtlRun
from ccutils.etl.orchestrator import run_v15_etl


# F01..F19 -- Tier 1 facets seeded by create_star_schema().
_TIER1_FACET_IDS = tuple(f"F{i:02d}" for i in range(1, 20))
_BOOL_FACET_IDS = ("F17", "F18", "F19")  # had_subagents, pr_referenced, had_plan_revision


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


@pytest.fixture
def shaped_session(tmp_path):
    """Session with enough shape to exercise most Tier 1 facets:

    one user prompt, one assistant turn that calls Bash, one tool_result,
    typed token usage on the assistant. No subagents, no PR link, no
    plan revision -- so F17/F18/F19 must default to FALSE (graceful
    absence), not go missing.
    """
    jsonl = tmp_path / "shaped.jsonl"
    lines = [
        {
            "type": "user", "uuid": "u1", "sessionId": "shape-s",
            "timestamp": "2026-04-19T14:00:00Z",
            "cwd": "/p", "gitBranch": "main", "version": "2.1.114",
            "permissionMode": "default",
            "message": {"role": "user", "content": "fix the broken thing"},
        },
        {
            "type": "assistant", "uuid": "a1", "parentUuid": "u1",
            "sessionId": "shape-s", "timestamp": "2026-04-19T14:00:05Z",
            "requestId": "req_1",
            "message": {
                "role": "assistant", "model": "claude-opus-4-7",
                "content": [
                    {"type": "text", "text": "running it"},
                    {"type": "tool_use", "id": "tu1", "name": "Bash",
                     "input": {"command": "ls"}},
                ],
                "stop_reason": "tool_use",
                "usage": {
                    "input_tokens": 50, "output_tokens": 12,
                    "cache_read_input_tokens": 200,
                    "service_tier": "standard",
                },
            },
        },
        {
            "type": "user", "uuid": "u2", "parentUuid": "a1",
            "sessionId": "shape-s", "timestamp": "2026-04-19T14:00:10Z",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu1",
                     "content": "ok"},
                ],
            },
            "toolUseResult": {
                "stdout": "ok", "interrupted": False, "exitCode": 0,
            },
        },
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


def _run_orchestrator_then_populator(conn, session_path, lake_root):
    """Populate every v0.15 fact via run_v15_etl, then explicitly call the
    Tier 1 populator with a fresh EtlRun. Decouples the populator contract
    from orchestrator wiring."""
    run_v15_etl(
        conn, session_path,
        project_name="test-project",
        parquet_lake_root=lake_root,
    )
    run = EtlRun.start(conn, source_path=str(session_path))
    populate_tier1_facets(conn, run=run)
    run.complete()
    return run


class TestTier1FacetCoverage:
    def test_one_row_per_seeded_facet(self, conn, shaped_session, tmp_path):
        _run_orchestrator_then_populator(conn, shaped_session, tmp_path / "lake")
        rows = conn.execute(
            """
            SELECT dft.facet_id
            FROM fact_session_facets fsf
            JOIN dim_facet_type dft USING (facet_type_key)
            WHERE fsf.session_id = 'shape-s'
              AND dft.tier = 1
            ORDER BY dft.facet_id
            """
        ).fetchall()
        emitted = [r[0] for r in rows]
        # No duplicates and no missing facets.
        assert sorted(set(emitted)) == sorted(emitted), (
            f"Duplicate facet rows emitted: {emitted}"
        )
        assert set(emitted) == set(_TIER1_FACET_IDS), (
            f"Missing Tier 1 facets. Emitted: {emitted}"
        )

    def test_bool_facets_default_false_on_absence(
        self, conn, shaped_session, tmp_path
    ):
        # Session has no subagents / PR / plan revision; bool facets MUST
        # still emit a row with value_bool=FALSE so downstream queries can
        # filter without NULL handling.
        _run_orchestrator_then_populator(conn, shaped_session, tmp_path / "lake")
        for facet_id in _BOOL_FACET_IDS:
            row = conn.execute(
                """
                SELECT fsf.value_bool
                FROM fact_session_facets fsf
                JOIN dim_facet_type dft USING (facet_type_key)
                WHERE fsf.session_id = 'shape-s'
                  AND dft.facet_id = ?
                """,
                [facet_id],
            ).fetchone()
            assert row is not None, f"{facet_id} row missing"
            assert row[0] is False, (
                f"{facet_id} should default to FALSE on absence, got {row[0]}"
            )

    def test_value_columns_match_output_type(
        self, conn, shaped_session, tmp_path
    ):
        # Each row populates exactly one of value_text / value_json /
        # value_numeric / value_bool, matching the dim_facet_type.output_type.
        _run_orchestrator_then_populator(conn, shaped_session, tmp_path / "lake")
        rows = conn.execute(
            """
            SELECT dft.facet_id, dft.output_type,
                   fsf.value_text, fsf.value_json,
                   fsf.value_numeric, fsf.value_bool
            FROM fact_session_facets fsf
            JOIN dim_facet_type dft USING (facet_type_key)
            WHERE fsf.session_id = 'shape-s' AND dft.tier = 1
            """
        ).fetchall()
        for facet_id, output_type, vt, vj, vn, vb in rows:
            populated = {
                "text": vt is not None,
                "enum": vt is not None,
                "json": vj is not None,
                "int": vn is not None,
                "float": vn is not None,
                "bool": vb is not None,
            }
            assert populated[output_type], (
                f"{facet_id} (output_type={output_type}) "
                f"has no value in its expected column "
                f"(text={vt!r}, json={vj!r}, num={vn!r}, bool={vb!r})"
            )


class TestTier1FacetLineage:
    def test_lineage_envelope_stamped(self, conn, shaped_session, tmp_path):
        _run_orchestrator_then_populator(conn, shaped_session, tmp_path / "lake")
        row = conn.execute(
            """
            SELECT created_at, last_updated_at, etl_run_id,
                   record_source, hash_diff, is_deleted
            FROM fact_session_facets
            WHERE session_id = 'shape-s'
            LIMIT 1
            """
        ).fetchone()
        assert row[0] is not None  # created_at
        assert row[1] is not None  # last_updated_at
        assert row[2] is not None  # etl_run_id
        assert row[3] == "claude_code_jsonl"
        assert row[4] is not None  # hash_diff
        assert row[5] is False     # is_deleted

    def test_is_idempotent(self, conn, shaped_session, tmp_path):
        _run_orchestrator_then_populator(conn, shaped_session, tmp_path / "lake")
        before = conn.execute(
            """
            SELECT facet_type_key, hash_diff, last_updated_at
            FROM fact_session_facets WHERE session_id = 'shape-s'
            ORDER BY facet_type_key
            """
        ).fetchall()
        # Second call should produce no UPDATEs because hash_diff matches.
        run2 = EtlRun.start(conn, source_path=str(shaped_session))
        populate_tier1_facets(conn, run=run2)
        run2.complete()
        after = conn.execute(
            """
            SELECT facet_type_key, hash_diff, last_updated_at
            FROM fact_session_facets WHERE session_id = 'shape-s'
            ORDER BY facet_type_key
            """
        ).fetchall()
        assert len(before) == len(after), "Row count diverged on re-run"
        for (k1, h1, t1), (k2, h2, t2) in zip(before, after):
            assert k1 == k2 and h1 == h2, "Same row, different hash_diff"
            assert t1 == t2, (
                f"last_updated_at changed on no-op re-run for {k1}: "
                f"{t1} -> {t2}"
            )


class TestFacetTypeKeyVersionProperty:
    """Reflection 3 from the step-1 review: prompt versioning semantics.

    Same facet_id + different prompt_version MUST produce different
    facet_type_keys (so a re-extraction with a new prompt produces NEW
    rows in fact_session_facets, not destructive overwrites). Both
    versions must coexist in dim_facet_type and resolve via the join.
    """

    def test_same_id_different_version_yields_different_key(self, conn):
        conn.execute(
            """
            INSERT INTO dim_facet_type
                (facet_type_key, facet_id, facet_name, tier, method,
                 output_type, prompt_text, prompt_version)
            VALUES
                (md5('F20' || '|' || 'v1'), 'F20', 'task_description',
                 2, 'llm', 'text', 'prompt v1...', 'v1'),
                (md5('F20' || '|' || 'v2'), 'F20', 'task_description',
                 2, 'llm', 'text', 'prompt v2 (better)...', 'v2')
            """
        )
        keys = conn.execute(
            "SELECT prompt_version, facet_type_key FROM dim_facet_type "
            "WHERE facet_id = 'F20' ORDER BY prompt_version"
        ).fetchall()
        assert len(keys) == 2
        assert keys[0][1] != keys[1][1], (
            "Same facet_id with different prompt_version must produce "
            "distinct facet_type_key"
        )

    def test_both_versions_resolve_after_new_version_lands(self, conn):
        # Seed two versions for F20.
        conn.execute(
            """
            INSERT INTO dim_facet_type
                (facet_type_key, facet_id, facet_name, tier, method,
                 output_type, prompt_text, prompt_version)
            VALUES
                (md5('F20' || '|' || 'v1'), 'F20', 'task_description',
                 2, 'llm', 'text', 'prompt v1', 'v1'),
                (md5('F20' || '|' || 'v2'), 'F20', 'task_description',
                 2, 'llm', 'text', 'prompt v2', 'v2')
            """
        )
        # Insert a fact row keyed against v1 BEFORE v2 was seeded (simulated
        # by inserting now under v1's key only).
        conn.execute(
            """
            INSERT INTO fact_session_facets (
                created_by_version_key, last_updated_by_version_key,
                etl_run_id, record_source, hash_diff,
                facet_row_key, session_key, session_id,
                facet_type_key, prompt_version, value_text
            )
            VALUES (
                'vk', 'vk', 'run-1', 'claude_code_jsonl',
                'h1',
                md5('s1' || '|' || 'F20' || '|' || 'v1'),
                md5('s1'), 's1',
                md5('F20' || '|' || 'v1'), 'v1',
                'fixed the bug under prompt v1'
            )
            """
        )
        # Now insert a v2 row.
        conn.execute(
            """
            INSERT INTO fact_session_facets (
                created_by_version_key, last_updated_by_version_key,
                etl_run_id, record_source, hash_diff,
                facet_row_key, session_key, session_id,
                facet_type_key, prompt_version, value_text
            )
            VALUES (
                'vk', 'vk', 'run-2', 'claude_code_jsonl',
                'h2',
                md5('s1' || '|' || 'F20' || '|' || 'v2'),
                md5('s1'), 's1',
                md5('F20' || '|' || 'v2'), 'v2',
                'addressed the bug (re-extracted under prompt v2)'
            )
            """
        )
        # Both rows must resolve through the join.
        rows = conn.execute(
            """
            SELECT dft.prompt_version, fsf.value_text
            FROM fact_session_facets fsf
            JOIN dim_facet_type dft USING (facet_type_key)
            WHERE fsf.session_id = 's1' AND dft.facet_id = 'F20'
            ORDER BY dft.prompt_version
            """
        ).fetchall()
        assert len(rows) == 2, (
            f"Both versions should resolve via the join, got {rows}"
        )
        assert rows[0][0] == "v1" and "v1" in rows[0][1]
        assert rows[1][0] == "v2" and "v2" in rows[1][1]
