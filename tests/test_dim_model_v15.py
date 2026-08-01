"""Tests for dim_model classification and context-variant handling.

Two defects this file guards, both found 2026-08-01 by inspecting the real
corpus rather than by a failing test:

1. `claude-fable-5` fell through every family branch to `unknown`. It was the
   third-most-used model in the corpus -- 19,419 API responses and 23.7M
   output tokens, MORE output than Opus 5 -- so any `GROUP BY model_family`
   silently bucketed the largest single block of generation as "unknown".

2. Claude Code writes context-window variants into `message.model` as a
   bracket suffix (`claude-opus-5[1m]`). Left alone that becomes a second
   dim_model row for the same model, splitting every per-model aggregate.
   `model_base` carries the suffix-stripped id so grouping stays correct
   while `model_name` remains byte-faithful to the transcript.

The family rule lives in TWO mirrored places -- `get_model_family` (Python)
and the CASE in `etl/orchestrator.py` (SQL) -- so both are asserted here.
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl
from ccutils.schemas.star.utils import get_model_family


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _session_using(tmp_path, session_id, model):
    jsonl = tmp_path / f"{session_id}.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": session_id,
         "timestamp": "2026-04-19T19:00:00Z", "cwd": "/p",
         "gitBranch": "main", "version": "2.1.114",
         "message": {"role": "user", "content": "hello"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": session_id, "timestamp": "2026-04-19T19:00:01Z",
         "requestId": "r1",
         "message": {"role": "assistant", "model": model,
                     "content": [{"type": "text", "text": "hi"}],
                     "usage": {"input_tokens": 5, "output_tokens": 3}}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


class TestModelFamilyPython:
    """The Python half of the mirrored rule."""

    @pytest.mark.parametrize(
        "model_name,expected",
        [
            ("claude-fable-5", "fable"),
            ("claude-opus-5", "opus"),
            ("claude-opus-5[1m]", "opus"),
            ("claude-sonnet-5", "sonnet"),
            ("claude-haiku-4-5-20251001", "haiku"),
            ("<synthetic>", "unknown"),
            (None, "unknown"),
        ],
    )
    def test_family(self, model_name, expected):
        assert get_model_family(model_name) == expected


class TestModelFamilySql:
    """The SQL half. Both must agree or dim_model disagrees with any Python
    consumer of the same rule."""

    def test_fable_family_in_dim_model(self, conn, tmp_path):
        run_v15_etl(conn, _session_using(tmp_path, "fable-s", "claude-fable-5"),
                    project_name="test-project",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT model_family FROM dim_model WHERE model_name = 'claude-fable-5'"
        ).fetchone()
        assert row == ("fable",)

    def test_sql_and_python_agree(self, conn, tmp_path):
        for i, model in enumerate(
            ["claude-fable-5", "claude-opus-5", "claude-sonnet-5",
             "claude-haiku-4-5-20251001"]
        ):
            run_v15_etl(conn, _session_using(tmp_path, f"agree-{i}", model),
                        project_name="test-project",
                        parquet_lake_root=tmp_path / "lake")
        rows = conn.execute(
            "SELECT model_name, model_family FROM dim_model"
        ).fetchall()
        for model_name, family in rows:
            assert family == get_model_family(model_name), (
                f"SQL says {family} for {model_name}, Python says "
                f"{get_model_family(model_name)}"
            )

    def test_preexisting_unknown_family_is_backfilled(self, conn, tmp_path):
        """Warehouses built before the fable branch hold 'unknown'."""
        conn.execute(
            "INSERT INTO dim_model (model_key, model_name, model_family) "
            "VALUES (md5('claude-fable-5'), 'claude-fable-5', 'unknown')"
        )
        run_v15_etl(conn, _session_using(tmp_path, "bf-s", "claude-opus-5"),
                    project_name="test-project",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT model_family FROM dim_model WHERE model_name = 'claude-fable-5'"
        ).fetchone()
        assert row == ("fable",)


class TestContextVariant:
    """`claude-opus-5[1m]` is the same model as `claude-opus-5`."""

    def test_model_base_strips_the_context_suffix(self, conn, tmp_path):
        run_v15_etl(
            conn, _session_using(tmp_path, "ctx-s", "claude-opus-5[1m]"),
            project_name="test-project", parquet_lake_root=tmp_path / "lake",
        )
        row = conn.execute(
            "SELECT model_name, model_base, model_family FROM dim_model "
            "WHERE model_name = 'claude-opus-5[1m]'"
        ).fetchone()
        # model_name stays byte-faithful to the transcript.
        assert row == ("claude-opus-5[1m]", "claude-opus-5", "opus")

    def test_variants_share_a_model_base(self, conn, tmp_path):
        """Claim: delete this and per-model aggregates split silently the
        first time a context variant appears in the corpus."""
        run_v15_etl(conn, _session_using(tmp_path, "v1", "claude-opus-5"),
                    project_name="test-project",
                    parquet_lake_root=tmp_path / "lake")
        run_v15_etl(conn, _session_using(tmp_path, "v2", "claude-opus-5[1m]"),
                    project_name="test-project",
                    parquet_lake_root=tmp_path / "lake")
        bases = conn.execute(
            "SELECT DISTINCT model_base FROM dim_model "
            "WHERE model_name LIKE 'claude-opus-5%'"
        ).fetchall()
        assert bases == [("claude-opus-5",)]
        # Still two distinct rows -- the transcript said different things.
        n = conn.execute(
            "SELECT COUNT(*) FROM dim_model WHERE model_base = 'claude-opus-5'"
        ).fetchone()[0]
        assert n == 2

    def test_unsuffixed_model_base_equals_model_name(self, conn, tmp_path):
        run_v15_etl(conn, _session_using(tmp_path, "plain-s", "claude-opus-5"),
                    project_name="test-project",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT model_name, model_base FROM dim_model "
            "WHERE model_name = 'claude-opus-5'"
        ).fetchone()
        assert row[0] == row[1]
