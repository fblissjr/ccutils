"""Tests for `--no-thinking` / `include_thinking=False` on the v0.15 ETL.

Two related contracts to pin:

1. **Characterization (already-true behavior).** `fact_messages.content_text`
   never contains thinking text -- the SQL projection in
   `populate_fact_messages` filters to `type='text'` blocks only. This test
   pins that so a future fact_messages refactor can't silently regress and
   start emitting thinking text into the user-visible content column.

2. **Behavior (new contract).** When `run_v15_etl(include_thinking=False)`,
   the orchestrator clears `stg_log_entries` after every populator runs.
   Without this, the staging table retains the last loaded session's raw
   JSON -- which includes thinking blocks in `message_json`. `fact_messages`
   is unaffected (already thinking-free); the truncate just removes the
   transient staging artifact from the user's archive.

3. **CLI behavior.** `--no-thinking` is accepted on `--format duckdb` /
   `--format json` (was previously rejected with a `click.UsageError`
   based on a misread that thinking text leaked into `content_text` --
   it does not, per #1).

The CLAUDE.md "exit-code-only flag tests are insufficient" rule applies:
these tests assert the actual stored data, not just that the CLI returned 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ccutils import create_star_schema
from ccutils.cli import cli
from ccutils.etl.orchestrator import run_v15_etl


_THINKING_PAYLOAD = "INTERNAL_THINKING_PAYLOAD_xyz123"
_VISIBLE_TEXT = "VISIBLE_TEXT_RESPONSE_abc456"


@pytest.fixture
def session_with_thinking(tmp_path):
    """JSONL session with an assistant turn carrying both a thinking block
    and a text block. The thinking payload contains a unique sentinel string
    so we can grep for it across tables."""
    jsonl = tmp_path / "thinking.jsonl"
    lines = [
        {
            "type": "user", "uuid": "u1", "sessionId": "think-s",
            "timestamp": "2026-04-19T10:00:00Z",
            "cwd": "/p", "gitBranch": "main", "version": "2.1.114",
            "message": {"role": "user", "content": "think about something"},
        },
        {
            "type": "assistant", "uuid": "a1", "parentUuid": "u1",
            "sessionId": "think-s", "timestamp": "2026-04-19T10:00:01Z",
            "requestId": "r1",
            "message": {
                "role": "assistant", "model": "claude-opus-4-7",
                "content": [
                    {"type": "thinking", "thinking": _THINKING_PAYLOAD},
                    {"type": "text", "text": _VISIBLE_TEXT},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 3,
                          "service_tier": "standard"},
            },
        },
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


class TestFactMessagesExcludesThinkingByDefault:
    """Characterization: fact_messages.content_text never contains thinking
    text, regardless of include_thinking. The SQL projection in
    populate_fact_messages picks only `type='text'` blocks.
    """

    def test_default_run_excludes_thinking_from_content_text(
        self, conn, session_with_thinking, tmp_path
    ):
        run_v15_etl(
            conn, session_with_thinking,
            project_name="x", parquet_lake_root=tmp_path / "lake",
        )
        row = conn.execute(
            "SELECT content_text, has_thinking FROM fact_messages "
            "WHERE session_id = 'think-s' AND message_type = 'assistant'"
        ).fetchone()
        content_text, has_thinking = row
        assert _THINKING_PAYLOAD not in (content_text or "")
        assert _VISIBLE_TEXT in content_text
        # has_thinking remains a correct boolean indicator; the flag is
        # what users filter on, not the absent text.
        assert has_thinking is True


class TestIncludeThinkingFalseClearsStaging:
    """run_v15_etl(include_thinking=False) clears stg_log_entries after the
    per-session populators run, so the user's archive doesn't carry the raw
    thinking JSON in the staging artifact."""

    def test_default_keeps_staging_populated(
        self, conn, session_with_thinking, tmp_path
    ):
        # Default (include_thinking=True) leaves staging as-is for the last
        # loaded session. The thinking payload is in message_json.
        run_v15_etl(
            conn, session_with_thinking,
            project_name="x", parquet_lake_root=tmp_path / "lake",
        )
        n = conn.execute(
            "SELECT COUNT(*) FROM stg_log_entries "
            "WHERE CAST(message_json AS VARCHAR) LIKE ?",
            [f"%{_THINKING_PAYLOAD}%"],
        ).fetchone()[0]
        assert n > 0, (
            "Default run should leave staging populated; "
            "the truncate only fires when include_thinking=False."
        )

    def test_include_thinking_false_truncates_staging(
        self, conn, session_with_thinking, tmp_path
    ):
        run_v15_etl(
            conn, session_with_thinking,
            project_name="x", parquet_lake_root=tmp_path / "lake",
            include_thinking=False,
        )
        n = conn.execute("SELECT COUNT(*) FROM stg_log_entries").fetchone()[0]
        assert n == 0, (
            "include_thinking=False should clear stg_log_entries entirely; "
            "otherwise the raw thinking JSON survives in the user's archive."
        )
        # And nothing in stg_log_entries -> the thinking payload is not
        # anywhere user-queryable.
        any_thinking = conn.execute(
            "SELECT COUNT(*) FROM stg_log_entries "
            "WHERE CAST(message_json AS VARCHAR) LIKE ?",
            [f"%{_THINKING_PAYLOAD}%"],
        ).fetchone()[0]
        assert any_thinking == 0

    def test_include_thinking_false_preserves_fact_messages(
        self, conn, session_with_thinking, tmp_path
    ):
        # The truncate is staging-only -- fact rows (which the user actually
        # queries) survive.
        run_v15_etl(
            conn, session_with_thinking,
            project_name="x", parquet_lake_root=tmp_path / "lake",
            include_thinking=False,
        )
        n = conn.execute(
            "SELECT COUNT(*) FROM fact_messages WHERE session_id = 'think-s'"
        ).fetchone()[0]
        assert n == 2  # user + assistant


class TestCliAcceptsNoThinkingOnDuckdb:
    """CLI honesty: --no-thinking is accepted on --format duckdb and
    --format json (was: rejected with click.UsageError). The semantic is
    "clear staging of thinking-bearing JSON"; fact_messages.content_text
    is already thinking-free regardless of the flag.
    """

    def test_no_thinking_accepted_on_duckdb(
        self, session_with_thinking, tmp_path
    ):
        runner = CliRunner()
        db_path = tmp_path / "out.duckdb"
        result = runner.invoke(
            cli,
            [str(session_with_thinking), "--format", "duckdb",
             "-o", str(db_path), "--no-thinking"],
        )
        assert result.exit_code == 0, result.output
        assert db_path.exists()
        # Stored DB has no thinking payload anywhere queryable.
        import duckdb as _duckdb
        conn = _duckdb.connect(str(db_path))
        stg_n = conn.execute("SELECT COUNT(*) FROM stg_log_entries").fetchone()[0]
        assert stg_n == 0, "stg_log_entries should be empty after --no-thinking"
        fm = conn.execute(
            "SELECT content_text FROM fact_messages "
            "WHERE session_id = 'think-s' AND message_type = 'assistant'"
        ).fetchone()
        assert _THINKING_PAYLOAD not in (fm[0] or "")
        conn.close()

    def test_no_thinking_accepted_on_json(
        self, session_with_thinking, tmp_path
    ):
        runner = CliRunner()
        out = tmp_path / "out-json"
        result = runner.invoke(
            cli,
            [str(session_with_thinking), "--format", "json",
             "-o", str(out), "--no-thinking"],
        )
        assert result.exit_code == 0, result.output
        assert (out / "meta.json").exists()
        # The JSON export already skips stg_log_entries, but verify the
        # combined archive contains no thinking payload anywhere.
        leaked = []
        for path in out.rglob("*.json"):
            if _THINKING_PAYLOAD in path.read_text():
                leaked.append(path.name)
        assert leaked == [], (
            f"Thinking payload leaked into JSON files: {leaked}"
        )
