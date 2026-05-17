# path-privacy: skip-file -- generic /Users/fred and /Users/dev placeholders only
"""Tests for history.jsonl parser and ETL."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ccutils import create_star_schema, run_star_schema_etl
from ccutils.parsers.history import HistoryEntry, iter_history_entries
from ccutils.schemas.star.history_etl import load_history


def _write_history(entries: list[dict]) -> Path:
    """Write a list of dicts as JSONL to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False)
    for entry in entries:
        tmp.write(json.dumps(entry) + "\n")
    tmp.close()
    return Path(tmp.name)


class TestIterHistoryEntries:
    def test_parses_basic_entry(self):
        path = _write_history(
            [
                {
                    "display": "Help me fix the auth bug",
                    "pastedContents": {},
                    "timestamp": 1759085988865,
                    "project": "/Users/dev/workspace/myproject",
                }
            ]
        )
        entries = list(iter_history_entries(path))
        assert len(entries) == 1
        assert entries[0].display == "Help me fix the auth bug"
        assert entries[0].project_path == "/Users/dev/workspace/myproject"
        assert entries[0].project_name == "myproject"
        assert entries[0].has_pasted_content is False

    def test_parses_timestamp(self):
        path = _write_history(
            [
                {
                    "display": "test",
                    "pastedContents": {},
                    "timestamp": 1759085988865,
                    "project": "/Users/dev/workspace/project",
                }
            ]
        )
        entries = list(iter_history_entries(path))
        assert entries[0].timestamp is not None
        assert isinstance(entries[0].timestamp, datetime)

    def test_extracts_session_id(self):
        path = _write_history(
            [
                {
                    "display": "test",
                    "pastedContents": {},
                    "timestamp": 1759085988865,
                    "project": "/Users/dev/workspace/project",
                    "sessionId": "abc-123-def",
                }
            ]
        )
        entries = list(iter_history_entries(path))
        assert entries[0].session_id == "abc-123-def"

    def test_missing_session_id_is_none(self):
        path = _write_history(
            [
                {
                    "display": "test",
                    "pastedContents": {},
                    "timestamp": 1759085988865,
                    "project": "/Users/dev/workspace/project",
                }
            ]
        )
        entries = list(iter_history_entries(path))
        assert entries[0].session_id is None

    def test_detects_pasted_content(self):
        path = _write_history(
            [
                {
                    "display": "test",
                    "pastedContents": {"file.txt": "some content"},
                    "timestamp": 1759085988865,
                    "project": "/Users/dev/workspace/project",
                }
            ]
        )
        entries = list(iter_history_entries(path))
        assert entries[0].has_pasted_content is True

    def test_empty_file(self):
        path = _write_history([])
        entries = list(iter_history_entries(path))
        assert entries == []

    def test_skips_malformed_lines(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False)
        tmp.write("not json\n")
        tmp.write(
            json.dumps(
                {
                    "display": "good",
                    "pastedContents": {},
                    "timestamp": 1759085988865,
                    "project": "/Users/dev/workspace/project",
                }
            )
            + "\n"
        )
        tmp.close()
        entries = list(iter_history_entries(Path(tmp.name)))
        assert len(entries) == 1
        assert entries[0].display == "good"

    def test_project_name_extraction(self):
        path = _write_history(
            [
                {
                    "display": "test",
                    "pastedContents": {},
                    "timestamp": 1759085988865,
                    "project": "/Users/dev/workspace/deep/nested/project-name",
                },
                {
                    "display": "test2",
                    "pastedContents": {},
                    "timestamp": 1759085988866,
                    "project": "",
                },
            ]
        )
        entries = list(iter_history_entries(path))
        assert entries[0].project_name == "project-name"
        assert entries[1].project_name is None

    def test_multiple_entries_ordered(self):
        path = _write_history(
            [
                {
                    "display": "first",
                    "pastedContents": {},
                    "timestamp": 1000000000000,
                    "project": "/Users/dev/workspace/project",
                },
                {
                    "display": "second",
                    "pastedContents": {},
                    "timestamp": 2000000000000,
                    "project": "/Users/dev/workspace/project",
                },
            ]
        )
        entries = list(iter_history_entries(path))
        assert len(entries) == 2
        assert entries[0].display == "first"
        assert entries[1].display == "second"


class TestHistoryETL:
    """Tests for loading history.jsonl into dim_prompt via ETL."""

    def test_loads_prompts_into_dim_prompt(self, output_dir):
        history_path = _write_history(
            [
                {
                    "display": "Fix the bug",
                    "pastedContents": {},
                    "timestamp": 1759085988865,
                    "project": "/Users/dev/workspace/project",
                    "sessionId": "sess-001",
                },
                {
                    "display": "Add a feature",
                    "pastedContents": {},
                    "timestamp": 1759086000000,
                    "project": "/Users/dev/workspace/project",
                },
            ]
        )
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        load_history(conn, history_path)

        count = conn.execute("SELECT COUNT(*) FROM dim_prompt").fetchone()[0]
        assert count == 2
        conn.close()

    def test_links_to_sessions_via_session_id(self, sample_session_file, output_dir):
        """When a history entry has sessionId matching dim_session, session_key links."""
        # First load a session to create a dim_session record
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        # Get the session_key that was created
        session_key = conn.execute(
            "SELECT session_key FROM dim_session"
        ).fetchone()[0]

        # Create history with matching sessionId
        history_path = _write_history(
            [
                {
                    "display": "Help me write a hello world program",
                    "pastedContents": {},
                    "timestamp": 1759085988865,
                    "project": "/Users/dev/workspace/project",
                    "sessionId": sample_session_file.stem,
                },
            ]
        )
        load_history(conn, history_path)

        result = conn.execute(
            "SELECT session_key FROM dim_prompt WHERE session_key IS NOT NULL"
        ).fetchone()
        assert result is not None
        assert result[0] == session_key
        conn.close()

    def test_semantic_prompt_history_view(self, output_dir):
        history_path = _write_history(
            [
                {
                    "display": "Test prompt",
                    "pastedContents": {},
                    "timestamp": 1759085988865,
                    "project": "/Users/dev/workspace/project",
                },
            ]
        )
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        load_history(conn, history_path)

        rows = conn.execute("SELECT * FROM semantic_prompt_history").fetchall()
        assert len(rows) == 1
        conn.close()

    def test_idempotent_reload(self, output_dir):
        history_path = _write_history(
            [
                {
                    "display": "Test",
                    "pastedContents": {},
                    "timestamp": 1759085988865,
                    "project": "/Users/dev/workspace/project",
                },
            ]
        )
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        load_history(conn, history_path)
        load_history(conn, history_path)

        count = conn.execute("SELECT COUNT(*) FROM dim_prompt").fetchone()[0]
        assert count == 1  # Not duplicated
        conn.close()

    def test_nonexistent_history_file(self, output_dir):
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        load_history(conn, output_dir / "nonexistent.jsonl")

        count = conn.execute("SELECT COUNT(*) FROM dim_prompt").fetchone()[0]
        assert count == 0
        conn.close()
