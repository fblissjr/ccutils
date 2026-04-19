"""Tests for the Parquet writer (Phase A4)."""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from ccutils.parsers.parquet_writer import (
    PARSER_VERSION,
    RECORD_SOURCE_CLAUDE_CODE_JSONL,
    make_etl_run_id,
    write_session_to_parquet,
)


@pytest.fixture
def sample_jsonl(tmp_path):
    """Synthesized session covering the entry types we care about most."""
    jsonl = tmp_path / "session-test.jsonl"
    lines = [
        {"type": "summary", "summary": "test", "leafUuid": "x"},
        {
            "type": "user", "uuid": "u1", "parentUuid": None,
            "sessionId": "session-test", "timestamp": "2026-04-19T10:00:00Z",
            "cwd": "/p", "gitBranch": "main", "version": "2.1.114", "userType": "external",
            "entrypoint": "cli", "isSidechain": False,
            "message": {"role": "user", "content": "hi"},
        },
        {
            "type": "assistant", "uuid": "a1", "parentUuid": "u1",
            "sessionId": "session-test", "timestamp": "2026-04-19T10:00:05Z",
            "requestId": "req_abc",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "toolu_001", "name": "Bash",
                     "input": {"command": "ls"}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        },
        {
            "type": "user", "uuid": "u2", "parentUuid": "a1",
            "sessionId": "session-test", "timestamp": "2026-04-19T10:00:06Z",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_001", "content": "ok"},
            ]},
            "toolUseResult": {
                "stdout": "file1\nfile2\n", "stderr": "",
                "interrupted": False, "exitCode": 0,
            },
        },
        {
            "type": "system", "subtype": "turn_duration",
            "uuid": "s1", "sessionId": "session-test",
            "timestamp": "2026-04-19T10:00:07Z",
            "durationMs": 1234, "messageCount": 3,
        },
        {
            "type": "attachment",
            "uuid": "att1", "sessionId": "session-test",
            "timestamp": "2026-04-19T10:00:08Z",
            "attachment": {"type": "diagnostics", "files": []},
        },
        {
            "type": "permission-mode",
            "sessionId": "session-test",
            "permissionMode": "acceptEdits",
        },
        {
            "type": "custom-title",
            "sessionId": "session-test",
            "customTitle": "test-title",
        },
        {
            "type": "progress",
            "sessionId": "session-test",
            "toolUseID": "toolu_001",
            "data": {"type": "bash_progress", "stdout": "running..."},
        },
        # Forward-compat: an unknown future-thing entry should also write
        # without exception.
        {
            "type": "future-fancy-thing",
            "sessionId": "session-test",
            "weirdField": "weird value",
        },
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


class TestWriteSessionToParquet:
    def test_writes_two_parquet_files(self, sample_jsonl, tmp_path):
        out_root = tmp_path / "lake"
        log_path, meta_path = write_session_to_parquet(
            sample_jsonl, out_root, project_slug="test-project",
        )
        assert log_path.exists()
        assert meta_path.exists()
        assert log_path.parent == out_root / "projects" / "test-project" / "sessions" / "session-test"

    def test_log_entries_row_count_matches_input(self, sample_jsonl, tmp_path):
        log_path, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake", project_slug="p")
        table = pq.read_table(log_path)
        assert table.num_rows == 10  # all 10 lines, including the unknown future type

    def test_lineage_columns_stamped_on_every_row(self, sample_jsonl, tmp_path):
        log_path, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake", project_slug="p")
        table = pq.read_table(log_path)
        for col in ("etl_run_id", "parsed_at", "parser_version", "record_source", "entry_id", "source_path", "sequence_num"):
            assert col in table.column_names
            # Every row has a non-null value
            assert all(v is not None for v in table.column(col).to_pylist())

    def test_envelope_fields_typed(self, sample_jsonl, tmp_path):
        log_path, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake", project_slug="p")
        table = pq.read_table(log_path).to_pylist()
        # Find the user entry "u1"
        u1 = next(r for r in table if r["uuid"] == "u1")
        assert u1["type"] == "user"
        assert u1["session_id"] == "session-test"
        assert u1["cwd"] == "/p"
        assert u1["entrypoint"] == "cli"
        assert u1["is_sidechain"] is False

    def test_message_payload_preserved_as_json(self, sample_jsonl, tmp_path):
        log_path, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake", project_slug="p")
        table = pq.read_table(log_path).to_pylist()
        a1 = next(r for r in table if r["uuid"] == "a1")
        msg = json.loads(a1["message_json"])
        assert msg["role"] == "assistant"
        assert msg["stop_reason"] == "tool_use"
        assert len(msg["content"]) == 2

    def test_tool_use_result_preserved(self, sample_jsonl, tmp_path):
        """R1: structured toolUseResult must round-trip through Parquet."""
        log_path, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake", project_slug="p")
        table = pq.read_table(log_path).to_pylist()
        u2 = next(r for r in table if r["uuid"] == "u2")
        result = json.loads(u2["tool_use_result_json"])
        assert result["stdout"] == "file1\nfile2\n"
        assert result["interrupted"] is False
        assert result["exitCode"] == 0

    def test_system_subtype_split_to_typed_column(self, sample_jsonl, tmp_path):
        log_path, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake", project_slug="p")
        table = pq.read_table(log_path).to_pylist()
        s1 = next(r for r in table if r["uuid"] == "s1")
        assert s1["system_subtype"] == "turn_duration"
        # System payload preserved as JSON for ETL extraction
        payload = json.loads(s1["system_payload_json"])
        assert payload["durationMs"] == 1234

    def test_progress_data_preserved(self, sample_jsonl, tmp_path):
        log_path, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake", project_slug="p")
        table = pq.read_table(log_path).to_pylist()
        p_rows = [r for r in table if r["type"] == "progress"]
        assert len(p_rows) == 1
        data = json.loads(p_rows[0]["progress_data_json"])
        assert data["type"] == "bash_progress"
        assert data["stdout"] == "running..."

    def test_attachment_preserved(self, sample_jsonl, tmp_path):
        log_path, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake", project_slug="p")
        table = pq.read_table(log_path).to_pylist()
        att = next(r for r in table if r["uuid"] == "att1")
        attachment = json.loads(att["attachment_json"])
        assert attachment["type"] == "diagnostics"

    def test_meta_payload_for_permission_mode(self, sample_jsonl, tmp_path):
        log_path, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake", project_slug="p")
        table = pq.read_table(log_path).to_pylist()
        pm = next(r for r in table if r["type"] == "permission-mode")
        meta = json.loads(pm["meta_payload_json"])
        assert meta["permission_mode"] == "acceptEdits"

    def test_meta_payload_for_custom_title(self, sample_jsonl, tmp_path):
        log_path, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake", project_slug="p")
        table = pq.read_table(log_path).to_pylist()
        ct = next(r for r in table if r["type"] == "custom-title")
        meta = json.loads(ct["meta_payload_json"])
        assert meta["customTitle"] == "test-title"

    def test_unknown_entry_type_writes_without_error(self, sample_jsonl, tmp_path):
        """Forward-compat: an unrecognized entry type should land in Parquet
        with type='future-fancy-thing' preserved, no payload columns set."""
        log_path, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake", project_slug="p")
        table = pq.read_table(log_path).to_pylist()
        future = next(r for r in table if r["type"] == "future-fancy-thing")
        assert future["raw_json"] is not None
        # extras_json should hold weirdField (it's not in the envelope)
        if future["extras_json"]:
            extras = json.loads(future["extras_json"])
            assert "weirdField" in extras or "weirdField" in json.loads(future["raw_json"])

    def test_sequence_num_is_strictly_monotonic(self, sample_jsonl, tmp_path):
        log_path, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake", project_slug="p")
        table = pq.read_table(log_path).to_pylist()
        seqs = [r["sequence_num"] for r in table]
        assert seqs == list(range(len(seqs)))

    def test_entry_ids_are_deterministic_across_runs(self, sample_jsonl, tmp_path):
        log_path1, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake1", project_slug="p")
        log_path2, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake2", project_slug="p")
        ids1 = pq.read_table(log_path1).column("entry_id").to_pylist()
        ids2 = pq.read_table(log_path2).column("entry_id").to_pylist()
        assert ids1 == ids2

    def test_session_meta_has_one_row(self, sample_jsonl, tmp_path):
        _, meta_path = write_session_to_parquet(sample_jsonl, tmp_path / "lake", project_slug="p")
        meta = pq.read_table(meta_path).to_pylist()
        assert len(meta) == 1
        assert meta[0]["session_id"] == "session-test"
        assert meta[0]["entry_count"] == 10
        assert meta[0]["project_slug"] == "p"
        assert meta[0]["cwd"] == "/p"

    def test_record_source_default(self, sample_jsonl, tmp_path):
        log_path, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake", project_slug="p")
        table = pq.read_table(log_path).to_pylist()
        assert all(r["record_source"] == RECORD_SOURCE_CLAUDE_CODE_JSONL for r in table)

    def test_parser_version_stamped(self, sample_jsonl, tmp_path):
        log_path, _ = write_session_to_parquet(sample_jsonl, tmp_path / "lake", project_slug="p")
        table = pq.read_table(log_path).to_pylist()
        assert all(r["parser_version"] == PARSER_VERSION for r in table)


class TestArchiveSmokeWrite:
    """Write the user's full archive through the Parquet writer to surface
    schema validation errors, dtype mismatches, or unexpected payload shapes."""

    def test_writes_full_archive(self, tmp_path):
        archive_dir = Path.home() / ".claude" / "projects"
        if not archive_dir.exists():
            pytest.skip("Archive dir not present")
        out_root = tmp_path / "lake"
        etl_run_id = make_etl_run_id()
        files = sorted(archive_dir.glob("**/*.jsonl"))
        write_count = 0
        errors = []
        for fp in files[:20]:  # cap at 20 files for test speed
            try:
                write_session_to_parquet(
                    fp, out_root,
                    etl_run_id=etl_run_id,
                    project_slug=fp.parent.name or "unknown",
                )
                write_count += 1
            except Exception as e:
                errors.append((str(fp.name), type(e).__name__, str(e)[:200]))
        if errors:
            print(f"Errors during write: {errors[:5]}")
        assert write_count > 0
        assert not errors, f"{len(errors)} write errors out of {len(files[:20])} files"
