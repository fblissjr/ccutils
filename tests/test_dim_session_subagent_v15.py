"""Tests for subagent enrichment on dim_session (Phase D follow-up).

Subagent JSONL files live at:
    .../projects/<project>/<parent-session-uuid>/subagents/agent-<id>.jsonl

with an optional sidecar .meta.json containing agentType + description.
The v0.15 populator detects subagent sessions by source_path shape during
_upsert_minimal_dimensions and sets:
- is_agent = TRUE
- agent_id = the agent-<id> part of the filename
- parent_session_key = md5(parent-session-uuid-from-path)
- agent_type / agent_description from the sidecar (if present)
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _subagent_layout(tmp_path, parent_session_id, agent_id, *, with_meta=True):
    """Create a fake subagent JSONL + optional sidecar."""
    parent_dir = (
        tmp_path / "projects" / "-Users-dev-myrepo"
        / parent_session_id / "subagents"
    )
    parent_dir.mkdir(parents=True, exist_ok=True)
    agent_jsonl = parent_dir / f"agent-{agent_id}.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": f"agent-{agent_id}",
         "timestamp": "2026-04-19T10:00:00Z",
         "cwd": "/work", "gitBranch": "main", "version": "2.1.114",
         "message": {"role": "user", "content": "explore"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": f"agent-{agent_id}",
         "timestamp": "2026-04-19T10:00:01Z", "requestId": "r1",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
                     "content": [{"type": "text", "text": "found stuff"}]}},
    ]
    agent_jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    if with_meta:
        meta_path = parent_dir / f"agent-{agent_id}.meta.json"
        meta_path.write_text(json.dumps({
            "agentType": "Explore",
            "description": "Explore plan data capture",
        }))
    return agent_jsonl


class TestSubagentDimSessionEnrichment:
    def test_is_agent_set_for_subagent_jsonl(self, conn, tmp_path):
        agent_jsonl = _subagent_layout(
            tmp_path, "parent-sess-uuid", "ac862efa4d26671f8",
        )
        run_v15_etl(conn, agent_jsonl, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT is_agent FROM dim_session "
            "WHERE session_id LIKE 'agent-%'"
        ).fetchone()
        assert row[0] is True

    def test_agent_id_extracted_from_path(self, conn, tmp_path):
        agent_jsonl = _subagent_layout(
            tmp_path, "parent-sess-uuid", "ac862efa4d26671f8",
        )
        run_v15_etl(conn, agent_jsonl, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT agent_id FROM dim_session WHERE is_agent = TRUE"
        ).fetchone()
        assert row[0] == "ac862efa4d26671f8"

    def test_parent_session_key_from_path(self, conn, tmp_path):
        agent_jsonl = _subagent_layout(
            tmp_path, "parent-sess-uuid", "ac862efa4d26671f8",
        )
        run_v15_etl(conn, agent_jsonl, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT parent_session_key FROM dim_session WHERE is_agent = TRUE"
        ).fetchone()
        # md5('parent-sess-uuid')
        expected = conn.execute(
            "SELECT md5('parent-sess-uuid')"
        ).fetchone()[0]
        assert row[0] == expected

    def test_agent_type_and_description_from_meta_json(self, conn, tmp_path):
        agent_jsonl = _subagent_layout(
            tmp_path, "parent-sess-uuid", "ac862efa4d26671f8",
            with_meta=True,
        )
        run_v15_etl(conn, agent_jsonl, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT agent_type, agent_description FROM dim_session "
            "WHERE is_agent = TRUE"
        ).fetchone()
        assert row[0] == "Explore"
        assert row[1] == "Explore plan data capture"

    def test_missing_meta_json_leaves_type_null(self, conn, tmp_path):
        agent_jsonl = _subagent_layout(
            tmp_path, "parent-sess-uuid", "ac862efa4d26671f8",
            with_meta=False,
        )
        run_v15_etl(conn, agent_jsonl, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT agent_type, agent_description FROM dim_session "
            "WHERE is_agent = TRUE"
        ).fetchone()
        assert row[0] is None
        assert row[1] is None

    def test_non_subagent_session_stays_non_agent(self, conn, tmp_path):
        # A regular session at the project root (not under /subagents/)
        regular_jsonl = tmp_path / "regular.jsonl"
        regular_jsonl.write_text("\n".join(json.dumps(d) for d in [
            {"type": "user", "uuid": "u1", "sessionId": "regular-sess",
             "timestamp": "2026-04-19T10:00:00Z",
             "cwd": "/work", "gitBranch": "main", "version": "2.1.114",
             "message": {"role": "user", "content": "hi"}},
        ]))
        run_v15_etl(conn, regular_jsonl, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT is_agent, agent_id, parent_session_key FROM dim_session"
        ).fetchone()
        assert row[0] is False
        assert row[1] is None
        assert row[2] is None
