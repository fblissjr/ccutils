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
from helpers_ccutils import write_minimal_session

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


class TestRealContractParentSessionId:
    """The REAL Claude Code contract (verified across the entire on-disk
    corpus): subagent JSONL entries carry
    the PARENT's sessionId, not their own. The transcript's identity comes
    from the file (agent-<id>), derived at staging load -- otherwise every
    agent collapses into its parent's dim_session row, the parent gets
    marked is_agent with a self-referencing parent_session_key, and depth
    flattens to 0 corpus-wide."""

    def _real_layout(self, tmp_path, parent_session_id, agent_id):
        proj = tmp_path / "-home-user-projects-proj"
        proj.mkdir(parents=True, exist_ok=True)
        parent_file = write_minimal_session(
            proj / f"{parent_session_id}.jsonl", parent_session_id
        )

        sub_dir = proj / parent_session_id / "subagents"
        sub_dir.mkdir(parents=True, exist_ok=True)
        # REAL contract: the agent file's entries carry the PARENT id.
        agent_file = write_minimal_session(
            sub_dir / f"agent-{agent_id}.jsonl", f"agent-{agent_id}",
            ts_base="2026-04-19T10:01",
            model="claude-haiku-4-5",
            entry_session_id=parent_session_id,
        )
        return parent_file, agent_file

    def test_agent_transcript_gets_own_session_row(self, conn, tmp_path):
        parent_file, agent_file = self._real_layout(
            tmp_path, "9f0e1d2c-real-parent", "abc123"
        )
        lake = tmp_path / "lake"
        run_v15_etl(conn, parent_file, parquet_lake_root=lake)
        run_v15_etl(conn, agent_file, parquet_lake_root=lake)

        rows = conn.execute(
            "SELECT session_id, is_agent, agent_id, parent_session_key, "
            "depth_level FROM dim_session ORDER BY is_agent"
        ).fetchall()
        assert len(rows) == 2

        parent_row, agent_row = rows
        assert parent_row[0] == "9f0e1d2c-real-parent"
        assert parent_row[1] is False          # parent NOT mislabeled
        assert parent_row[4] == 0

        assert agent_row[0] == "agent-abc123"  # identity from the file
        assert agent_row[1] is True
        assert agent_row[2] == "abc123"
        assert agent_row[3] == conn.execute(
            "SELECT md5('9f0e1d2c-real-parent')"
        ).fetchone()[0]
        assert agent_row[4] == 1               # real depth, not 0

    def test_parent_metrics_not_inflated_by_agent_content(self, conn, tmp_path):
        parent_file, agent_file = self._real_layout(
            tmp_path, "8e9f0a1b-real-parent", "def456"
        )
        lake = tmp_path / "lake"
        run_v15_etl(conn, parent_file, parquet_lake_root=lake)
        run_v15_etl(conn, agent_file, parquet_lake_root=lake)

        counts = dict(conn.execute(
            "SELECT session_id, COUNT(*) FROM fact_messages GROUP BY 1"
        ).fetchall())
        assert counts == {"8e9f0a1b-real-parent": 2, "agent-def456": 2}

        summary_msgs = conn.execute(
            "SELECT total_messages FROM fact_session_summary fss "
            "JOIN dim_session ds USING (session_key) "
            "WHERE ds.session_id = '8e9f0a1b-real-parent'"
        ).fetchone()[0]
        assert summary_msgs == 2               # parent's own messages only
