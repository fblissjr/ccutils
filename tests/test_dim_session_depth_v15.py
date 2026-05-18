"""Tests for dim_session.depth_level computation (Phase D).

depth_level walks the parent_session_key chain:
- Root sessions (is_agent=FALSE or parent_session_key=NULL): depth 0
- Subagents: depth = parent.depth_level + 1

Computed as a post-pass UPDATE after subagent enrichment.
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _subagent_layout(tmp_path, parent_session_id, agent_id, project="proj"):
    parent_dir = (
        tmp_path / "projects" / f"-Users-dev-{project}"
        / parent_session_id / "subagents"
    )
    parent_dir.mkdir(parents=True, exist_ok=True)
    agent_jsonl = parent_dir / f"agent-{agent_id}.jsonl"
    agent_jsonl.write_text(json.dumps({
        "type": "user", "uuid": "u1", "sessionId": f"agent-{agent_id}",
        "timestamp": "2026-04-19T10:00:00Z",
        "cwd": "/work", "gitBranch": "main", "version": "2.1.114",
        "message": {"role": "user", "content": "explore"}}))
    return agent_jsonl


class TestDimSessionDepth:
    def test_root_session_depth_zero(self, conn, tmp_path):
        regular = tmp_path / "regular.jsonl"
        regular.write_text(json.dumps({
            "type": "user", "uuid": "u1", "sessionId": "root-s",
            "timestamp": "2026-04-19T10:00:00Z",
            "cwd": "/p", "gitBranch": "main", "version": "2.1.114",
            "message": {"role": "user", "content": "hi"}}))
        run_v15_etl(conn, regular, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        depth = conn.execute(
            "SELECT depth_level FROM dim_session WHERE session_id = 'root-s'"
        ).fetchone()[0]
        assert depth == 0

    def test_subagent_depth_one_when_parent_present(self, conn, tmp_path):
        # First load a regular root session that has the right session_id
        # for the subagent's parent.
        root = tmp_path / "root.jsonl"
        root.write_text(json.dumps({
            "type": "user", "uuid": "ru1", "sessionId": "parent-root",
            "timestamp": "2026-04-19T09:00:00Z",
            "cwd": "/work", "gitBranch": "main", "version": "2.1.114",
            "message": {"role": "user", "content": "root"}}))
        run_v15_etl(conn, root, project_name="test",
                    parquet_lake_root=tmp_path / "lake")

        # Now load a subagent under parent-root
        sub = _subagent_layout(tmp_path, "parent-root", "agA")
        run_v15_etl(conn, sub, project_name="test",
                    parquet_lake_root=tmp_path / "lake")

        row = conn.execute(
            """
            SELECT depth_level FROM dim_session
            WHERE session_id = 'agent-agA'
            """
        ).fetchone()
        assert row[0] == 1

    def test_nested_subagent_depth_two(self, conn, tmp_path):
        """Subagent A under root, subagent B under A -> B depth = 2."""
        # Root
        root = tmp_path / "root.jsonl"
        root.write_text(json.dumps({
            "type": "user", "uuid": "ru1", "sessionId": "parent-root",
            "timestamp": "2026-04-19T09:00:00Z",
            "cwd": "/work", "gitBranch": "main", "version": "2.1.114",
            "message": {"role": "user", "content": "root"}}))
        run_v15_etl(conn, root, project_name="test",
                    parquet_lake_root=tmp_path / "lake")

        # Subagent A under root
        subA = _subagent_layout(tmp_path, "parent-root", "agA")
        run_v15_etl(conn, subA, project_name="test",
                    parquet_lake_root=tmp_path / "lake")

        # Subagent B under A (parent uuid = subagent A's session_id)
        subB = _subagent_layout(tmp_path, "agent-agA", "agB")
        run_v15_etl(conn, subB, project_name="test",
                    parquet_lake_root=tmp_path / "lake")

        row = conn.execute(
            "SELECT depth_level FROM dim_session WHERE session_id = 'agent-agB'"
        ).fetchone()
        assert row[0] == 2

    def test_subagent_with_unresolved_parent_stays_at_zero(
        self, conn, tmp_path
    ):
        """If the parent session hasn't been ETL'd yet, depth_level
        stays at 0 (cannot resolve)."""
        sub = _subagent_layout(tmp_path, "missing-parent", "agX")
        run_v15_etl(conn, sub, project_name="test",
                    parquet_lake_root=tmp_path / "lake")
        row = conn.execute(
            "SELECT is_agent, depth_level FROM dim_session "
            "WHERE session_id = 'agent-agX'"
        ).fetchone()
        assert row[0] is True
        # depth defaults to 0 when parent can't be resolved
        assert row[1] == 0
