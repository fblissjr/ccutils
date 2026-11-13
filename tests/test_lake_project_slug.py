"""An agent transcript lands in the Parquet lake under its project, not "subagents".

The lake partitions by `projects/<project_slug>/sessions/<session_id>/`. The
named-files and picker path passed `session_file.parent.name` as the slug,
and the writer's own default did the same, so every agent file (which lives
at `<project>/<parent-uuid>/subagents/agent-<id>.jsonl`) was written under
`projects/subagents/`, and a workflow agent under its workflow id. Agents of
every project shared one partition and none sat beside its parent.

The warehouse was unaffected: dim_project derives from source_path through
project_dir_sql. The slug now comes from the same walk-up discovery uses
(`parsers/discovery.py::project_folder_for`), not a third copy of the rule.

What breaks without it: a lake reader partitioning by project finds every
agent of every project under one directory named "subagents".
"""

from helpers_ccutils import write_minimal_session

from ccutils import create_star_schema
from ccutils.cli.local import _etl_session_files
from ccutils.parsers.discovery import project_folder_for
from ccutils.parsers.parquet_writer import write_session_to_parquet

PROJECT = "-home-user-proj"
PARENT = "11111111-2222-3333-4444-555555555555"


def _layout(tmp_path):
    proj = tmp_path / "projects" / PROJECT
    sub = proj / PARENT / "subagents"
    wf = sub / "workflows" / "wf-123"
    wf.mkdir(parents=True)
    parent = write_minimal_session(proj / f"{PARENT}.jsonl", PARENT)
    agent = write_minimal_session(
        sub / "agent-a1.jsonl", "agent-a1", entry_session_id=PARENT
    )
    wf_agent = write_minimal_session(
        wf / "agent-w1.jsonl", "agent-w1", entry_session_id=PARENT
    )
    return parent, agent, wf_agent


def test_project_folder_for_walks_past_subagents(tmp_path):
    parent, agent, wf_agent = _layout(tmp_path)
    for f in (parent, agent, wf_agent):
        assert project_folder_for(f).name == PROJECT


def test_the_writer_default_slug_is_the_project(tmp_path):
    _, agent, _ = _layout(tmp_path)
    log_path, _ = write_session_to_parquet(agent, tmp_path / "lake")
    assert log_path.parent.parent.parent.name == PROJECT


def test_named_files_put_agents_under_their_project(tmp_path):
    files = list(_layout(tmp_path))
    lake = tmp_path / "lake"
    conn = create_star_schema(tmp_path / "w.duckdb")
    failures = _etl_session_files(
        conn, files, project_name=None, parquet_lake=lake,
        facet_extractor=None, include_thinking=True, output_format="duckdb",
    )
    assert failures == []
    partitions = sorted(
        (p.parent.parent.name, p.name)
        for p in (lake / "projects").glob("*/sessions/*")
    )
    assert partitions == sorted([
        (PROJECT, PARENT), (PROJECT, "agent-a1"), (PROJECT, "agent-w1"),
    ])
