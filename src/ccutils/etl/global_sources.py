"""Sources that are global to the machine rather than per-session.

`dim_prompt` from history.jsonl, auto memory, and the cross-session
delegation reconciliation. All three run once after the per-session loop,
and all three must run on EVERY path that builds a warehouse.

That last part is the whole reason this module exists. These steps used to
live inline in the batch exporter, and the single-session path ran one of
the three -- so `ccutils <file> --format duckdb` produced a warehouse with
`fact_messages.prompt_id` populated and `dim_prompt` empty, an FK pointing
at nothing, plus no memory tables at all. An external audit read that as a
defect in the ETL. It was a defect in the entry point, and the shape of it
-- two callers that were supposed to agree and quietly did not -- is the
same shape as the `local`/`all` split that 0.20.0 removed.

One warehouse means one shape. One function means one shape.
"""

from __future__ import annotations

from pathlib import Path

def _import_auto_memory(conn, *, batch_run_id: str | None = None,
                        claude_home: Path | None = None) -> int:
    """Ingest Claude Code auto memory for the projects this archive covers.

    Scoped to the project directories already in ``dim_project`` so a
    filtered run (``-p mitate``) does not pull in every other project's
    memory from the same machine -- ``dim_project.project_name`` is the
    encoded ``projects/`` directory name, which is exactly the memory
    directory's owner.

    Subagent memory declared ``memory: project`` / ``memory: local`` lives
    in the repository rather than under the Claude home, so repo roots come
    from the sessions themselves (``dim_session.cwd``). A session started in
    a subdirectory of its repo will not surface that repo's committed agent
    memory; the user-scope directory is always scanned.
    """
    from .dim_memory import run_memory_import

    def _scope() -> dict:
        """Resolve the import's arguments. Runs INSIDE the recorded run."""
        home = Path(claude_home) if claude_home else Path.home() / ".claude"
        projects = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT project_name FROM dim_project "
                "WHERE project_name IS NOT NULL"
            ).fetchall()
        }
        repo_paths = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT cwd FROM dim_session WHERE cwd IS NOT NULL"
            ).fetchall()
        ]
        return {
            "projects_root": home / "projects",
            # Deliberately not `projects or None`: None means "every project
            # on the machine", so an empty dim_project would invert the
            # scoping and ingest everything. An empty set ingests nothing.
            "only": projects,
            "agent_user_root": home / "agent-memory",
            "agent_repo_paths": repo_paths,
        }

    return run_memory_import(conn, batch_run_id=batch_run_id, resolve_kwargs=_scope)


def run_global_sources(conn, *, batch_run_id: str | None = None,
                       claude_home: Path | None = None,
                       scope_to_covered_projects: bool = False) -> None:
    """Run every global source, in dependency order.

    Ordering: reconciliation first, because it derives delegation rollups
    from sessions and nothing else depends on it; then the two additive
    imports.

    Error handling differs per source ON PURPOSE, by whether the output is
    load-bearing. Reconciliation RE-RAISES: a silent failure there puts the
    warehouse back to reporting spawn-acknowledgment latency as agent
    duration, which is worse than no warehouse. History is best-effort and
    swallowed -- it is optional and unrecorded. Memory is additive and
    records its own failure as a failed run rather than vanishing.

    ``claude_home`` is injected rather than read from ``Path.home()`` here so
    a test can point the whole thing at a temp directory. Reading the real
    home inside made every warehouse-building test depend on the developer's
    machine state -- and slow, since it parsed a multi-megabyte history file.

    ``scope_to_covered_projects`` keeps history to the projects the warehouse
    already covers. The caller decides, because the answer follows INTENT:

    - a subset build (named PATHS, the picker, or ``-p``) must not carry
      other projects' prompts -- a one-session warehouse held 11,606 prompts
      from 103 projects before this;
    - a full-corpus build asked for everything, and scoping it is NOT the
      no-op it looks like. Measured: forward-encoding matches 88.9% of
      history entries to a project directory, so scoping a full build would
      silently drop the other 11% -- prompts typed in directories that never
      became a session, or whose project has since been pruned.

    Hence the default is False: taking too much is a privacy problem only
    where the user asked for a subset, whereas dropping rows is a data-loss
    problem everywhere.
    """
    from .orchestrator import run_post_session_reconciliation

    home = Path(claude_home) if claude_home else Path.home() / ".claude"

    run_post_session_reconciliation(conn, batch_run_id=batch_run_id)

    try:
        from .dim_prompt import import_history

        import_history(conn, home / "history.jsonl",
                       only_projects=scope_to_covered_projects)
    except Exception:
        # Optional: never fail a build because the global prompt history is
        # missing or malformed.
        pass

    _import_auto_memory(conn, batch_run_id=batch_run_id, claude_home=home)
