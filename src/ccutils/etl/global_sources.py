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

    _run_history_import(
        conn, home / "history.jsonl",
        batch_run_id=batch_run_id,
        only_projects=scope_to_covered_projects,
    )

    _import_auto_memory(conn, batch_run_id=batch_run_id, claude_home=home)


def _run_history_import(conn, history_path, *, batch_run_id, only_projects):
    """Ingest history.jsonl as a RECORDED run, not a bare call.

    CLAUDE.md has named this call out for a while -- "`dim_prompt` is still
    un-wrapped, fix it when touched" -- and it was touched twice: once to
    move it here, once to add scoping.

    The old `except Exception: pass` lost the fact that history was meant to
    be there. A warehouse whose import raised mid-parse, or whose scope
    matched nothing because the encoder was wrong, looked exactly like one
    built on a machine with no history.jsonl: empty `dim_prompt`, no row
    saying so. The scoping change made that worse, because the swallowed
    count was the only signal that scoping had worked.

    Records rather than re-raises, matching `run_memory_import`: history is
    additive, so losing it costs prompt rows and corrupts nothing else, and
    an archive whose sessions all processed should still finish. What must
    not happen is losing the fact that it was attempted.
    """
    from .dim_prompt import import_history
    from .lineage import EtlRun

    run = EtlRun.start(
        conn,
        source_path=str(history_path),
        batch_run_id=batch_run_id,
        description="prompt-history import",
        run_kind="global_source",
    )
    try:
        with run.step("dim_prompt", kind="stage") as counts:
            counts.rows_inserted = import_history(
                conn, history_path, only_projects=only_projects
            )
        run.complete(sessions_seen=0, sessions_inserted=0, sessions_updated=0)
    except (KeyboardInterrupt, SystemExit) as e:
        run.fail(str(e) or type(e).__name__)
        raise
    except Exception as e:
        run.fail(str(e) or type(e).__name__)
    except BaseException as e:
        run.fail(str(e) or type(e).__name__)
        raise
