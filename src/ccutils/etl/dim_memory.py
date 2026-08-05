"""Import Claude Code auto memory into dim_memory / bridge_memory_link.

Not part of ``run_v15_etl`` -- memory directories are per-repository, not
per-session, so this importer is called once after the per-session loop (the
same slot as ``dim_prompt``'s ``import_history``).

**Type 2, because nothing else keeps the history.** Claude Code overwrites
memory files in place; the ``modified:`` frontmatter stamp records only the
last write, and earlier contents survive only in ``file-history`` rollback
checkpoints, which are pruned. Storing one row per file would mean each
re-ingest destroyed the previous state. So each import compares the parsed
content hash against the open row and, when it differs, closes that row
(``valid_to`` / ``is_current = FALSE``) and inserts the next version.

Version identity is ``(memory_id, version_num)``, NOT ``(memory_id,
content_hash)``: a memory reverted to earlier text repeats its hash, and
keying on content would silently drop the revert.

Idempotent. Re-running with nothing changed on disk inserts no rows and
closes none.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from ccutils.etl.lineage import EtlRun, record_source_label
from ccutils.etl.utils import insert_missing_dim_dates
from ccutils.parsers.memory import (
    MemoryFile,
    iter_agent_memories,
    iter_project_memories,
    scanned_owners,
)

__all__ = ["import_memories", "run_memory_import"]


#: source_path value for the memory import's own ``fact_etl_runs`` row.
#: Mirrors ``<post-session-reconciliation>`` -- a run that is not a session
#: still needs a stable, greppable identity in the run table.
MEMORY_RUN_SOURCE = "<auto-memory>"

#: Provenance label stamped on every memory row. Distinct from the run
#: sentinel above: that identifies the RUN, this identifies the SOURCE, and
#: it is validated against lineage.py's allow-list so a typo cannot reach
#: 100k rows.
MEMORY_RECORD_SOURCE = "claude_code_memory"


def _provenance(run: EtlRun | None) -> tuple:
    """(version_key, etl_run_id, record_source) for the INSERTs.

    NULL when no run was supplied, which only unit tests do. The version key
    matters here as much as on any fact: what ``content_hash`` covers and how
    links are extracted are parser semantics, so rows written under different
    parser versions are not interchangeable, and a stale stamp would make
    them indistinguishable.
    """
    if run is None:
        return (None, None, None)
    return (run.version_key, run.etl_run_id, record_source_label(MEMORY_RECORD_SOURCE))


def run_memory_import(
    conn,
    *,
    batch_run_id: str | None = None,
    resolve_kwargs=None,
    **kwargs,
) -> int:
    """Import auto memory as a recorded ETL run. Returns versions written.

    Memory is a per-repository source, not a per-session one, so it cannot
    live inside ``run_v15_etl``. That is a reason to run it after the loop,
    NOT a reason to run it outside the run-metadata system: a global source
    that writes rows without a run is invisible -- nothing reports how many
    memory versions a run wrote, no row says which run observed it, and a
    failure leaves no trace. This wrapper gives it the same three-grain
    treatment every other populator gets, following
    ``run_post_session_reconciliation``.

    **Failures are recorded, not raised.** The reconciliation pass re-raises
    because its output is load-bearing (without it the warehouse reports
    acknowledgment latencies as agent durations). Memory is additive: losing
    it costs the memory rows and corrupts nothing else, so an archive build
    should finish. What must NOT happen is the previous ``except Exception:
    pass`` -- that lost the fact that memory was meant to be there at all,
    leaving a warehouse indistinguishable from one built where auto memory
    was disabled. The run row carries the error instead.
    """
    run = EtlRun.start(
        conn,
        source_path=MEMORY_RUN_SOURCE,
        batch_run_id=batch_run_id,
        description="auto-memory import",
        run_kind="global_source",
    )
    try:
        # Callers whose arguments themselves require work (querying the
        # warehouse for scope, resolving the home directory) pass a callable
        # so that work happens INSIDE the guard. Computing it at the call
        # site would put it outside any recorded boundary, where a raise
        # escapes the enclosing BatchRun and aborts an archive whose sessions
        # were all already processed.
        if resolve_kwargs is not None:
            kwargs = {**kwargs, **resolve_kwargs()}
        with run.step("dim_memory", kind="stage") as counts:
            written = import_memories(conn, run=run, **kwargs)
            counts.rows_inserted = written
        run.complete(sessions_seen=0, sessions_inserted=0, sessions_updated=0)
        return written
    except (KeyboardInterrupt, SystemExit) as e:
        # Mark-and-propagate, matching BatchRun.__exit__. Swallowing these
        # would record a failed run and let the archive sail on to
        # complete(), discarding the user's interrupt.
        run.fail(str(e) or type(e).__name__)
        raise
    except Exception as e:
        run.fail(str(e) or type(e).__name__)
        return 0


def _memory_id(mem: MemoryFile) -> str:
    """Stable identity of a memory FILE across all its versions.

    Scoped by (scope, agent_scope, owner_root, owner) so that memories which
    merely share a file name stay distinct rather than collapsing into one
    memory that appears to flip-flop between bodies on every import. Three
    real collisions this prevents: a project's ``MEMORY.md`` against a
    subagent's; the same subagent declared ``memory: user`` in one place and
    ``memory: project`` in another; and a committed ``reviewer`` agent
    existing in two repositories, which is ordinary since that memory is
    shared with a team.
    """
    return hashlib.md5(
        "\x00".join(
            [
                mem.scope,
                mem.agent_scope or "",
                mem.owner_root or "",
                mem.owner_key,
                mem.file_name,
            ]
        ).encode("utf-8")
    ).hexdigest()


def _memory_key(memory_id: str, version_num: int) -> str:
    return hashlib.md5(f"{memory_id}\x00{version_num}".encode("utf-8")).hexdigest()


def _collect(
    projects_root: str | Path | None,
    only: Iterable[str] | None,
    agent_user_root: str | Path | None,
    agent_repo_paths: Iterable[str | Path],
) -> list[MemoryFile]:
    found: list[MemoryFile] = []
    if projects_root is not None:
        found.extend(iter_project_memories(projects_root, only=only))
    if agent_user_root is not None or agent_repo_paths:
        found.extend(
            iter_agent_memories(
                user_root=agent_user_root, repo_paths=agent_repo_paths
            )
        )
    return found


def _load_open_versions(conn) -> dict[str, tuple[str, str, int]]:
    """memory_id -> (memory_key, content_hash, version_num) for open rows.

    "Open" means ``is_current`` -- the row this import may need to close.
    """
    return {
        row[0]: (row[1], row[2], row[3])
        for row in conn.execute(
            "SELECT memory_id, memory_key, content_hash, version_num "
            "FROM dim_memory WHERE is_current"
        ).fetchall()
    }


def _max_version(conn) -> dict[str, int]:
    """memory_id -> highest version_num ever recorded.

    Read separately from the open row so a closed-then-resurrected memory
    (file deleted, later rewritten) continues its numbering instead of
    restarting at 1 and colliding on memory_key.
    """
    return {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT memory_id, MAX(version_num) FROM dim_memory GROUP BY memory_id"
        ).fetchall()
    }


def import_memories(
    conn,
    *,
    projects_root: str | Path | None = None,
    only: Iterable[str] | None = None,
    agent_user_root: str | Path | None = None,
    agent_repo_paths: Sequence[str | Path] = (),
    now: datetime | None = None,
    run: EtlRun | None = None,
) -> int:
    """Load auto memory into dim_memory. Returns the number of versions written.

    Args:
        conn: Star-schema DuckDB connection.
        projects_root: ``<HOME>/.claude/projects``. Omit to skip project  # path-privacy: ignore
            memory entirely.
        only: Restrict project memory to these encoded project directory
            names. Pass the projects present in ``dim_project`` so a filtered
            archive run does not ingest the whole machine's memory corpus.
        agent_user_root: ``<HOME>/.claude/agent-memory`` (subagent memory  # path-privacy: ignore
            declared ``memory: user``).
        agent_repo_paths: Repository roots to scan for committed and local
            subagent memory.
        now: Injected clock for the SCD boundary; defaults to UTC now. Tests
            pass it so ``valid_from`` / ``valid_to`` are deterministic.
        run: The ``EtlRun`` this import belongs to. Supplied by
            :func:`run_memory_import`, which every real caller goes through;
            rows are stamped with its id and version key so a Type 2 row can
            answer "which run observed this version". Omitted only by unit
            tests exercising the import in isolation, where the provenance
            columns are left NULL.

    Missing directories are a no-op (returns 0): memory is optional and can
    be disabled entirely via ``autoMemoryEnabled: false``.
    """
    stamp = now or datetime.now(timezone.utc)
    found = _collect(projects_root, only, agent_user_root, agent_repo_paths)

    open_rows = _load_open_versions(conn)
    max_versions = _max_version(conn)

    seen_ids: set[str] = set()
    inbound: list[tuple] = []
    links: list[tuple] = []

    for mem in found:
        memory_id = _memory_id(mem)
        seen_ids.add(memory_id)

        existing = open_rows.get(memory_id)
        if existing is not None and existing[1] == mem.content_hash:
            # Unchanged since the last import. Note this does NOT mean the
            # file was untouched -- a rewrite that only moved the `modified:`
            # stamp lands here, which is the point.
            #
            # But the EDGES can still be wrong here, because they are derived
            # from the body by a parser whose contract changes between
            # releases. v0.19.0 started reading `[Title](file.md)` index
            # entries; every one of those lives in an unchanged MEMORY.md, so
            # without this the entire index graph stayed invisible on any
            # upgraded warehouse until the index text happened to change.
            # Re-derive rather than open a version: the memory did not change,
            # our reading of it did, and asserting an edit that never happened
            # would corrupt the history for every memory in the corpus at once.
            _sync_links(conn, memory_key=existing[0], memory_id=memory_id,
                        mem=mem, run=run)
            continue

        version_num = max_versions.get(memory_id, 0) + 1
        max_versions[memory_id] = version_num
        memory_key = _memory_key(memory_id, version_num)

        if existing is not None:
            _close_version(conn, existing[0], stamp)

        try:
            file_mtime = datetime.fromtimestamp(
                mem.source_path.stat().st_mtime, tz=timezone.utc
            )
        except OSError:
            file_mtime = None

        # date_key/time_key describe when the MEMORY was written, not when
        # ccutils happened to observe it -- modified: is the stated value,
        # file mtime the fallback when frontmatter carries none.
        stamp_for_date = mem.modified or file_mtime

        inbound.append(
            (
                memory_key,
                memory_id,
                mem.scope,
                mem.owner_key,
                mem.owner_root,
                mem.agent_scope,
                str(mem.source_path),
                mem.file_name,
                mem.memory_name,
                mem.description,
                mem.memory_type,
                mem.node_type,
                mem.origin_session_id,
                mem.is_index,
                mem.has_frontmatter,
                mem.body_text,
                mem.content_hash,
                mem.body_chars,
                mem.body_lines,
                len(mem.links),
                mem.modified,
                file_mtime,
                version_num,
                stamp,
                int(stamp_for_date.strftime("%Y%m%d")) if stamp_for_date else None,
                int(stamp_for_date.strftime("%H%M")) if stamp_for_date else None,
            )
        )

        links.extend(_link_rows(mem, memory_key, memory_id))

    # A memory file that vanished from disk is closed, never deleted: a
    # retired memory is a fact about the project's history, and erasing the
    # row would make the warehouse forget the memory ever existed.
    #
    # Retirement is scoped to the owners this call SCANNED, not to the ones
    # it got files back from. Those differ exactly when a memory directory
    # was emptied -- and keying off results there would leave every one of
    # its rows marked current forever. It also keeps a filtered run from
    # retiring another project's memory as collateral.
    _close_absent(
        conn,
        seen_ids,
        scanned_owners(
            projects_root=projects_root,
            only=only,
            agent_user_root=agent_user_root,
            agent_repo_paths=agent_repo_paths,
        ),
        stamp,
    )

    if not inbound:
        return 0

    _insert_versions(conn, inbound, run)
    _insert_links(conn, links, run)
    _resolve_project_and_session_keys(conn)
    _resolve_link_targets(conn)
    insert_missing_dim_dates(conn, "dim_memory", "modified_at")
    return len(inbound)


def _link_rows(mem: MemoryFile, memory_key: str, memory_id: str) -> list[tuple]:
    """The bridge rows one memory version should have, in document order."""
    return [
        (
            hashlib.md5(f"{memory_key}\x00{ordinal}".encode("utf-8")).hexdigest(),
            memory_key,
            memory_id,
            mem.scope,
            mem.owner_key,
            link.target,
            link.syntax,
            link.text,
            ordinal,
        )
        for ordinal, link in enumerate(mem.links)
    ]


def _sync_links(conn, *, memory_key: str, memory_id: str, mem: MemoryFile, run) -> None:
    """Re-derive a current version's edges when the PARSER, not the memory, changed.

    Compares the stored edge set against what the current parser extracts and
    rewrites only on a difference, so the ordinary unchanged case stays a
    cheap read. Rewriting is a delete-then-insert of that version's rows:
    edges are a pure projection of the body, ordinals must stay contiguous,
    and a partial update could leave a version holding edges from two
    different parser contracts at once.

    ``link_count`` is updated with them -- it is a denormalised copy of the
    edge count, and leaving it stale would make the column disagree with the
    bridge table it summarises.
    """
    wanted = _link_rows(mem, memory_key, memory_id)
    stored = conn.execute(
        "SELECT target_name, link_syntax, link_text, ordinal "
        "FROM bridge_memory_link WHERE memory_key = ? ORDER BY ordinal",
        [memory_key],
    ).fetchall()

    # Compare on the columns the parser produces; the key columns are derived
    # from (memory_key, ordinal) and cannot differ independently.
    if [(r[5], r[6], r[7], r[8]) for r in wanted] == [tuple(r) for r in stored]:
        return

    conn.execute("DELETE FROM bridge_memory_link WHERE memory_key = ?", [memory_key])
    _insert_links(conn, wanted, run)
    conn.execute(
        "UPDATE dim_memory SET link_count = ? WHERE memory_key = ?",
        [len(wanted), memory_key],
    )


def _close_version(conn, memory_key: str, stamp: datetime) -> None:
    conn.execute(
        "UPDATE dim_memory SET is_current = FALSE, valid_to = ? "
        "WHERE memory_key = ? AND is_current",
        [stamp, memory_key],
    )


def _close_absent(
    conn,
    seen_ids: set[str],
    owners: set[tuple[str, str, str | None]],
    stamp: datetime,
) -> None:
    """Close open rows whose file is gone, within the owners that were scanned.

    Scoped by (scope, owner_key) rather than globally: an import that only
    looked at one project must not retire every other project's memory just
    because it did not see those files.
    """
    for scope, owner_key, owner_root in owners:
        stale = [
            row[0]
            for row in conn.execute(
                "SELECT memory_key, memory_id FROM dim_memory "
                "WHERE is_current AND scope = ? AND owner_key = ? "
                "AND owner_root IS NOT DISTINCT FROM ?",
                [scope, owner_key, owner_root],
            ).fetchall()
            if row[1] not in seen_ids
        ]
        for memory_key in stale:
            _close_version(conn, memory_key, stamp)


def _insert_versions(conn, inbound: list[tuple], run: EtlRun | None) -> None:
    conn.execute("DROP TABLE IF EXISTS _inbound_memory")
    conn.execute(
        """
        CREATE TEMP TABLE _inbound_memory (
            memory_key VARCHAR, memory_id VARCHAR, scope VARCHAR,
            owner_key VARCHAR, owner_root VARCHAR, agent_scope VARCHAR,
            source_path VARCHAR,
            file_name VARCHAR, memory_name VARCHAR, description VARCHAR,
            memory_type VARCHAR, node_type VARCHAR, origin_session_id VARCHAR,
            is_index BOOLEAN, has_frontmatter BOOLEAN, body_text VARCHAR,
            content_hash VARCHAR, body_chars INTEGER, body_lines INTEGER,
            link_count INTEGER, modified_at TIMESTAMP, file_mtime TIMESTAMP,
            version_num INTEGER, valid_from TIMESTAMP,
            date_key INTEGER, time_key INTEGER
        )
        """
    )
    conn.executemany(
        "INSERT INTO _inbound_memory VALUES "
        "(" + ",".join("?" * 26) + ")",
        inbound,
    )
    conn.execute(
        """
        INSERT INTO dim_memory (
            created_by_version_key, etl_run_id, record_source,
            created_at,
            memory_key, memory_id, project_key, session_key, scope, owner_key,
            owner_root, agent_scope, source_path, file_name, memory_name, description,
            memory_type, node_type, origin_session_id, is_index,
            has_frontmatter, body_text, content_hash, body_chars, body_lines,
            link_count, modified_at, file_mtime, version_num, valid_from,
            valid_to, is_current, date_key, time_key
        )
        SELECT
            ?, ?, ?,
            current_timestamp,
            memory_key, memory_id, NULL, NULL, scope, owner_key,
            owner_root, agent_scope, source_path, file_name, memory_name, description,
            memory_type, node_type, origin_session_id, is_index,
            has_frontmatter, body_text, content_hash, body_chars, body_lines,
            link_count, modified_at, file_mtime, version_num, valid_from,
            NULL, TRUE, date_key, time_key
        FROM _inbound_memory
        """,
        _provenance(run),
    )
    conn.execute("DROP TABLE IF EXISTS _inbound_memory")


def _insert_links(conn, links: list[tuple], run: EtlRun | None) -> None:
    if not links:
        return
    conn.executemany(
        """
        INSERT INTO bridge_memory_link (
            created_by_version_key, etl_run_id, record_source, created_at,
            memory_link_key, memory_key, memory_id, project_key, scope,
            owner_key, target_name, target_memory_id, is_resolved,
            link_syntax, link_text, ordinal
        ) VALUES (?, ?, ?, current_timestamp, ?, ?, ?, NULL, ?, ?, ?, NULL, FALSE, ?, ?, ?)
        """,
        [_provenance(run) + row for row in links],
    )


def _resolve_project_and_session_keys(conn) -> None:
    """Fill project_key and session_key on rows that still lack them.

    Project memory is keyed by the encoded ``projects/`` directory name,
    which is exactly ``dim_project.project_name`` (the last segment of
    ``project_path``). Agent-scope memory belongs to no project and keeps a
    NULL project_key.

    Both resolutions are LEFT-JOIN semantics on purpose: an unresolvable
    origin session keeps its raw ``origin_session_id`` with a NULL
    ``session_key``, since the stated id is the only evidence of which
    session wrote the memory.
    """
    conn.execute(
        """
        UPDATE dim_memory SET project_key = (
            SELECT dp.project_key FROM dim_project dp
            WHERE dp.project_name = dim_memory.owner_key LIMIT 1
        )
        WHERE scope = 'project' AND project_key IS NULL
        """
    )
    conn.execute(
        """
        UPDATE dim_memory SET session_key = (
            SELECT ds.session_key FROM dim_session ds
            WHERE ds.session_id = dim_memory.origin_session_id LIMIT 1
        )
        WHERE session_key IS NULL AND origin_session_id IS NOT NULL
        """
    )
    conn.execute(
        """
        UPDATE bridge_memory_link SET project_key = (
            SELECT m.project_key FROM dim_memory m
            WHERE m.memory_key = bridge_memory_link.memory_key LIMIT 1
        )
        WHERE project_key IS NULL
        """
    )


def _resolve_link_targets(conn) -> None:
    """Match each link to a sibling memory in the same scope+owner.

    The two syntaxes resolve differently because they name different things.

    A ``markdown`` target is a FILE NAME the author wrote as a path
    (``[Title](batch-import-notes.md)``), so it matches ``file_name`` exactly.
    Running it through the name/stem matching below would be wrong in both
    directions: it could match a memory whose frontmatter ``name`` merely
    resembles the filename, and it would ignore the one unambiguous thing
    the author actually supplied.

    A ``wiki`` target is an identifier, and both naming styles in the corpus
    are honoured: it may be written as the frontmatter ``name`` or as the
    file stem, and in a real corpus those routinely disagree
    (``feedback_timeout_defaults.md`` carries
    ``name: timeout-defaults-not-retries``), so a link may legitimately
    point at either.

    Wiki matching is modulo separator -- ``-`` and ``_`` are the same
    identifier in different clothes, and corpora mix them freely
    (``[[feedback-timeout-defaults]]`` referring to
    ``feedback_timeout_defaults.md``). That is still an EXACT match on a
    normalised string, not fuzzy matching: prefixes and substrings are
    deliberately NOT matched, because a link whose text resembles a memory
    without naming it is authoring drift, and guessing would invent edges
    the author never wrote. Unmatched links stay ``is_resolved = FALSE``
    with a NULL target rather than being dropped.
    """
    conn.execute(
        """
        UPDATE bridge_memory_link SET
            target_memory_id = (
                SELECT t.memory_id FROM dim_memory t
                JOIN dim_memory src ON src.memory_key = bridge_memory_link.memory_key
                WHERE t.scope = bridge_memory_link.scope
                  AND t.owner_key = bridge_memory_link.owner_key
                  -- owner_key alone is NOT unique for agent memory (that is
                  -- why memory_id carries owner_root), so without this a
                  -- `reviewer` agent present in two repositories could have
                  -- one repo's index point at the other repo's memory.
                  AND t.owner_root IS NOT DISTINCT FROM src.owner_root
                  AND t.is_current
                  AND t.file_name = bridge_memory_link.target_name
                LIMIT 1
            )
        WHERE target_memory_id IS NULL AND link_syntax = 'markdown'
        """
    )
    norm = "lower(replace({}, '_', '-'))"
    conn.execute(
        f"""
        UPDATE bridge_memory_link SET
            target_memory_id = (
                SELECT t.memory_id FROM dim_memory t
                JOIN dim_memory src ON src.memory_key = bridge_memory_link.memory_key
                WHERE t.scope = bridge_memory_link.scope
                  AND t.owner_key = bridge_memory_link.owner_key
                  AND t.owner_root IS NOT DISTINCT FROM src.owner_root
                  AND t.is_current
                  AND (
                        {norm.format('t.memory_name')}
                        = {norm.format('bridge_memory_link.target_name')}
                     OR {norm.format("regexp_replace(t.file_name, '\\.md$', '')")}
                        = {norm.format('bridge_memory_link.target_name')}
                  )
                LIMIT 1
            )
        -- IS DISTINCT FROM, not <>: pre-0.19 rows carry link_syntax
        -- NULL, and `NULL <> 'markdown'` is NULL, so the wiki branch
        -- skipped every legacy row forever -- a dangling link whose
        -- target was written later could never resolve.
        WHERE target_memory_id IS NULL AND link_syntax IS DISTINCT FROM 'markdown'
        """
    )
    conn.execute(
        "UPDATE bridge_memory_link SET is_resolved = (target_memory_id IS NOT NULL)"
    )
