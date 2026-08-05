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

from ccutils.etl.utils import insert_missing_dim_dates
from ccutils.parsers.memory import (
    MemoryFile,
    iter_agent_memories,
    iter_project_memories,
    scanned_owners,
)

__all__ = ["import_memories"]


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

        for ordinal, target in enumerate(mem.links):
            links.append(
                (
                    hashlib.md5(
                        f"{memory_key}\x00{ordinal}".encode("utf-8")
                    ).hexdigest(),
                    memory_key,
                    memory_id,
                    mem.scope,
                    mem.owner_key,
                    target,
                    ordinal,
                )
            )

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

    _insert_versions(conn, inbound)
    _insert_links(conn, links)
    _resolve_project_and_session_keys(conn)
    _resolve_link_targets(conn)
    insert_missing_dim_dates(conn, "dim_memory", "modified_at")
    return len(inbound)


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


def _insert_versions(conn, inbound: list[tuple]) -> None:
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
            memory_key, memory_id, project_key, session_key, scope, owner_key,
            owner_root, agent_scope, source_path, file_name, memory_name, description,
            memory_type, node_type, origin_session_id, is_index,
            has_frontmatter, body_text, content_hash, body_chars, body_lines,
            link_count, modified_at, file_mtime, version_num, valid_from,
            valid_to, is_current, date_key, time_key
        )
        SELECT
            memory_key, memory_id, NULL, NULL, scope, owner_key,
            owner_root, agent_scope, source_path, file_name, memory_name, description,
            memory_type, node_type, origin_session_id, is_index,
            has_frontmatter, body_text, content_hash, body_chars, body_lines,
            link_count, modified_at, file_mtime, version_num, valid_from,
            NULL, TRUE, date_key, time_key
        FROM _inbound_memory
        """
    )
    conn.execute("DROP TABLE IF EXISTS _inbound_memory")


def _insert_links(conn, links: list[tuple]) -> None:
    if not links:
        return
    conn.executemany(
        """
        INSERT INTO bridge_memory_link (
            memory_link_key, memory_key, memory_id, project_key, scope,
            owner_key, target_name, target_memory_id, is_resolved, ordinal
        ) VALUES (?, ?, ?, NULL, ?, ?, ?, NULL, FALSE, ?)
        """,
        links,
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
    """Match each ``[[name]]`` to a sibling memory in the same scope+owner.

    Both naming styles in the corpus are honoured: the target may be written
    as the frontmatter ``name`` or as the file stem, and in a real corpus
    those routinely disagree (``feedback_signal_honesty.md`` carries
    ``name: signal-honesty-over-green-boards``), so a link may legitimately
    point at either.

    Matching is modulo separator -- ``-`` and ``_`` are the same identifier
    in different clothes, and corpora mix them freely (``[[feedback-signal-
    honesty]]`` referring to ``feedback_signal_honesty.md``). That is still
    an EXACT match on a normalised string, not fuzzy matching: prefixes and
    substrings are deliberately NOT matched, because a link whose text
    resembles a memory without naming it is authoring drift, and guessing
    would invent edges the author never wrote. Unmatched links stay
    ``is_resolved = FALSE`` with a NULL target rather than being dropped.
    """
    norm = "lower(replace({}, '_', '-'))"
    conn.execute(
        f"""
        UPDATE bridge_memory_link SET
            target_memory_id = (
                SELECT t.memory_id FROM dim_memory t
                WHERE t.scope = bridge_memory_link.scope
                  AND t.owner_key = bridge_memory_link.owner_key
                  AND t.is_current
                  AND (
                        {norm.format('t.memory_name')}
                        = {norm.format('bridge_memory_link.target_name')}
                     OR {norm.format("regexp_replace(t.file_name, '\\.md$', '')")}
                        = {norm.format('bridge_memory_link.target_name')}
                  )
                LIMIT 1
            )
        WHERE target_memory_id IS NULL
        """
    )
    conn.execute(
        "UPDATE bridge_memory_link SET is_resolved = (target_memory_id IS NOT NULL)"
    )
