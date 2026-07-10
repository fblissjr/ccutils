"""Detect subagent sessions and enrich dim_session accordingly.

Subagent JSONL files live at:
    .../projects/<project>/<parent-session-uuid>/subagents/agent-<id>.jsonl

with an optional sidecar .meta.json carrying agentType + description.
This populator examines the source_path of sessions currently in
staging and, for each subagent layout it recognises, UPDATEs dim_session
with is_agent / agent_id / parent_session_key + reads the sidecar to
populate agent_type / agent_description. Then propagates_depth_level
across the parent_session_key chain.

Run AFTER _upsert_minimal_dimensions (which inserted the dim_session
rows in the first place).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ccutils.etl.lineage import EtlRun


_SUBAGENT_PATH_RE = re.compile(
    r"/(?P<parent>[^/]+)/subagents/agent-(?P<agent_id>[^/]+)\.jsonl$"
)


def populate_subagent_dim_session(conn, *, run: EtlRun) -> None:
    """Mark subagent sessions in dim_session by source-path inspection."""
    source_paths = [
        r[0] for r in conn.execute(
            """
            SELECT DISTINCT source_path FROM stg_log_entries
            WHERE source_path IS NOT NULL
              AND session_id IS NOT NULL
            """
        ).fetchall()
    ]

    updates = []
    for source_path in source_paths:
        match = _SUBAGENT_PATH_RE.search(source_path)
        if not match:
            continue
        agent_id = match.group("agent_id")
        parent_session_id = match.group("parent")

        # Sidecar lookup: agent-<id>.meta.json next to the .jsonl
        agent_type = None
        agent_description = None
        meta_path = Path(source_path).with_suffix(".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
            agent_type = meta.get("agentType") or None
            agent_description = meta.get("description") or None

        updates.append(
            (
                agent_id,
                conn.execute(
                    "SELECT md5(?)", [parent_session_id]
                ).fetchone()[0],
                agent_type,
                agent_description,
                source_path,
            )
        )

    if not updates:
        return

    # Match by source_path against dim_session via stg_log_entries lookup.
    conn.execute("DROP TABLE IF EXISTS _inbound_subagents")
    conn.execute(
        """
        CREATE TEMP TABLE _inbound_subagents (
            agent_id VARCHAR,
            parent_session_key VARCHAR,
            agent_type VARCHAR,
            agent_description VARCHAR,
            source_path VARCHAR
        )
        """
    )
    conn.executemany(
        "INSERT INTO _inbound_subagents VALUES (?, ?, ?, ?, ?)",
        updates,
    )
    conn.execute(
        """
        UPDATE dim_session ds
        SET is_agent = TRUE,
            agent_id = ins.agent_id,
            parent_session_key = ins.parent_session_key,
            agent_type = ins.agent_type,
            agent_description = ins.agent_description
        FROM _inbound_subagents ins
        JOIN stg_log_entries sle ON sle.source_path = ins.source_path
        WHERE ds.session_key = md5(sle.session_id)
        """
    )
    conn.execute("DROP TABLE IF EXISTS _inbound_subagents")
    _propagate_depth_level(conn)
    _ = run  # signature symmetry


def _propagate_depth_level(conn) -> None:
    """Walk the parent_session_key chain to set depth_level.

    Uses a single recursive CTE update in DuckDB to traverse the subagent
    hierarchy natively.
    """
    conn.execute("UPDATE dim_session SET depth_level = NULL")
    conn.execute(
        """
        WITH RECURSIVE depth_calc AS (
            SELECT session_key, 0 AS computed_depth
            FROM dim_session
            WHERE is_agent = FALSE
               OR parent_session_key IS NULL
               OR parent_session_key NOT IN (SELECT session_key FROM dim_session)

            UNION ALL

            SELECT child.session_key, parent.computed_depth + 1
            FROM dim_session child
            JOIN depth_calc parent ON child.parent_session_key = parent.session_key
            WHERE child.is_agent = TRUE
        )
        UPDATE dim_session
        SET depth_level = dc.computed_depth
        FROM depth_calc dc
        WHERE dim_session.session_key = dc.session_key
        """
    )
    conn.execute(
        "UPDATE dim_session SET depth_level = 0 WHERE depth_level IS NULL"
    )
