"""Shared utilities for v0.15 ETL populators.

Functions here are used by multiple populators (currently
`dim_session_heuristics` and `facets.populator`) and don't fit any
single populator's module. The bar for adding something here is "needed
by at least two populators."
"""

from __future__ import annotations

import json
import re


# Single source for the subagent-file layout rule. Every consumer builds
# from these (drifted copies of path rules are how the "subagents"
# mis-attribution shipped): SUBAGENT_PATH_RE for Python matching (match on
# Path(...).as_posix() so separators are uniform), the *_sql builders for
# DuckDB. Layout: .../<parent-uuid>/subagents/agent-<id>.jsonl
_SUBAGENT_TAIL = r"/subagents/agent-[^/]+\.jsonl$"
_SUBAGENT_SESSION_ID_TAIL = r"/subagents/(agent-[^/]+)\.jsonl$"

SUBAGENT_PATH_RE = re.compile(
    r"/(?P<parent>[^/]+)/subagents/agent-(?P<agent_id>[^/]+)\.jsonl$"
)


def subagent_match_sql(col: str) -> str:
    """SQL predicate: does ``col`` look like a subagent transcript path?"""
    return f"regexp_matches({col}, '{_SUBAGENT_TAIL}')"


def subagent_session_id_sql(col: str) -> str:
    """SQL expression: the file-identity session id ('agent-<id>') for a
    subagent transcript path."""
    return f"regexp_extract({col}, '{_SUBAGENT_SESSION_ID_TAIL}', 1)"


def project_dir_sql(col: str) -> str:
    """SQL expression: the project directory for a session source_path.

    For a top-level session (``.../projects/<project>/<uuid>.jsonl``) that is
    the parent directory. Subagent files live at
    ``.../projects/<project>/<parent-uuid>/subagents/agent-<id>.jsonl`` --
    stripping only the filename would attribute every subagent to a synthetic
    "subagents" project.

    Two steps, and the order matters. Drop the filename, then drop
    everything from the first ``/<seg>/subagents`` onward. Cutting at the
    FIRST such layer collapses nested delegation in one pass, and letting
    the tail be any depth covers grouping directories below ``subagents``:
    workflow-tool agents sit at
    ``.../<parent-uuid>/subagents/workflows/<wf-id>/agent-<id>.jsonl``, and
    the previous pattern -- which stripped only ``/<seg>/subagents`` layers
    -- attributed those to a project named after the workflow id.

    Every populator that derives ``project_key`` MUST use this expression;
    drifted copies were how the "subagents" mis-attribution shipped.
    """
    without_file = f"regexp_replace({col}, '/[^/]+$', '')"
    return f"regexp_replace({without_file}, '/[^/]+/subagents(/.*)?$', '')"


def project_key_from_dir_sql(col: str) -> str:
    """md5 over an ALREADY-computed project dir. Same key material as
    :func:`project_key_sql`; use when the dir was computed once in a
    subquery to avoid re-running the regexp."""
    return f"md5({col})"


def project_key_sql(col: str) -> str:
    """SQL expression: md5 surrogate key of :func:`project_dir_sql`."""
    return project_key_from_dir_sql(project_dir_sql(col))


def fetch_scalar(conn, sql: str, params=None):
    """Run a query expected to return at least one row; return row[0].

    Replaces the bare ``conn.execute(...).fetchone()[0]`` pattern:
    ``fetchone()`` is Optional, so subscripting it is unsound when a
    query can return zero rows. Raises RuntimeError (with the SQL) on
    zero rows instead of an opaque ``NoneType`` subscript error. A NULL
    value inside an existing row is returned as None, not raised.
    """
    cursor = conn.execute(sql, params) if params is not None else conn.execute(sql)
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"Query returned no rows: {sql!r}")
    return row[0]


def insert_missing_dim_dates(conn, table: str, *timestamp_cols: str) -> None:
    """Insert dim_date rows for calendar dates a table references.

    Derives the set of dates from ``timestamp_cols`` on ``table`` (each
    ``TRY_CAST`` to TIMESTAMP, so VARCHAR and TIMESTAMP columns both work,
    and unparseable values drop out), then inserts any not already in
    dim_date. date_key matches the YYYYMMDD integers lineage_upsert derives
    on every fact; day_of_week mirrors Python's weekday() (Monday=0).

    Typed rather than raw-SQL so the "one DATE column named day" contract
    can't be broken by a caller and no runtime value can be interpolated
    into the INSERT. Callers: _upsert_minimal_dimensions (staging dates +
    dim_session reconcile) and import_history (dim_prompt dates).
    """
    if not timestamp_cols:
        return
    union = " UNION ALL ".join(
        f"SELECT CAST(TRY_CAST({col} AS TIMESTAMP) AS DATE) AS day "
        f"FROM {table} WHERE TRY_CAST({col} AS TIMESTAMP) IS NOT NULL"
        for col in timestamp_cols
    )
    conn.execute(
        f"""
        INSERT INTO dim_date
        SELECT
            CAST(strftime(d.day, '%Y%m%d') AS INTEGER) AS date_key,
            d.day AS full_date,
            EXTRACT(year FROM d.day) AS year,
            EXTRACT(month FROM d.day) AS month,
            EXTRACT(day FROM d.day) AS day,
            EXTRACT(isodow FROM d.day) - 1 AS day_of_week,
            dayname(d.day) AS day_name,
            monthname(d.day) AS month_name,
            EXTRACT(quarter FROM d.day) AS quarter,
            EXTRACT(isodow FROM d.day) >= 6 AS is_weekend,
            EXTRACT(week FROM d.day) AS week_of_year
        FROM (SELECT DISTINCT day FROM ({union})) d
        WHERE d.day IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM dim_date dd
              WHERE dd.date_key = CAST(strftime(d.day, '%Y%m%d') AS INTEGER)
          )
        """
    )


def extract_text_from_content_json(
    content_json_raw: str | None,
    *,
    include_thinking: bool = True,
) -> str:
    """Pull plain text out of a `message.content` JSON payload.

    Claude Code emits user content as either a bare JSON string or a
    list of content blocks; assistant content is always a list of
    blocks. This helper concatenates every text-bearing block
    (`type='text'` and `type='thinking'`); `tool_result` and other
    block types are skipped because they're tool output, not user
    intent or assistant conclusion.

    When `include_thinking=False`, `type='thinking'` blocks are skipped
    too -- this is the seam that lets `--no-thinking` propagate beyond
    `fact_messages` (whose SQL projection already excludes thinking)
    into derived columns like `dim_session.last_assistant_message` and
    the Tier 2 facet extractor's `SessionInputs`. `type='redacted_thinking'`
    blocks are never emitted by this helper regardless of the flag --
    redacted content is the API's signal that the payload is sensitive,
    so we drop it unconditionally. Note the asymmetry with
    `fact_messages.has_thinking`, which IS set TRUE for redacted blocks.

    Returns an empty string when the input is None, unparseable, or
    contains no text blocks. The empty-string fallback (vs. None) lets
    downstream classifiers and prompt builders treat "no extractable
    text" and "literally empty text" the same way.
    """
    if not content_json_raw:
        return ""
    try:
        content = json.loads(content_json_raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(block.get("text", ""))
            elif block_type == "thinking" and include_thinking:
                parts.append(block.get("thinking", ""))
        return " ".join(p for p in parts if p)
    return ""


# One tool -> kind-of-work mapping for the whole warehouse.
#
# This is the single source of truth for `dim_tool.tool_category`. It is a
# DEFINITIONAL grouping (what the tool does to the world), not a heuristic
# classifier: Read reads, Edit mutates, Bash executes. Nothing here is a
# threshold or an inferred label -- those belong in the analysis layer where
# cutoffs can be derived from the corpus distribution.
#
# `{col}` is the tool-name column to categorize, so the same expression
# serves both the INSERT of new tools and the backfill UPDATE of old rows.
# MCP tools match on the `mcp__` naming convention rather than being
# enumerated, so a new MCP server needs no code change here.
#
# Categories: read, search, mutate, execute, web, delegate, plan, interact,
# mcp, other.
TOOL_CATEGORY_SQL = """CASE
    WHEN LOWER({col}) LIKE 'mcp\\_\\_%' ESCAPE '\\' THEN 'mcp'
    WHEN LOWER({col}) IN ('read', 'notebookread') THEN 'read'
    WHEN LOWER({col}) IN ('grep', 'glob') THEN 'search'
    WHEN LOWER({col}) IN ('edit', 'write', 'notebookedit', 'multiedit')
        THEN 'mutate'
    WHEN LOWER({col}) IN ('bash', 'bashoutput', 'killshell') THEN 'execute'
    WHEN LOWER({col}) IN ('webfetch', 'websearch') THEN 'web'
    WHEN LOWER({col}) IN ('agent', 'task', 'skill') THEN 'delegate'
    WHEN LOWER({col}) IN (
        'exitplanmode', 'enterplanmode', 'todowrite', 'taskcreate',
        'taskupdate', 'taskget', 'tasklist', 'taskoutput', 'taskstop'
    ) THEN 'plan'
    WHEN LOWER({col}) IN (
        'askuserquestion', 'senduserfile', 'artifact', 'structuredoutput'
    ) THEN 'interact'
    ELSE 'other'
END"""


# Context-window suffix stripped from a model id: 'claude-opus-5[1m]' is the
# same model as 'claude-opus-5' with a larger window. Without this they become
# separate dim_model rows and every per-model aggregate splits silently.
# `model_name` stays byte-faithful to the transcript; model_base is the
# grouping key. Mirrors get_model_base() in schemas/star/utils.py.
MODEL_BASE_SQL = """NULLIF(TRIM(SPLIT_PART({col}, '[', 1)), '')"""

# Family parsed STRUCTURALLY from the claude-<family>-<version...> convention,
# not matched against a list of known families. An enumerated list goes stale
# the moment a new model line ships -- `fable` was missing from the previous
# LIKE-chain, which bucketed the corpus's third-most-used model (more output
# tokens than Opus 5) as 'unknown' in every GROUP BY model_family.
# Mirrors get_model_family() in schemas/star/utils.py; the two are asserted
# to agree in tests/test_dim_model_v15.py.
MODEL_FAMILY_SQL = """CASE
    WHEN {col} IS NULL THEN 'unknown'
    WHEN LOWER(SPLIT_PART(TRIM(SPLIT_PART({col}, '[', 1)), '-', 1)) <> 'claude'
        THEN 'unknown'
    WHEN NULLIF(SPLIT_PART(TRIM(SPLIT_PART({col}, '[', 1)), '-', 2), '') IS NULL
        THEN 'unknown'
    ELSE LOWER(SPLIT_PART(TRIM(SPLIT_PART({col}, '[', 1)), '-', 2))
END"""
