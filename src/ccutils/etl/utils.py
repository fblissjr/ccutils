"""Shared utilities for v0.15 ETL populators.

Functions here are used by multiple populators (currently
`dim_session_heuristics` and `facets.populator`) and don't fit any
single populator's module. The bar for adding something here is "needed
by at least two populators."
"""

from __future__ import annotations

import json


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
