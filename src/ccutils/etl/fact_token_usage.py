"""Populate fact_token_usage from stg_log_entries (Phase C5a).

One row per assistant API response that carried `usage` data. R11
correction: cache_creation split into pricing tiers (_5m / _1h) and a
derived total_uncached_equivalent_tokens (= cache_read + cache_creation
+ input_tokens), so downstream cost views can apply 1.25x and 2x
multipliers correctly.
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert
from ccutils.etl.utils import project_key_sql


_PAYLOAD_COLS = [
    "project_key", "model_key",
    "timestamp",
    "input_tokens", "output_tokens",
    "cache_creation_5m_tokens", "cache_creation_1h_tokens",
    "cache_creation_total_tokens", "cache_read_tokens",
    "total_uncached_equivalent_tokens",
    "service_tier", "speed", "inference_geo",
    "server_tool_use_web_search_requests",
    "server_tool_use_web_fetch_requests",
]
_HASH_COLS = [
    "timestamp",
    "input_tokens", "output_tokens",
    "cache_creation_5m_tokens", "cache_creation_1h_tokens",
    "cache_creation_total_tokens", "cache_read_tokens",
    "total_uncached_equivalent_tokens",
    "service_tier", "speed", "inference_geo",
    "server_tool_use_web_search_requests",
    "server_tool_use_web_fetch_requests",
]


_PROJECT_SQL = """
SELECT
    sle.entry_id,
    sle.session_id,
    sle.source_path,
    TRY_CAST(sle.timestamp AS TIMESTAMP) AS timestamp,

    -- Anthropic usage shape
    json_extract(sle.message_json, '$.usage.input_tokens')::INTEGER
        AS input_tokens,
    json_extract(sle.message_json, '$.usage.output_tokens')::INTEGER
        AS output_tokens,
    json_extract(sle.message_json, '$.usage.cache_creation.ephemeral_5m_input_tokens')::INTEGER
        AS cache_creation_5m_tokens,
    json_extract(sle.message_json, '$.usage.cache_creation.ephemeral_1h_input_tokens')::INTEGER
        AS cache_creation_1h_tokens,
    -- Additive total of the two TTL tiers; matches the legacy
    -- cache_creation_input_tokens field for back-compat queries.
    COALESCE(json_extract(sle.message_json, '$.usage.cache_creation.ephemeral_5m_input_tokens')::INTEGER, 0)
        + COALESCE(json_extract(sle.message_json, '$.usage.cache_creation.ephemeral_1h_input_tokens')::INTEGER, 0)
        AS cache_creation_total_tokens,
    json_extract(sle.message_json, '$.usage.cache_read_input_tokens')::INTEGER
        AS cache_read_tokens,
    -- R11: total uncached equivalent = read + creation_total + input
    COALESCE(json_extract(sle.message_json, '$.usage.input_tokens')::INTEGER, 0)
        + COALESCE(json_extract(sle.message_json, '$.usage.cache_creation.ephemeral_5m_input_tokens')::INTEGER, 0)
        + COALESCE(json_extract(sle.message_json, '$.usage.cache_creation.ephemeral_1h_input_tokens')::INTEGER, 0)
        + COALESCE(json_extract(sle.message_json, '$.usage.cache_read_input_tokens')::INTEGER, 0)
        AS total_uncached_equivalent_tokens,

    json_extract_string(sle.message_json, '$.usage.service_tier') AS service_tier,
    json_extract_string(sle.message_json, '$.usage.speed') AS speed,
    json_extract_string(sle.message_json, '$.usage.inference_geo') AS inference_geo,
    json_extract(sle.message_json, '$.usage.server_tool_use.web_search_requests')::INTEGER
        AS server_tool_use_web_search_requests,
    json_extract(sle.message_json, '$.usage.server_tool_use.web_fetch_requests')::INTEGER
        AS server_tool_use_web_fetch_requests
FROM stg_log_entries sle
WHERE sle.type = 'assistant'
  AND json_extract(sle.message_json, '$.usage') IS NOT NULL
"""


def populate_fact_token_usage(conn, *, run: EtlRun) -> None:
    conn.execute("DROP TABLE IF EXISTS _inbound_token_usage")
    conn.execute(f"CREATE TEMP TABLE _inbound_token_usage AS {_PROJECT_SQL}")

    # Derive project_key + model_key (model_key from the assistant message).
    conn.execute("ALTER TABLE _inbound_token_usage ADD COLUMN project_key VARCHAR")
    conn.execute("ALTER TABLE _inbound_token_usage ADD COLUMN model_key VARCHAR")
    conn.execute(
        "UPDATE _inbound_token_usage "
        f"SET project_key = {project_key_sql('source_path')}"
    )
    conn.execute(
        """
        UPDATE _inbound_token_usage im
        SET model_key = md5(json_extract_string(sle.message_json, '$.model'))
        FROM stg_log_entries sle
        WHERE sle.entry_id = im.entry_id
          AND json_extract_string(sle.message_json, '$.model') IS NOT NULL
        """
    )

    lineage_upsert(
        conn,
        run=run,
        table="fact_token_usage",
        inbound_table="_inbound_token_usage",
        natural_key="entry_id",
        payload_cols=_PAYLOAD_COLS,
        hash_cols=_HASH_COLS,
    )
