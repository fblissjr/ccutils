"""Typed reader: Parquet (Tier 1) → Pydantic typed entries.

Inverse of `parquet_writer`. Two modes:

- `iter_log_entries_typed(parquet_path)` -- yields Pydantic SessionLogEntry
  objects by reconstructing from the `raw_json` column. Best for Python-side
  analytical scripts that want full typed access to historical data.

- `iter_log_entries_raw(parquet_path)` -- yields raw row dicts for callers
  that need only specific columns. Cheaper than constructing Pydantic models.

DuckDB consumers don't need this module: they read Parquet directly via
`SELECT ... FROM read_parquet('parquet_lake/.../log_entries.parquet')`.
This module is for Python integration tests + analysis scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from ccutils.parsers.models import SessionLogEntry, parse_log_entry


def iter_log_entries_raw(parquet_path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield log_entries.parquet rows as dicts in source order.

    Cheap iteration -- no Pydantic construction. Useful when downstream
    only needs lineage columns or raw JSON payloads.
    """
    table = pq.read_table(parquet_path)
    for row in table.to_pylist():
        yield row


def iter_log_entries_typed(parquet_path: str | Path) -> Iterator[SessionLogEntry]:
    """Yield log_entries.parquet rows as typed Pydantic entries in source order.

    Re-parses the `raw_json` column through parse_log_entry, so the result
    is byte-for-byte equivalent to what would have been yielded by
    iter_typed_entries() on the original JSONL.
    """
    for row in iter_log_entries_raw(parquet_path):
        raw_json = row.get("raw_json")
        if not raw_json:
            continue
        try:
            raw = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        yield parse_log_entry(raw)


def read_session_meta(parquet_path: str | Path) -> dict[str, Any]:
    """Return the single session_meta row as a dict."""
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"session_meta parquet at {parquet_path} is empty")
    return rows[0]
