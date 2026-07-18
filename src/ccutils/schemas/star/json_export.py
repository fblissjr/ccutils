"""Export star schema DuckDB to JSON directory structure."""

import json
from datetime import datetime
from pathlib import Path

# Exported tables are discovered from the live database at export time
# (see _star_tables) rather than a hardcoded list -- the old FACT_TABLES
# literal drifted badly (it exported [] for tables that no longer existed
# and silently omitted most populated v0.15 facts).


def _star_tables(conn):
    """Return (dimension_tables, fact_tables) present in the connection.

    Dimensions are dim_*; facts are fact_* plus bridge_*. stg_* (per-run
    scratch, always cleared) and meta_* (DDL bookkeeping) are not star
    data and are excluded.
    """
    tables = sorted(
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_type = 'BASE TABLE'"
        ).fetchall()
    )
    dims = [t for t in tables if t.startswith("dim_")]
    facts = [t for t in tables if t.startswith(("fact_", "bridge_"))]
    return dims, facts

# FK-by-convention key columns -> the dimension they join to. Relationships
# are DERIVED from the live database at export time (same discipline as
# _star_tables) -- the old hardcoded RELATIONSHIPS literal drifted just like
# the old FACT_TABLES list did (it referenced the removed fact_tool_calls and
# omitted most populated v0.15 facts). The uniform lineage columns
# (etl_run_id, *_version_key) are deliberately excluded: they appear on every
# fact and are documented in docs/STAR_SCHEMA.md, so listing them per-table
# would triple the list without adding signal.
_KEY_TARGETS = {
    "session_key": ("dim_session", "session_key"),
    "parent_session_key": ("dim_session", "session_key"),
    "agent_session_key": ("dim_session", "session_key"),
    "first_session_key": ("dim_session", "session_key"),
    "last_session_key": ("dim_session", "session_key"),
    "project_key": ("dim_project", "project_key"),
    "tool_key": ("dim_tool", "tool_key"),
    "prev_tool_key": ("dim_tool", "tool_key"),
    "next_tool_key": ("dim_tool", "tool_key"),
    "model_key": ("dim_model", "model_key"),
    "file_key": ("dim_file", "file_key"),
    "date_key": ("dim_date", "date_key"),
    "time_key": ("dim_time", "time_key"),
    "chain_key": ("dim_session_chain", "chain_key"),
    "facet_type_key": ("dim_facet_type", "facet_type_key"),
    "prompt_key": ("dim_prompt", "prompt_key"),
}


def star_relationships(conn):
    """Derive star-schema join relationships from the live database.

    One entry per (table, FK-convention column) whose target dimension
    exists. A dimension's own primary key is skipped (dim_session.session_key
    is not a relationship to itself); self-referencing hierarchies via a
    differently-named column (dim_session.parent_session_key) are kept.
    """
    dims, facts = _star_tables(conn)
    dim_set = set(dims)
    relationships = []
    for table in dims + facts:
        columns = [
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table],
            ).fetchall()
        ]
        for column in columns:
            target = _KEY_TARGETS.get(column)
            if target is None:
                continue
            to_table, to_column = target
            if to_table not in dim_set:
                continue
            if table == to_table and column == to_column:
                continue
            relationships.append(
                {
                    "from_table": table,
                    "from_column": column,
                    "to_table": to_table,
                    "to_column": to_column,
                }
            )
    return relationships


def export_star_schema_to_json(conn, output_dir):
    """Export star schema DuckDB tables to JSON directory structure.

    Creates:
        output_dir/
            meta.json           - Schema metadata and relationships
            dimensions/         - One JSON file per dimension table
            facts/              - One JSON file per fact table

    Args:
        conn: DuckDB connection with star schema data
        output_dir: Directory to write JSON files to
    """
    output_dir = Path(output_dir)
    dimensions_dir = output_dir / "dimensions"
    facts_dir = output_dir / "facts"

    dimensions_dir.mkdir(parents=True, exist_ok=True)
    facts_dir.mkdir(parents=True, exist_ok=True)

    table_manifest = {"dimensions": [], "facts": []}
    dimension_tables, fact_tables = _star_tables(conn)

    # Export dimension tables
    for table_name in dimension_tables:
        rows = _export_table(conn, table_name, dimensions_dir)
        table_manifest["dimensions"].append(
            {
                "name": table_name,
                "file": f"dimensions/{table_name}.json",
                "row_count": rows,
            }
        )

    # Export fact tables
    for table_name in fact_tables:
        rows = _export_table(conn, table_name, facts_dir)
        table_manifest["facts"].append(
            {
                "name": table_name,
                "file": f"facts/{table_name}.json",
                "row_count": rows,
            }
        )

    # Write meta.json
    meta = {
        "version": "2.0",
        "schema_type": "star",
        "exported_at": datetime.now().astimezone().isoformat(),
        "tables": table_manifest,
        "relationships": star_relationships(conn),
    }

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def _export_table(conn, table_name, output_dir):
    """Export a single table to JSON.

    Args:
        conn: DuckDB connection
        table_name: Name of the table to export
        output_dir: Directory to write to

    Returns:
        Number of rows exported
    """
    try:
        # Get column names
        columns_result = conn.execute(f"DESCRIBE {table_name}").fetchall()
        column_names = [row[0] for row in columns_result]

        # Get all rows
        rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()

        # Convert to list of dicts
        data = []
        for row in rows:
            record = {}
            for i, col_name in enumerate(column_names):
                value = row[i]
                # Handle special types
                if hasattr(value, "isoformat"):
                    value = value.isoformat()
                record[col_name] = value
            data.append(record)

        # Write JSON file
        output_path = output_dir / f"{table_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return len(data)

    except Exception:
        # Table might not exist or be empty - write empty array
        output_path = output_dir / f"{table_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        return 0
