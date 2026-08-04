"""Populate dim_file + fact_file_operations from the v0.15 facts.

dim_file is populated from distinct file_paths observed in tool_use
inputs (Read/Write/Edit/MultiEdit/NotebookEdit) and tool_result payloads
(Read returns numLines/file_path even when input is a glob pattern).
Stays minimal -- matches dim_tool/dim_model: no lineage block, simple
catalog.

fact_file_operations is one row per file-touching tool call, joined from
fact_tool_uses to fact_tool_results on tool_use_id. operation_type is
inferred from the tool name. Carries the full v0.15 lineage block.

Both populators expect fact_tool_uses and fact_tool_results to be
populated already -- run them after populate_fact_tool_results in the
orchestrator.
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert


# Tools whose input.file_path (or similar) names a single file we want
# in dim_file. Mapped to the operation_type column on fact_file_operations.
_FILE_TOOL_OPS = {
    "Read": "read",
    "Write": "write",
    "Edit": "edit",
    "MultiEdit": "edit",
    "NotebookEdit": "edit",
    "Glob": "list",
    "Grep": "search",
}

# File extension -> language label. Subset of the v0.14 mapping, kept
# narrow on purpose; unknown extensions fall through to NULL.
_LANGUAGE_BY_EXT = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "jsx": "javascript",
    "rs": "rust",
    "go": "go",
    "rb": "ruby",
    "java": "java",
    "kt": "kotlin",
    "swift": "swift",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "cc": "cpp",
    "hpp": "cpp",
    "cs": "csharp",
    "php": "php",
    "sh": "shell",
    "bash": "shell",
    "zsh": "shell",
    "sql": "sql",
    "html": "html",
    "css": "css",
    "scss": "scss",
    "md": "markdown",
    "mdx": "markdown",
    "rst": "rst",
    "txt": "text",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "xml": "xml",
    "ipynb": "jupyter",
}


def populate_dim_file(conn, *, run: EtlRun) -> None:
    """Insert any distinct file_path observed in v0.15 tool facts that isn't
    already in dim_file."""
    # Build the language CASE expression from the dict so the schema and
    # the populator share a single source for the mapping.
    case_parts = [
        f"WHEN file_extension = '{ext}' THEN '{lang}'"
        for ext, lang in _LANGUAGE_BY_EXT.items()
    ]
    language_case = "CASE " + " ".join(case_parts) + " ELSE NULL END"

    conn.execute(
        f"""
        INSERT INTO dim_file (
            file_key, file_path, file_name, file_extension,
            directory_path, language
        )
        WITH observed AS (
            -- File paths from tool_use inputs (Read/Write/Edit/MultiEdit/NotebookEdit)
            SELECT DISTINCT
                json_extract_string(ftu.input_json, '$.file_path') AS file_path
            FROM fact_tool_uses ftu
            WHERE ftu.is_deleted = FALSE
              AND json_extract_string(ftu.input_json, '$.file_path') IS NOT NULL
            UNION
            -- File paths from tool_result payloads (Read records file_path
            -- even when the input was something else, e.g. a notebook read).
            SELECT DISTINCT ftr.read_file_path AS file_path
            FROM fact_tool_results ftr
            WHERE ftr.is_deleted = FALSE
              AND ftr.read_file_path IS NOT NULL
        ),
        parsed AS (
            SELECT
                file_path,
                regexp_extract(file_path, '([^/]+)$', 1) AS file_name,
                regexp_replace(file_path, '/[^/]+$', '') AS directory_path
            FROM observed
            WHERE file_path IS NOT NULL AND file_path <> ''
        ),
        typed AS (
            SELECT
                file_path,
                file_name,
                directory_path,
                CASE
                    WHEN file_name LIKE '%.%'
                        THEN regexp_extract(file_name, '\\.([^.]+)$', 1)
                    ELSE NULL
                END AS file_extension
            FROM parsed
        )
        SELECT
            md5(file_path) AS file_key,
            file_path,
            file_name,
            file_extension,
            directory_path,
            {language_case} AS language
        FROM typed
        WHERE NOT EXISTS (
            SELECT 1 FROM dim_file df WHERE df.file_key = md5(typed.file_path)
        )
        """
    )
    # `run` accepted for signature symmetry with the fact populators;
    # dim_file matches the dim_tool / dim_model pattern (no lineage block).


_FILE_OP_PAYLOAD_COLS = [
    "timestamp", "tool_key", "file_key", "operation_type",
    "file_path", "file_size_chars",
]
_FILE_OP_HASH_COLS = [
    "timestamp", "tool_key", "file_key", "operation_type",
    "file_path", "file_size_chars",
]


def populate_fact_file_operations(conn, *, run: EtlRun) -> None:
    """Derive one fact_file_operations row per file-touching tool_use."""
    # Build the tool->operation CASE from the dict.
    op_case = "CASE " + " ".join(
        f"WHEN ftu.tool_name = '{tool}' THEN '{op}'"
        for tool, op in _FILE_TOOL_OPS.items()
    ) + " ELSE NULL END"
    tool_list = ", ".join(f"'{t}'" for t in _FILE_TOOL_OPS)

    conn.execute("DROP TABLE IF EXISTS _inbound_file_ops")
    conn.execute(
        f"""
        CREATE TEMP TABLE _inbound_file_ops AS
        SELECT
            ftu.tool_use_id,
            ftu.session_id,
            ftu.timestamp,
            ftu.tool_key,
            COALESCE(
                json_extract_string(ftu.input_json, '$.file_path'),
                ftr.read_file_path
            ) AS file_path,
            md5(COALESCE(
                json_extract_string(ftu.input_json, '$.file_path'),
                ftr.read_file_path
            )) AS file_key,
            {op_case} AS operation_type,
            -- Only meaningful for write/edit (chars actually written). Other
            -- ops have no comparable "size in chars" -- leave NULL.
            CASE WHEN ftu.tool_name IN ('Write', 'Edit', 'MultiEdit', 'NotebookEdit')
                THEN length(json_extract_string(ftu.input_json, '$.content'))
                ELSE NULL
            END AS file_size_chars
        FROM fact_tool_uses ftu
        -- is_deleted belongs in the ON clause, not just on ftu: a repaired
        -- (soft-deleted) duplicate result row would otherwise fan the join
        -- out and hand lineage_upsert a duplicate tool_use_id. The upgrade
        -- path hits exactly this -- repair soft-deletes twins at open, then
        -- this populator re-runs over the same session.
        LEFT JOIN fact_tool_results ftr
               ON ftr.tool_use_id = ftu.tool_use_id
              AND ftr.is_deleted = FALSE
        WHERE ftu.is_deleted = FALSE
          AND ftu.tool_name IN ({tool_list})
          -- Scope to the session currently being ETL'd: prior sessions'
          -- rows are already in target with matching hash_diff and would
          -- be no-ops to recompute.
          AND ftu.session_id IN (
              SELECT DISTINCT session_id FROM stg_log_entries
              WHERE session_id IS NOT NULL
          )
        """
    )
    lineage_upsert(
        conn, run=run,
        table="fact_file_operations",
        inbound_table="_inbound_file_ops",
        natural_key="tool_use_id",
        payload_cols=_FILE_OP_PAYLOAD_COLS,
        hash_cols=_FILE_OP_HASH_COLS,
    )
