<!-- path-privacy: skip-file -- references universal ~/.claude data paths (not personal) -->
# ccutils

Claude utilities for session transcripts, star schema analytics, and probably more as it comes up as a use case in my day to day.

> **Origin:** This project began as a fork of Simon Willison's [claude-code-transcripts](https://github.com/simonw/claude-code-transcripts). It has since diverged significantly with star schema analytics, modular architecture, and as a broader Claude utility.

## Installation

```bash
uv tool install -e .
```

Or run without installing:

```bash
uv run ccutils --help
```

## Quick Start

```bash
# Interactive two-phase picker - select projects, then sessions
ccutils

# Convert a single session file (opens in browser)
ccutils session.jsonl

# Export to DuckDB (v0.15 star schema + four-tier ETL + Parquet lake)
ccutils --format duckdb -o ./analytics
```

## Commands

| Command | Description |
|---------|-------------|
| `local` | Interactive picker + single-file conversion -- **default** (no subcommand needed) |
| `all` | Batch convert all sessions (HTML archive, DuckDB, or JSON) |
| `web` | Import from Claude API (auto-detects credentials from macOS keychain) |
| `explore` | Open DuckDB database in harlequin (requires `ccutils[explore]`) |
| `import` | Import Claude.ai account exports (Settings > Privacy > Export) |
| `schema` | Inspect JSON structure without exposing content (safe to share publicly) |

### local (default)

The default command. With no arguments, launches a two-phase interactive picker (projects then sessions). Pass a session file to convert it directly.

Thinking blocks and subagent sessions are included by default.

```bash
ccutils                                          # Interactive picker
ccutils session.jsonl                            # Convert file, open in browser
ccutils session.jsonl --format duckdb -o .       # v0.15 star schema DuckDB
ccutils --format duckdb -o ./analytics           # Pick sessions, star schema
ccutils -p myproject                             # Filter by project name
ccutils --flat                                   # Legacy single-list mode
ccutils --no-thinking --no-subagents             # Exclude thinking (all formats) / agents (HTML only)
ccutils --format duckdb --embed -o .             # With ColBERT embeddings
ccutils --format duckdb --with-llm-facets -o .   # + Tier 2 LLM facets (F20 via Haiku)
```

`--with-llm-facets` requires `ANTHROPIC_API_KEY` in the environment OR a `ccutils-anthropic` keychain entry (`security add-generic-password -s ccutils-anthropic -a $USER -w`). Star schema only.

Note: pairing `--batch-llm-facets` with `--format json` runs the full LLM extraction against a temporary DuckDB that's discarded after the JSON export — you pay the API cost but don't get a queryable database to inspect the F20 outputs. If you want to see the extracted facets, use `--format duckdb` instead.

### all

Batch convert every session. Agents and thinking blocks included by default.

```bash
ccutils all -o ./archive                         # HTML archive with search index
ccutils all --format duckdb -o ./analytics       # v0.15 star schema for all sessions
ccutils all --format duckdb --embed -o ./out     # With ColBERT embeddings
ccutils all --format duckdb --batch-llm-facets -o ./out  # + Tier 2 LLM facets
ccutils all -j 4 --batch-size 20 -o ./archive    # Parallel processing
ccutils all --no-agents --no-thinking             # Exclude agents and thinking (any format)
ccutils all --dry-run                            # Preview without converting
```

### import

Import Claude.ai web conversation exports (the ZIP/directory from Settings > Privacy). HTML output only.

```bash
ccutils import ./my-claude-export --open         # HTML, opens in browser
ccutils import ./export --interactive            # Pick conversations
ccutils import ./export --list                   # List without converting
```

### web

Import a session from the Claude API. Auto-detects credentials from macOS keychain.

```bash
ccutils web                                      # Interactive session picker
ccutils web SESSION_ID -o ./transcript --open     # Convert specific session
ccutils web --repo owner/name                    # Filter by GitHub repo
```

### explore

Open a star schema DuckDB database in harlequin for interactive SQL exploration.

```bash
uv pip install ccutils[explore]    # one-time setup
ccutils explore ./analytics/archive.duckdb
```

### schema

Inspect JSON file structure without exposing content. Output is safe to share publicly or paste into AI assistants.

```bash
ccutils schema conversations.json
ccutils schema ./my-claude-export/               # Inspect all files in directory
ccutils schema ./export --json > schema.json     # Machine-readable output
```

## Export Formats

Three formats: `html`, `duckdb`, `json`. All three are first-class; `duckdb` and `json` both write the v0.15 star schema.

### HTML Transcripts

Clean, mobile-friendly HTML with pagination, commit timeline, tool stats, and full-text search.

```bash
ccutils -o ./transcript --open
ccutils all -o ./archive                    # Archive with master index and search
```

### DuckDB Analytics (v0.15 star schema)

```bash
ccutils --format duckdb -o ./analytics
```

v0.15 rebuilds the ETL as a four-tier pipeline:

1. **Tier 0** -- raw JSONL on disk (Claude Code writes).
2. **Tier 1** -- Parquet lake under `<output>/parquet_lake/projects/<project>/<session>/`. Typed-columnar cache of every parsed JSONL entry. Persistent and re-derivable; the DuckDB warehouse can be torn down and rebuilt from Parquet without re-parsing JSONL.
3. **Tier 2** -- DuckDB staging tables (`stg_log_entries`, `stg_task_agent_map`) loaded from Parquet via `read_parquet()`.
4. **Tier 3** -- DuckDB warehouse: dimensions, facts, and semantic views consumers query.

Every fact carries the v0.15 lineage convention: `created_at`, `last_updated_at`, `created_by_version_key`, `last_updated_by_version_key`, `etl_run_id`, `record_source`, `hash_diff`, plus soft-delete (`is_deleted`, `deleted_at`). Re-running ETL on unchanged source is a no-op (the `hash_diff` gate prevents spurious UPDATEs). Mutations are tracked via `dim_etl_version` + `fact_etl_runs`; DDL migrations are tracked via `meta_schema_version`.

**Tables populated by `run_v15_etl`:**

- **Lineage / Meta:** `dim_etl_version`, `fact_etl_runs`, `meta_schema_version`.
- **Dimensions:** `dim_session` (with intent/complexity/outcome/domain enrichment + subagent linkage), `dim_project`, `dim_tool`, `dim_model`, `dim_file`, `dim_session_chain`, `dim_prompt`, `dim_facet_type` (facet registry).
- **Core facts:** `fact_messages`, `fact_tool_uses`, `fact_tool_results` (R1 structured `toolUseResult` payloads: Edit `structured_patch`, Bash `exit_code` / `interrupted`, Read `num_lines`, Agent rollups), `fact_token_usage` (R11 cache split: `cache_creation_5m_tokens` + `cache_creation_1h_tokens`), `fact_session_summary`.
- **Entry-type facts:** `fact_attachments`, `fact_progress_events`, `fact_system_events`, `fact_meta_events` (permission-mode time series), `fact_file_history_snapshots`, `fact_queue_operations`, `fact_pr_links`.
- **Derived:** `fact_file_operations` + `bridge_session_file`, `fact_diagnostics`, `fact_plan_revisions` (structural outcome from `fact_tool_results.is_error`), `fact_agent_delegations` (cross-session linkage via `dim_session.agent_id`), `fact_errors`, `fact_tool_chain_steps`.
- **Facets:** `fact_session_facets` Tier 1 (F01-F19, SQL-computed; always on). Tier 2 (F20+, LLM-extracted via Haiku) is opt-in via `--with-llm-facets` / `--batch-llm-facets`.

**Not yet populated (DDL stubs only):** `fact_content_blocks`, `fact_code_blocks`, `fact_entity_mentions`, `fact_session_embeddings`, `fact_tool_input_params`. `fact_turn_durations` / `fact_stop_events` are subsumed by `fact_system_events`; `fact_tool_calls` by `fact_tool_uses` + `fact_tool_results`.

```sql
-- Sessions ranked by uncached-equivalent token cost
SELECT
  ds.session_id,
  fss.total_input_tokens,
  fss.total_output_tokens,
  fss.total_cache_creation_5m_tokens,
  fss.total_cache_creation_1h_tokens,
  fss.total_cache_read_tokens,
  fss.total_uncached_equivalent_tokens
FROM fact_session_summary fss
JOIN dim_session ds USING (session_key)
ORDER BY total_uncached_equivalent_tokens DESC
LIMIT 20;

-- Tool usage by category
SELECT dt.tool_category, COUNT(*) AS uses
FROM fact_tool_uses ftu
JOIN dim_tool dt USING (tool_key)
GROUP BY dt.tool_category
ORDER BY uses DESC;

-- Bash invocations that exited non-zero or were interrupted (R1 toolUseResult capture)
SELECT
  ftu.session_id,
  json_extract_string(ftu.input_json, '$.command') AS command,
  ftr.bash_exit_code,
  ftr.bash_interrupted,
  ftr.timestamp
FROM fact_tool_uses ftu
JOIN fact_tool_results ftr USING (tool_use_id)
WHERE ftu.tool_name = 'Bash'
  AND (ftr.bash_exit_code <> 0 OR ftr.bash_interrupted = TRUE)
ORDER BY ftr.timestamp DESC
LIMIT 20;

-- ETL lineage: when did each batch run, what version produced it, what was touched
SELECT
  fer.etl_run_id,
  fer.started_at,
  dev.ccutils_version,
  fer.sessions_seen,
  fer.facts_inserted,
  fer.facts_updated,
  fer.status
FROM fact_etl_runs fer
LEFT JOIN dim_etl_version dev USING (version_key)
ORDER BY fer.started_at DESC
LIMIT 10;
```

### JSON Export

The v0.15 star schema as a JSON directory tree (`meta.json` + `dimensions/` + `facts/`).

```bash
ccutils --format json -o ./json-export/
```

## Common Options

```bash
# Output
-o, --output PATH          Output directory or file
--format FORMAT            html, duckdb, json (+ both for all)
--open                     Open result in browser

# Content (included by default -- use flags to exclude)
--no-thinking              Exclude thinking from outputs. Drops thinking from
                           dim_session messages + Tier 2 facet inputs and
                           clears the staging artifact (fact_messages already
                           excludes thinking by projection). Parquet lake is
                           unaffected -- delete it post-run if needed.
--no-subagents             Exclude related agent sessions (local)
--no-agents                Exclude agent-* session files (all)
--private                  Sanitize file paths for sharing (HTML only; v0.15 sanitization not yet wired)

# Selection
--flat                     Flat single-list mode (local)
--expand-chains            Show individual sessions in resumed chains (local)
-p, --project TEXT         Filter by project name (local, all)
--dry-run                  Preview without converting (all)

# Embeddings (local and all, star schema only)
--embed [MODEL]            Run ColBERT embeddings (optionally specify model)

# Batch processing (all command)
-s, --source PATH          Source directory (default: ~/.claude/projects)
-j, --jobs N               Parallel workers (default: 1)
--batch-size N             Sessions per batch (default: 10)
-q, --quiet                Suppress output except errors
--no-search-index          Skip search index generation
```

## Documentation

- [Star Schema Reference](docs/STAR_SCHEMA.md) -- table definitions, populator notes, example queries.
- [Facet & Cluster Pipeline](docs/FACET_CLUSTER_PIPELINE.md) -- Tier 1/2/3 facet design, status, and roadmap.

## Development

```bash
uv run pytest              # Run tests (~934 + 1 skipped live-API)
uv run ccutils --help      # Run development version
uv run pytest --cov=ccutils  # Coverage
```

## License

Apache-2.0
