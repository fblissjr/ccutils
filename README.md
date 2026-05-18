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

# Export to DuckDB for SQL analytics
ccutils --format duckdb -o ./archive

# Export with star schema (v0.15 four-tier ETL + Parquet lake)
ccutils --format duckdb-star -o ./analytics
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
ccutils session.jsonl --format duckdb-star -o .  # Star schema from file
ccutils --format duckdb-star -o ./analytics      # Pick sessions, star schema
ccutils -p myproject                             # Filter by project name
ccutils --flat                                   # Legacy single-list mode
ccutils --no-thinking --no-subagents             # Exclude thinking/agents
ccutils --format duckdb-star --embed -o .        # With ColBERT embeddings
```

### all

Batch convert every session. Agents and thinking blocks included by default.

```bash
ccutils all -o ./archive                         # HTML archive with search index
ccutils all --format duckdb-star -o ./analytics   # Star schema for all sessions
ccutils all --format duckdb-star --embed -o ./out # With ColBERT embeddings
ccutils all -j 4 --batch-size 20 -o ./archive    # Parallel processing
ccutils all --no-agents --no-thinking             # Exclude agents and thinking
ccutils all --dry-run                            # Preview without converting
```

### import

Import Claude.ai web conversation exports (the ZIP/directory from Settings > Privacy).

```bash
ccutils import ./my-claude-export --open         # HTML, opens in browser
ccutils import ./export --format duckdb -o data.duckdb  # DuckDB
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

Schema type is auto-inferred from `--format`: `duckdb-star` and `json-star` use star schema, plain `duckdb` and `json` use simple.

### HTML Transcripts

Clean, mobile-friendly HTML with pagination, commit timeline, tool stats, and full-text search.

```bash
ccutils -o ./transcript --open
ccutils all -o ./archive                    # Archive with master index and search
```

### DuckDB Analytics

#### Simple Schema (4 tables)

```bash
ccutils --format duckdb -o ./archive
```

Tables: `sessions`, `messages`, `tool_calls`, `thinking`

#### Star Schema (v0.15, on the `etl-rethink` branch)

```bash
ccutils --format duckdb-star -o ./analytics
```

v0.15 rebuilds the ETL as a four-tier pipeline:

1. **Tier 0** -- raw JSONL on disk (Claude Code writes).
2. **Tier 1** -- Parquet lake under `<output>/parquet_lake/projects/<project>/<session>/`. Typed-columnar cache of every parsed JSONL entry. Persistent and re-derivable; the DuckDB warehouse can be torn down and rebuilt from Parquet without re-parsing JSONL.
3. **Tier 2** -- DuckDB staging tables (`stg_log_entries`, `stg_task_agent_map`) loaded from Parquet via `read_parquet()`.
4. **Tier 3** -- DuckDB warehouse: dimensions, facts, and semantic views consumers query.

Every fact carries the v0.15 lineage convention: `created_at`, `last_updated_at`, `created_by_version_key`, `last_updated_by_version_key`, `etl_run_id`, `record_source`, `hash_diff`, plus soft-delete (`is_deleted`, `deleted_at`). Re-running ETL on unchanged source is a no-op (the `hash_diff` gate prevents spurious UPDATEs). Mutations are tracked via `dim_etl_version` + `fact_etl_runs`; DDL migrations are tracked via `meta_schema_version`.

**Tables wired by the v0.15 orchestrator (`run_v15_etl`):**

- **Dimensions:** `dim_etl_version`, `dim_session`, `dim_project`, `dim_tool`, `dim_model` (minimal envelope -- heuristic enrichment is Phase D; `dim_date` / `dim_time` DDL exists but is not wired yet)
- **Core facts:** `fact_messages`, `fact_tool_uses`, `fact_tool_results` (with structured per-tool `toolUseResult` payloads -- Edit `structuredPatch`, Bash `exit_code`/`interrupted`, Read `numLines`, Agent rollups), `fact_token_usage` (R11 cache split: `cache_creation_5m_tokens` + `cache_creation_1h_tokens`), `fact_session_summary`
- **Entry-type facts:** `fact_attachments`, `fact_progress_events`, `fact_system_events`, `fact_meta_events` (permission-mode time series), `fact_file_history_snapshots`, `fact_queue_operations`, `fact_pr_links`
- **Lineage:** `fact_etl_runs`, `meta_schema_version`

**Also populated by Phase D ports:** `dim_file`, `bridge_session_file`, `fact_file_operations`, `fact_diagnostics`, `fact_plan_revisions` (with structural outcome classification from `fact_tool_results.is_error`), `fact_agent_delegations` (with cross-session subagent linkage via `dim_session.agent_id`), `fact_errors`, `fact_tool_chain_steps`, `dim_session_chain`, `dim_prompt` (from prompt history JSONL), plus heuristic enrichment on `dim_session` (intent, complexity, outcome, domain, first_user_message, last_assistant_message, is_agent, agent_id, parent_session_key, agent_type, agent_description, depth_level).

**Pending:** the granular content/code/entity extracts (`fact_content_blocks`, `fact_code_blocks`, `fact_entity_mentions`), the legacy stop/turn telemetry that the v0.15 `fact_system_events` already overlaps (`fact_stop_events`, `fact_turn_durations`), and `fact_session_embeddings` / `fact_tool_input_params`. The semantic views are now rebuilt against the v0.15 facts and return rows.

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

```bash
# Simple schema - single file
ccutils --format json -o ./sessions.json

# Star schema - directory structure (dimensions/ + facts/ + meta.json)
ccutils --format json-star -o ./star-export/
```

## Common Options

```bash
# Output
-o, --output PATH          Output directory or file
--format FORMAT            html, duckdb, duckdb-star, json, json-star (+ both for all)
--open                     Open result in browser

# Content (included by default -- use flags to exclude)
--no-thinking              Exclude thinking blocks
--no-subagents             Exclude related agent sessions (local)
--no-agents                Exclude agent-* session files (all)
--private                  Sanitize file paths for sharing

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

- [Star Schema Reference](docs/STAR_SCHEMA.md) -- table definitions, ETL capabilities, heuristic classification, example queries

## Development

```bash
uv run pytest              # Run tests
uv run ccutils --help      # Run development version
uv run pytest --cov=ccutils  # Coverage
```

> On the `etl-rethink` branch the v0.15 pipeline (133 tests) passes; legacy tests that target dropped fact columns will be ported / replaced as Phase D lands.

## License

Apache-2.0
