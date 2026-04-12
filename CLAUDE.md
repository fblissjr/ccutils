# Start Here

Uses uv. Run tests like this:

    uv run pytest

Run the development version of the tool like this:

    uv run ccutils --help

Always practice TDD: write a failing test, watch it fail, then make it pass.

Commit early and often. Commits should bundle the test, implementation, and documentation changes together.

## Project Structure

```
ccutils/
├── src/ccutils/
│   ├── __init__.py           # Public API re-exports
│   ├── sanitize.py           # Path sanitization for --private mode
│   ├── cli/                   # CLI commands
│   │   ├── __init__.py       # CLI group and entry point
│   │   ├── local.py          # local command (default) -- picker + single-file conversion
│   │   ├── web.py            # web command
│   │   ├── all.py            # all command
│   │   ├── import_cmd.py     # import command (Claude.ai exports)
│   │   ├── schema.py         # schema command (JSON structure inspector)
│   │   └── utils.py          # CLI utilities
│   ├── api/                   # API client and credentials
│   │   └── __init__.py
│   ├── parsers/              # Session file parsing utilities
│   │   ├── __init__.py       # Public API exports
│   │   ├── jsonl_reader.py   # Canonical JSONL parser (iter_session_entries, iter_all_session_entries)
│   │   ├── history.py        # ~/.claude/history.jsonl parser (HistoryEntry, iter_history_entries)
│   │   ├── session.py        # JSONL/JSON session parsing
│   │   ├── discovery.py      # Session discovery + two-phase selection UI
│   │   ├── metadata.py       # SessionMetadata dataclass + rich extraction
│   │   ├── claude_ai.py      # Claude.ai export parser
│   │   └── schema_inspector.py # JSON structure analysis
│   ├── schemas/              # Schema definitions
│   │   ├── __init__.py       # Unified exports for both schemas
│   │   ├── simple/           # Simple 4-table schema
│   │   │   ├── __init__.py
│   │   │   ├── schema.py     # DDL for simple schema
│   │   │   └── etl.py        # Simple schema ETL
│   │   └── star/             # Star schema (27 tables + 13 views)
│   │       ├── __init__.py   # Public API exports
│   │       ├── schema.py     # DDL for star schema tables + semantic views
│   │       ├── etl.py        # Main ETL pipeline
│   │       ├── semantic.py   # Semantic model generation
│   │       ├── extractors.py # Code blocks, entities, file extraction
│   │       ├── heuristics.py # Keyword/metric-based classification
│   │       ├── history_etl.py# History.jsonl -> dim_prompt ETL
│   │       ├── json_export.py# JSON export for star schema
│   │       └── utils.py      # Key generation, tool/model classification, dim_date helper
│   ├── export/                # Export format handlers
│   │   ├── __init__.py
│   │   ├── html.py           # HTML generation
│   │   └── duckdb_archive.py # DuckDB batch export + finalize_star_schema()
│   ├── tui/                   # Terminal UI components
│   │   ├── __init__.py
│   │   ├── theme.py          # Color theme
│   │   ├── formatters.py     # Label formatters
│   │   ├── layout.py         # Table layout
│   │   ├── components.py     # Reusable UI components
│   │   └── selection.py      # Interactive selection
│   └── templates/            # Jinja2 templates for HTML export
│       ├── base.html
│       ├── macros.html       # Shared rendering macros (tools, messages, etc.)
│       ├── page.html
│       ├── master_index.html # Archive-level index
│       ├── project_index.html
│       ├── index.html        # Per-session index
│       ├── search.js         # Per-session search (Jinja2 template)
│       └── global_search.js  # Archive-wide search (Jinja2 template)
├── tests/
│   ├── conftest.py                   # Shared fixtures (sample_session_file, interrupted_session_file, etc.)
│   └── test_*.py                     # 23 test files (star schema split across ddl/etl/analytics/advanced)
├── docs/
│   └── STAR_SCHEMA.md        # Star schema documentation
└── README.md
```

## Key Components

### 1. CLI Commands
- `local` - **default command**: pass a file to convert it (`ccutils session.jsonl`), or no args for interactive two-phase picker. `--flat` for legacy single-list mode
- `web` - Import from Claude API (auto-detects credentials from macOS keychain)
- `all` - Batch convert all sessions (supports parallel processing with `-j`)
- `import` - Import Claude.ai account exports (Settings > Privacy > Export)
- `schema` - Inspect JSON structure without exposing content (safe to share publicly)
- `convert` - Hidden alias for `local` (backwards compatibility)

### 2. Export Formats
Three output formats with two schema types:

**Simple schema** (4 tables: `sessions`, `messages`, `tool_calls`, `thinking`):
- `--format duckdb` - DuckDB database file
- `--format json` - Single JSON file with nested tables

**Star schema** (27 tables + 13 views):
- `--format duckdb-star` - DuckDB database file
- `--format json-star` - Directory with meta.json + dimensions/*.json + facts/*.json
- Modular package at `schemas/star/` (schema, etl, semantic, extractors, heuristics, json_export, utils)
- See `create_star_schema()`, `run_star_schema_etl()`, `finalize_star_schema()`, `export_star_schema_to_json()` functions
- `finalize_star_schema(conn)` MUST be called after all ETL runs -- populates session chains, agent delegations, file bridge, depth levels, and `_incl_agents` metric rollup
- Heuristic classification (intent, complexity, outcome, domain, error_type) runs during ETL -- no LLM required
- `--embed [MODEL]` flag available on both `local` and `all` commands (requires pylate optional dependency)
- Full documentation in docs/STAR_SCHEMA.md

**Schema inference**: Schema type is auto-inferred from `--format` -- `duckdb-star` and `json-star` use star schema, plain `duckdb` and `json` use simple schema.

**Defaults**: Thinking blocks and subagents/agents are included by default. Use `--no-thinking`, `--no-subagents` (local), or `--no-agents` (all) to exclude them.

### 3. Star Schema Tables (27 tables + 13 views)

**Core Dimensions (7):** dim_session (with heuristics, entrypoint/custom_title/permission_mode/agent_type/agent_description), dim_project, dim_tool, dim_model, dim_date, dim_time, dim_prompt (from ~/.claude/history.jsonl)

**Core Facts (6):** fact_messages (with actual_input/output/cache_read_tokens), fact_tool_calls (with duration_seconds), fact_session_summary (with _incl_agents rollup, actual token totals, turn duration, diagnostics, hook runs, stop counts), fact_file_operations, fact_errors (with heuristic error_type), fact_tool_chain_steps

**Granular (5):** dim_file (with language), dim_session_chain, fact_content_blocks, fact_code_blocks, fact_entity_mentions

**Telemetry Facts (4):** fact_token_usage (per-API-response token breakdown), fact_turn_durations (actual turn timing), fact_diagnostics (LSP diagnostics), fact_stop_events (stop reasons and hooks)

**Agent/Bridge/Staging (3):** fact_agent_delegations (with denormalized metrics), bridge_session_file, stg_task_agent_map

**Optional (2):** fact_session_embeddings (pylate), fact_tool_input_params

**Views (13):** semantic_sessions, semantic_messages, semantic_tool_calls, semantic_file_operations, semantic_session_chains, semantic_agent_delegations, semantic_file_evolution, semantic_tool_patterns, semantic_project_context, semantic_project_files, semantic_token_usage, semantic_cost_analysis, semantic_prompt_history

### 4. Token Tracking

**Actual tokens** (from API usage data on assistant messages, since v0.13.0):
- `fact_token_usage`: per-API-response breakdown (input, output, cache creation, cache read, ephemeral tiers, service_tier, speed)
- `fact_messages`: `actual_input_tokens`, `actual_output_tokens`, `cache_read_tokens`
- `fact_session_summary`: aggregated `actual_input_tokens`, `actual_output_tokens`, `cache_creation_tokens`, `cache_read_tokens`
- `semantic_cost_analysis` view: includes `cache_hit_rate_pct`

**Estimated tokens** (word-count heuristic, all versions):
- `estimate_tokens()` in `schemas/star/extractors.py`: text x1.3, code x1.5
- `fact_session_summary`: `total_estimated_tokens`, `total_thinking_tokens`, `total_tool_io_tokens`
- `_incl_agents` rollup columns populated by `finalize_star_schema()`

Old sessions without usage data get NULL for actual columns; estimated tokens remain available for all sessions.

### 5. Simple Schema ETL Architecture

`schemas/simple/etl.py` uses a shared extraction core:
- `_extract_session_core()` → returns `SimpleExtractionResult` dataclass with all parsed data
- `export_session_to_duckdb()` → thin wrapper that INSERTs the result into DuckDB
- `_extract_session_data()` → thin wrapper that converts the result to dicts for JSON export

### 6. Heuristic Classification

Runs during ETL with zero external dependencies:
```python
from ccutils import classify_intent, classify_complexity, classify_outcome, classify_domain, classify_error_type
```

## Testing

Run all tests:

    uv run pytest

Run star schema tests specifically:

    uv run pytest tests/test_star_schema_ddl.py tests/test_star_schema_etl.py tests/test_star_schema_analytics.py tests/test_star_schema_advanced.py -v

Run with coverage:

    uv run pytest --cov=ccutils

## Common Workflows

### Adding a new dimension
1. Write failing tests in `test_star_schema_ddl.py` (table exists, columns correct) and `test_star_schema_etl.py` (rows populated)
2. Add CREATE TABLE in `schemas/star/schema.py`
3. Add ETL logic in `schemas/star/etl.py`
4. Run tests green, then update docs/STAR_SCHEMA.md

### Adding a new fact table
1. Write failing tests in `test_star_schema_ddl.py` (schema) and `test_star_schema_etl.py` or `test_star_schema_advanced.py` (ETL)
2. Add CREATE TABLE in `schemas/star/schema.py`
3. Add extraction + INSERT in `schemas/star/etl.py` (per-session) or `export/duckdb_archive.py` (post-ETL cross-session)
4. If post-ETL, call from `finalize_star_schema()` and make idempotent (DELETE before INSERT)
5. Run tests green, then update docs/STAR_SCHEMA.md

### Star schema ETL pipeline order
```
create_star_schema(conn)                    # DDL
run_star_schema_etl(conn, ...)              # Per-session ETL (call once per session)
finalize_star_schema(conn, history_path=..) # Post-ETL: chains, delegations, file bridge, depths, history
create_semantic_model(conn)                 # Semantic views metadata
# Optional: EmbeddingPipeline(conn).embed_sessions(conn)
```

Key details:
- `run_star_schema_etl` reads `.meta.json` sidecar for agent_type/agent_description automatically
- `finalize_star_schema` accepts optional `history_path` to load `~/.claude/history.jsonl` into `dim_prompt`
- `load_history(conn, path)` can also be called directly from `schemas.star.history_etl`

## HTML Export Gotchas

- CSS classes used in `templates/macros.html` MUST be defined in `static/transcript.css` -- Jinja2 won't warn
- `--snapshot-update` needed after ANY change to `transcript.css`, `macros.html`, or `base.html` (CSS is inlined in every page)
- `global_search.js` and `search.js` are Jinja2 templates (not static files) -- rendered via `_jinja_env.get_template()`
- Template variables render empty (not error) if not passed -- always test rendered HTML for expected content
- Never use real usernames/paths in docstrings or test fixtures -- use `/Users/dev/workspace/project`

### HTML Security

- `render_markdown_text()` sanitizes output via `nh3.clean()` to strip dangerous HTML (`<script>`, event handlers, `<iframe>`, etc.)
- `nh3.clean(raw, attributes={"code": {"class"}})` preserves `class` on `<code>` elements for fenced code block syntax highlighting
- Jinja2 environment uses `autoescape=True` -- the `|safe` filter in macros is safe because all content is either pre-sanitized (nh3) or pre-escaped (`html.escape()`)
- `base.html` includes a Content-Security-Policy meta tag that blocks external scripts, iframes, and restricts fetch to same-origin
- Do NOT remove the CSP or nh3 sanitization without understanding the XSS implications
