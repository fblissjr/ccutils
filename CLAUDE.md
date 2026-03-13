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
│   │   ├── local.py          # local command (default)
│   │   ├── web.py            # web command
│   │   ├── json_cmd.py       # convert command (single-file conversion)
│   │   ├── all.py            # all command
│   │   ├── explore.py        # explore command
│   │   ├── import_cmd.py     # import command (Claude.ai exports)
│   │   ├── schema.py         # schema command (JSON structure inspector)
│   │   └── utils.py          # CLI utilities
│   ├── api/                   # API client and credentials
│   │   └── __init__.py
│   ├── parsers/              # Session file parsing utilities
│   │   ├── __init__.py       # Public API exports
│   │   ├── jsonl_reader.py   # Canonical JSONL parser (iter_session_entries, iter_loglines)
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
│   │   └── star/             # Star schema (22 tables + 10 views)
│   │       ├── __init__.py   # Public API exports
│   │       ├── schema.py     # DDL for star schema tables + semantic views
│   │       ├── etl.py        # Main ETL pipeline
│   │       ├── semantic.py   # Semantic model generation
│   │       ├── extractors.py # Code blocks, entities, file extraction
│   │       ├── heuristics.py # Keyword/metric-based classification
│   │       ├── json_export.py# JSON export for star schema
│   │       └── utils.py      # Key generation, tool/model classification
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
│   ├── explorer/             # Data Explorer SPA
│   │   ├── index.html
│   │   ├── css/styles.css
│   │   └── js/{app,state,duckdb,query-builder,ui}.js
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
│   ├── STAR_SCHEMA.md        # Star schema documentation
│   └── DATA_EXPLORER.md      # Data explorer documentation
└── README.md
```

## Key Components

### 1. CLI Commands
- `local` - Two-phase session picker: select project(s) then session(s) with rich metadata tables. `--flat` for legacy single-list mode. **default command**
- `web` - Import from Claude API (auto-detects credentials from macOS keychain)
- `convert` - Convert a single JSON/JSONL file or URL (supports all output formats: html, duckdb, json)
- `all` - Batch convert all sessions (supports parallel processing with `-j`)
- `explore` - Launch Data Explorer web server
- `import` - Import Claude.ai account exports (Settings > Privacy > Export)
- `schema` - Inspect JSON structure without exposing content (safe to share publicly)

### 2. Export Formats
Three output formats with two schema types:

**Simple schema** (4 tables: `sessions`, `messages`, `tool_calls`, `thinking`):
- `--format duckdb` - DuckDB database file
- `--format json` - Single JSON file with nested tables

**Star schema** (22 tables + 10 views):
- `--format duckdb-star` - DuckDB database file
- `--format json-star` - Directory with meta.json + dimensions/*.json + facts/*.json
- Modular package at `schemas/star/` (schema, etl, semantic, extractors, heuristics, json_export, utils)
- See `create_star_schema()`, `run_star_schema_etl()`, `finalize_star_schema()`, `export_star_schema_to_json()` functions
- `finalize_star_schema(conn)` MUST be called after all ETL runs -- populates session chains, agent delegations, file bridge, depth levels, and `_incl_agents` metric rollup
- Heuristic classification (intent, complexity, outcome, domain, error_type) runs during ETL -- no LLM required
- `--embed` flag available on both `local` and `all` commands (requires pylate optional dependency)
- Visual explorer at `explorer/`
- Full documentation in docs/STAR_SCHEMA.md and docs/DATA_EXPLORER.md

**Hybrid CLI**: Use `--schema simple|star` with `--format duckdb|json` for explicit control.

### 3. Star Schema Tables (22 tables + 10 views)

**Core Dimensions (6):** dim_session (with intent/complexity/outcome/domain heuristics + first_user_message/last_assistant_message), dim_project, dim_tool, dim_model, dim_date, dim_time

**Core Facts (6):** fact_messages, fact_tool_calls (with duration_seconds), fact_session_summary (with _incl_agents rollup, total_errors, unique_tools_used, etc.), fact_file_operations, fact_errors (with heuristic error_type), fact_tool_chain_steps (with next_tool_key, is_error)

**Granular (5):** dim_file (with language), dim_session_chain, fact_content_blocks, fact_code_blocks, fact_entity_mentions

**Agent/Bridge/Staging (3):** fact_agent_delegations (with denormalized metrics), bridge_session_file, stg_task_agent_map

**Optional (2):** fact_session_embeddings (pylate), fact_tool_input_params

**Views (10):** semantic_sessions, semantic_messages, semantic_tool_calls, semantic_file_operations, semantic_session_chains, semantic_agent_delegations, semantic_file_evolution, semantic_tool_patterns, semantic_project_context, semantic_project_files

### 4. Token Estimation

Both schemas estimate tokens using a word-count heuristic (`estimate_tokens()` in `schemas/star/extractors.py`):
- Text content: words x 1.3
- Code content: words x 1.5 (detected by presence of `` ``` ``, `def `, `function `)

Token counts cover text blocks, thinking blocks, tool input JSON, and tool result text.

**Simple schema:** `sessions.estimated_tokens` (total across all sources)
**Star schema:** `fact_session_summary` has `total_estimated_tokens`, `total_thinking_tokens`, `total_tool_io_tokens`, plus `_incl_agents` rollup columns (`total_estimated_tokens_incl_agents`, `total_tool_calls_incl_agents`, `total_errors_incl_agents`, `total_duration_incl_agents`) populated by `finalize_star_schema()`

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
create_star_schema(conn)       # DDL
run_star_schema_etl(conn, ...) # Per-session ETL (call once per session)
finalize_star_schema(conn)     # Post-ETL: chains, delegations, file bridge, depths
create_semantic_model(conn)    # Semantic views metadata
# Optional: EmbeddingPipeline(conn).embed_sessions(conn)
```

## HTML Export Gotchas

- CSS classes used in `templates/macros.html` MUST be defined in `static/transcript.css` -- Jinja2 won't warn
- `--snapshot-update` needed after ANY change to `transcript.css` or `macros.html` (CSS is inlined in every page)
- `global_search.js` and `search.js` are Jinja2 templates (not static files) -- rendered via `_jinja_env.get_template()`
- Template variables render empty (not error) if not passed -- always test rendered HTML for expected content
- Never use real usernames/paths in docstrings or test fixtures -- use `/Users/dev/workspace/project`
