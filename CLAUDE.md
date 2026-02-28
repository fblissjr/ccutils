# Start Here

| Document | Purpose |
|----------|---------|
| [internal/hub.md](internal/hub.md) | Central documentation hub |
| [internal/state/current.md](internal/state/current.md) | Current project state - read first for active work |

---

Uses uv. Run tests like this:

    uv run pytest

Run the development version of the tool like this:

    uv run ccutils --help

Always practice TDD: write a failing test, watch it fail, then make it pass.

Commit early and often. Commits should bundle the test, implementation, and documentation changes together.

Run Black to format code before you commit:

    uv run black .

## Project Structure

```
ccutils/
├── src/ccutils/
│   ├── __init__.py           # Public API re-exports
│   ├── cli/                   # CLI commands
│   │   ├── __init__.py       # CLI group and entry point
│   │   ├── local.py          # local command
│   │   ├── web.py            # web command
│   │   ├── json_cmd.py       # json command
│   │   ├── all.py            # all command
│   │   ├── explore.py        # explore command
│   │   └── utils.py          # CLI utilities
│   ├── api/                   # API client and credentials
│   │   └── __init__.py
│   ├── parsers/              # Session file parsing utilities
│   │   ├── __init__.py       # Public API exports
│   │   ├── session.py        # JSONL/JSON session parsing
│   │   ├── discovery.py      # Session discovery + two-phase selection UI
│   │   └── metadata.py       # SessionMetadata dataclass + rich extraction
│   ├── schemas/              # Schema definitions
│   │   ├── __init__.py       # Unified exports for both schemas
│   │   ├── simple/           # Simple 4-table schema
│   │   │   ├── __init__.py
│   │   │   ├── schema.py     # DDL for simple schema
│   │   │   └── etl.py        # Simple schema ETL
│   │   └── star/             # Star schema (25+ tables)
│   │       ├── __init__.py   # Public API exports
│   │       ├── schema.py     # DDL for star schema tables
│   │       ├── etl.py        # Main ETL pipeline
│   │       ├── semantic.py   # Semantic model generation
│   │       ├── extractors.py # Code blocks, entities, file extraction
│   │       ├── heuristics.py # Keyword/metric-based classification
│   │       ├── json_export.py# JSON export for star schema
│   │       └── utils.py      # Key generation, tool/model classification
│   ├── export/                # Export format handlers
│   │   ├── __init__.py
│   │   ├── html.py           # HTML generation
│   │   └── duckdb_archive.py # DuckDB batch export
│   ├── explorer/             # Data Explorer SPA
│   │   ├── index.html
│   │   ├── css/styles.css
│   │   └── js/{app,state,duckdb,query-builder,ui}.js
│   └── templates/            # Jinja2 templates for HTML export
│       ├── base.html
│       ├── page.html
│       └── star_schema_dashboard.html
├── tests/
│   ├── test_generate_html.py         # HTML generation tests
│   ├── test_metadata.py              # SessionMetadata extraction tests
│   ├── test_star_schema.py           # Star schema & ETL tests
│   ├── test_json_export.py           # JSON export tests
│   └── test_all.py                   # Batch conversion tests
├── docs/
│   ├── STAR_SCHEMA.md        # Star schema documentation
│   └── DATA_EXPLORER.md      # Data explorer documentation
└── README.md
```

## Key Components

### 1. CLI Commands
- `local` - Two-phase session picker: select project(s) then session(s) with rich metadata tables. `--flat` for legacy single-list mode.
- `web` - Import from Claude API
- `json` - Convert specific JSON/JSONL files
- `all` - Batch convert all sessions
- `explore` - Launch Data Explorer web server

### 2. Export Formats
Three output formats with two schema types:

**Simple schema** (4 tables: `sessions`, `messages`, `tool_calls`, `thinking`):
- `--format duckdb` - DuckDB database file
- `--format json` - Single JSON file with nested tables

**Star schema** (22 tables + 8 views):
- `--format duckdb-star` - DuckDB database file
- `--format json-star` - Directory with meta.json + dimensions/*.json + facts/*.json
- Modular package at `schemas/star/` (schema, etl, semantic, extractors, heuristics, json_export, utils)
- See `create_star_schema()`, `run_star_schema_etl()`, `export_star_schema_to_json()` functions
- Heuristic classification (intent, complexity, outcome, domain) runs during ETL -- no LLM required
- Visual explorer at `explorer/`
- Full documentation in docs/STAR_SCHEMA.md and docs/DATA_EXPLORER.md

**Hybrid CLI**: Use `--schema simple|star` with `--format duckdb|json` for explicit control.

### 3. Star Schema Tables (22 tables + 8 views)

**Core Dimensions (6):** dim_session (with intent/complexity/outcome/domain heuristics), dim_project, dim_tool, dim_model, dim_date, dim_time

**Core Facts (6):** fact_messages, fact_tool_calls (with duration_seconds), fact_session_summary (with total_errors, unique_tools_used, etc.), fact_file_operations, fact_errors (with heuristic error_type), fact_tool_chain_steps (with next_tool_key, is_error)

**Granular (5):** dim_file (with language), dim_session_chain, fact_content_blocks, fact_code_blocks, fact_entity_mentions

**Agent/Bridge/Staging (3):** fact_agent_delegations (with denormalized metrics), bridge_session_file, stg_task_agent_map

**Optional (2):** fact_session_embeddings (pylate), fact_tool_input_params

**Views (8):** semantic_sessions, semantic_messages, semantic_tool_calls, semantic_file_operations, semantic_session_chains, semantic_agent_delegations, semantic_file_evolution, semantic_tool_patterns

### 4. Heuristic Classification

Runs during ETL with zero external dependencies:
```python
from ccutils import classify_intent, classify_complexity, classify_outcome, classify_domain, classify_error_type
```

## Testing

Run all tests:

    uv run pytest

Run star schema tests specifically:

    uv run pytest tests/test_star_schema.py -v

Run with coverage:

    uv run pytest --cov=ccutils

## Common Workflows

### Adding a new dimension
1. Add CREATE TABLE statement in `schemas/star/schema.py`
2. Add ETL logic in `schemas/star/etl.py` to populate the dimension
3. Write tests in `test_star_schema.py`
4. Update docs/STAR_SCHEMA.md

### Adding a new fact table
1. Add CREATE TABLE statement in `schemas/star/schema.py`
2. Add data collection logic in ETL extraction phase
3. Add INSERT statement in ETL loading phase
4. Write tests covering schema and ETL
5. Update documentation
