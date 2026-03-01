# ccutils

Claude utilities for session transcripts, star schema analytics, data exploration, and probably more as it comes up as a use case in my day to day.

> **Origin:** This project began as a fork of Simon Willison's [claude-code-transcripts](https://github.com/simonw/claude-code-transcripts). It has since diverged significantly with star schema analytics, a visual data explorer, modular architecture, and as a broader Claude utility.

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

# Export to DuckDB for SQL analytics
ccutils local --format duckdb -o ./archive

# Export with star schema (22 tables + 8 views)
ccutils local --format duckdb-star -o ./analytics

# Launch visual data explorer
ccutils explore ./analytics/archive.duckdb

# Sanitize file paths for sharing
ccutils local --format duckdb-star --private -o ./analytics
```

## Commands

| Command | Description |
|---------|-------------|
| `local` | Two-phase session picker: projects then sessions with rich metadata -- **default** |
| `web` | Import from Claude API (auto-detects credentials from macOS keychain) |
| `json` | Convert specific JSON/JSONL file or URL |
| `all` | Batch convert all sessions (HTML archive, DuckDB, or JSON) |
| `explore` | Launch Data Explorer web UI for star schema databases |
| `import` | Import Claude.ai account exports (Settings > Privacy > Export) |
| `schema` | Inspect JSON structure without exposing content (safe to share publicly) |

### local (default)

Two-phase interactive picker: first select project(s), then select session(s) within them. Uses rich terminal tables with metadata like model, branch, duration, and message count.

```bash
ccutils                                          # Interactive picker, opens in browser
ccutils local --format duckdb-star -o ./analytics # Star schema DuckDB
ccutils local -p myproject                       # Filter by project name
ccutils local --flat                             # Legacy single-list mode
ccutils local --include-subagents                # Include agent sessions
ccutils local --format duckdb-star --embed -o .  # With ColBERT embeddings
```

### all

Batch convert every session. Supports parallel processing and HTML archive generation with search.

```bash
ccutils all -o ./archive                         # HTML archive with search index
ccutils all --format duckdb-star -o ./analytics   # Star schema for all sessions
ccutils all --format duckdb-star --embed -o ./out # With ColBERT embeddings
ccutils all -j 4 --batch-size 20 -o ./archive    # Parallel processing
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

### explore

Visual query builder for star schema DuckDB databases. Runs a local web server.

```bash
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

### HTML Transcripts

Clean, mobile-friendly HTML with pagination, commit timeline, tool stats, and full-text search.

```bash
ccutils local -o ./transcript --open
ccutils all -o ./archive                    # Archive with master index and search
```

### DuckDB Analytics

#### Simple Schema (4 tables)

```bash
ccutils local --format duckdb -o ./archive
```

Tables: `sessions`, `messages`, `tool_calls`, `thinking`

#### Star Schema (22 tables + 8 views)

```bash
ccutils local --format duckdb-star -o ./analytics
```

Dimensional model designed for analytics:

- **6 dimensions:** sessions (with heuristic classifications), projects, tools (with categories), models (with families), dates, times
- **6 core facts:** messages, tool calls (with duration tracking), session summaries, file operations, errors (with type classification), tool chain steps
- **5 granular tables:** files (with language detection), session chains, content blocks, code blocks, entity mentions
- **3 agent/bridge tables:** agent delegations (with denormalized metrics), cross-session file tracking, task-agent mapping
- **2 optional:** ColBERT embeddings, tool input parameters
- **8 semantic views:** pre-joined views for common queries

#### Heuristic Classification

The star schema ETL runs heuristic classification during ingestion with zero external dependencies -- no LLM, no API key needed. Results are stored on `dim_session`:

| Classifier | Method | Values |
|------------|--------|--------|
| **Intent** | Score-based keyword matching on first user message | bug_fix, feature, refactor, debug, test, docs, review, explore |
| **Complexity** | Points-based scoring from session metrics | trivial, simple, moderate, complex |
| **Outcome** | Inferred from last assistant message + error rate | success, failure, unknown |
| **Domain** | Inferred from file extensions touched | web, backend, data, devops, docs, mixed, unknown |
| **Error type** | Classified from error message text (on `fact_errors`) | permission_denied, file_not_found, syntax_error, timeout, import_error, tool_error |

```sql
-- What kinds of sessions do I have?
SELECT intent, complexity, COUNT(*) as sessions
FROM dim_session GROUP BY intent, complexity ORDER BY sessions DESC;

-- Tool usage by category
SELECT dt.tool_category, COUNT(*) as uses
FROM fact_tool_calls ftc
JOIN dim_tool dt ON ftc.tool_key = dt.tool_key
GROUP BY dt.tool_category ORDER BY uses DESC;

-- Am I more productive mornings or evenings?
SELECT dti.time_of_day, COUNT(*) as sessions, AVG(fss.total_messages) as avg_msgs
FROM fact_session_summary fss
JOIN dim_time dti ON fss.time_key = dti.time_key
GROUP BY dti.time_of_day;

-- Most-touched files across all sessions
SELECT df.file_path, SUM(bsf.write_count + bsf.edit_count) as modifications
FROM bridge_session_file bsf
JOIN dim_file df ON bsf.file_key = df.file_key
GROUP BY df.file_path ORDER BY modifications DESC LIMIT 20;
```

### JSON Export

```bash
# Simple schema - single file
ccutils local --format json -o ./sessions.json

# Star schema - directory structure (dimensions/ + facts/ + meta.json)
ccutils local --format json-star -o ./star-export/
```

## Common Options

```bash
# Output
-o, --output PATH          Output directory or file
-a, --output-auto          Auto-name based on session
--open                     Open result in browser

# Format
--format FORMAT            html, duckdb, duckdb-star, json, json-star
--schema SCHEMA            simple or star (auto-inferred from format)

# Content
--include-thinking         Include thinking blocks (can be large)
--include-subagents        Include related agent sessions

# Privacy
--private                  Sanitize file paths (removes home directory, absolute paths)

# Selection (local command)
--flat                     Flat single-list mode (skip project grouping)
--expand-chains            Show individual sessions in resumed chains
-p, --project TEXT         Filter by project name

# Embeddings (local and all commands, star schema only)
--embed                    Run ColBERT embedding pipeline (requires pylate)
--embed-model TEXT         Override default ColBERT model

# Batch processing (all command)
-j, --jobs N               Parallel workers (default: 1)
--batch-size N             Sessions per transaction (default: 10)
--no-search-index          Skip search index generation
```

## Documentation

- [Star Schema Reference](docs/STAR_SCHEMA.md) -- table definitions, ETL capabilities, heuristic classification, example queries
- [Data Explorer Guide](docs/DATA_EXPLORER.md) -- visual query builder features and architecture

## Development

```bash
uv run pytest              # Run tests (~660 passing)
uv run ccutils --help      # Run development version
uv run black .             # Format code
uv run pytest --cov=ccutils  # Coverage
```

## License

Apache-2.0
