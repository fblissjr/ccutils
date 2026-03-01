# Star Schema DuckDB Implementation

Last updated: 2026-02-28

A dimensional data model for Claude Code transcript analytics with 22 tables and 8 views. Designed for questions the simple 4-table schema cannot answer: intent classification, tool duration analysis, cross-session file tracking, time-of-day patterns, tool sequence analysis, and semantic similarity clustering.

## Quick Start (CLI)

```bash
# Generate star schema DuckDB from local sessions
ccutils local --format duckdb-star -o ./analytics

# Or export to JSON directory structure
ccutils local --format json-star -o ./star-export/

# Or generate from all sessions
ccutils all --format duckdb-star -o ./analytics

# Launch the visual Data Explorer
ccutils explore ./analytics/archive.duckdb

# Query directly with DuckDB CLI
duckdb ./analytics/archive.duckdb
```

Once in DuckDB:

```sql
-- What kinds of sessions do I have?
SELECT intent, complexity, COUNT(*) as sessions
FROM dim_session
GROUP BY intent, complexity ORDER BY sessions DESC;

-- Which tools are slowest?
SELECT dt.tool_name, AVG(ftc.duration_seconds) as avg_sec
FROM fact_tool_calls ftc
JOIN dim_tool dt ON ftc.tool_key = dt.tool_key
WHERE ftc.duration_seconds IS NOT NULL
GROUP BY dt.tool_name ORDER BY avg_sec DESC;

-- Most-touched files across all sessions
SELECT df.file_path, SUM(bsf.write_count + bsf.edit_count) as modifications
FROM bridge_session_file bsf
JOIN dim_file df ON bsf.file_key = df.file_key
GROUP BY df.file_path ORDER BY modifications DESC LIMIT 20;

-- Am I more productive mornings or evenings?
SELECT dti.time_of_day, COUNT(*) as sessions, AVG(fss.total_messages) as avg_msgs
FROM fact_session_summary fss
JOIN dim_time dti ON fss.time_key = dti.time_key
GROUP BY dti.time_of_day;
```

See [DATA_EXPLORER.md](DATA_EXPLORER.md) for the visual interface.

---

## Overview

The star schema follows dimensional modeling best practices:

- **Dimension tables** contain descriptive attributes (who, what, when, where)
- **Fact tables** contain measurable events and metrics
- **Hash-based surrogate keys** link facts to dimensions (no hard PK/FK constraints)
- **Degenerate dimensions** for low-cardinality categoricals (message_type, block_type, error_type, entity_type, language) -- stored directly on fact tables instead of in separate lookup tables
- **Heuristic classification** runs during ETL with zero external dependencies (no LLM, no API key)

## Schema Summary

| Category | Count | Tables |
|----------|-------|--------|
| Core dimensions | 6 | dim_session, dim_project, dim_tool, dim_model, dim_date, dim_time |
| Core facts | 6 | fact_messages, fact_tool_calls, fact_session_summary, fact_file_operations, fact_errors, fact_tool_chain_steps |
| Granular dimensions | 2 | dim_file, dim_session_chain |
| Granular facts | 3 | fact_content_blocks, fact_code_blocks, fact_entity_mentions |
| Agent/bridge/staging | 3 | fact_agent_delegations, bridge_session_file, stg_task_agent_map |
| Optional (require pylate) | 2 | fact_session_embeddings, fact_tool_input_params |
| **Total** | **22 tables** | + 8 semantic views |

---

## Tables

### Core Dimensions

#### dim_session
One row per Claude Code session.

| Column | Type | Description |
|--------|------|-------------|
| session_key | VARCHAR | PK (hash of session_id) |
| session_id | VARCHAR | Original session UUID |
| project_key | VARCHAR | FK to dim_project |
| chain_key | VARCHAR | FK to dim_session_chain |
| slug | VARCHAR | Session slug for chain grouping |
| cwd | VARCHAR | Working directory |
| is_agent | BOOLEAN | Whether this is an agent session |
| agent_id | VARCHAR | Agent identifier |
| parent_session_id | VARCHAR | Parent session UUID if agent |
| depth_level | INTEGER | Nesting depth (0=root) |
| intent | VARCHAR | Heuristic: bug_fix, feature, refactor, debug, test, docs, review, explore |
| complexity | VARCHAR | Heuristic: trivial, simple, moderate, complex |
| outcome | VARCHAR | Heuristic: success, failure, unknown |
| domain | VARCHAR | Heuristic: web, backend, data, devops, docs, mixed, unknown |

#### dim_project
One row per project (working directory).

| Column | Type | Description |
|--------|------|-------------|
| project_key | VARCHAR | PK (hash of project_path) |
| project_path | VARCHAR | Full path |
| project_name | VARCHAR | Directory name |

#### dim_tool
One row per tool name.

| Column | Type | Description |
|--------|------|-------------|
| tool_key | VARCHAR | PK (hash of tool_name) |
| tool_name | VARCHAR | Tool name (e.g., "Bash", "Read") |
| tool_category | VARCHAR | Category from TOOL_CATEGORIES lookup |

#### dim_model
One row per model string.

| Column | Type | Description |
|--------|------|-------------|
| model_key | VARCHAR | PK (hash of model_id) |
| model_id | VARCHAR | Full model identifier |
| model_family | VARCHAR | Family: opus, sonnet, haiku |

#### dim_date
One row per calendar date.

| Column | Type | Description |
|--------|------|-------------|
| date_key | VARCHAR | PK (hash of date) |
| full_date | DATE | Calendar date |
| year | INTEGER | Year |
| month | INTEGER | Month (1-12) |
| day | INTEGER | Day of month |
| day_of_week | VARCHAR | Day name |
| is_weekend | BOOLEAN | Weekend flag |
| week_of_year | INTEGER | ISO week number |

#### dim_time
One row per hour:minute.

| Column | Type | Description |
|--------|------|-------------|
| time_key | VARCHAR | PK (hash of time) |
| hour | INTEGER | Hour (0-23) |
| minute | INTEGER | Minute (0-59) |
| time_of_day | VARCHAR | morning, afternoon, evening, night |

### Core Facts

#### fact_messages
One row per message.

| Column | Type | Description |
|--------|------|-------------|
| message_key | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| model_key | VARCHAR | FK to dim_model |
| date_key | VARCHAR | FK to dim_date |
| time_key | VARCHAR | FK to dim_time |
| message_type | VARCHAR | Degenerate: user, assistant, system |
| message_index | INTEGER | Position in conversation |
| conversation_depth | INTEGER | Turn count at this point |
| content_text | VARCHAR | Message text content |
| has_tool_use | BOOLEAN | Contains tool calls |
| has_thinking | BOOLEAN | Contains thinking blocks |
| is_sidechain | BOOLEAN | Part of agent sidechain |
| timestamp | TIMESTAMP | Message timestamp |
| cache_creation_input_tokens | INTEGER | Token count (cache create) |
| cache_read_input_tokens | INTEGER | Token count (cache read) |
| input_tokens | INTEGER | Input token count |
| output_tokens | INTEGER | Output token count |

#### fact_tool_calls
One row per tool invocation.

| Column | Type | Description |
|--------|------|-------------|
| tool_call_key | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| tool_key | VARCHAR | FK to dim_tool |
| date_key | VARCHAR | FK to dim_date |
| time_key | VARCHAR | FK to dim_time |
| tool_use_id | VARCHAR | Tool use identifier |
| tool_input | VARCHAR | Raw JSON input |
| tool_result | VARCHAR | Result text |
| is_error | BOOLEAN | Whether the call errored |
| error_message | VARCHAR | Error text if failed |
| duration_seconds | FLOAT | Time between invoke and result |
| file_path | VARCHAR | Extracted file path (if applicable) |
| command | VARCHAR | Extracted command (if applicable) |
| pattern | VARCHAR | Extracted pattern (if applicable) |
| query_text | VARCHAR | Extracted query (if applicable) |
| timestamp | TIMESTAMP | Invocation timestamp |

#### fact_session_summary
One row per session with pre-aggregated metrics.

| Column | Type | Description |
|--------|------|-------------|
| summary_key | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| project_key | VARCHAR | FK to dim_project |
| date_key | VARCHAR | FK to dim_date |
| time_key | VARCHAR | FK to dim_time |
| total_messages | INTEGER | Total message count |
| user_messages | INTEGER | User message count |
| assistant_messages | INTEGER | Assistant message count |
| total_tool_calls | INTEGER | Tool call count |
| total_input_tokens | BIGINT | Total input tokens |
| total_output_tokens | BIGINT | Total output tokens |
| total_errors | INTEGER | Error count |
| unique_tools_used | INTEGER | Distinct tools used |
| unique_files_touched | INTEGER | Distinct files touched |
| max_conversation_depth | INTEGER | Maximum conversation depth |
| total_estimated_tokens | BIGINT | Estimated total tokens |
| duration_seconds | FLOAT | Session duration |
| start_time | TIMESTAMP | First message time |
| end_time | TIMESTAMP | Last message time |

#### fact_file_operations
One row per file touch.

| Column | Type | Description |
|--------|------|-------------|
| operation_key | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| file_key | VARCHAR | FK to dim_file |
| tool_key | VARCHAR | FK to dim_tool |
| operation_type | VARCHAR | Degenerate: read, write, edit |
| timestamp | TIMESTAMP | Operation timestamp |

#### fact_errors
One row per error occurrence.

| Column | Type | Description |
|--------|------|-------------|
| error_key | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| tool_key | VARCHAR | FK to dim_tool |
| error_type | VARCHAR | Heuristic classification (permission_denied, file_not_found, syntax_error, timeout, import_error, tool_error) |
| error_message | VARCHAR | Error text |
| timestamp | TIMESTAMP | Error timestamp |

#### fact_tool_chain_steps
One row per step in a tool invocation chain.

| Column | Type | Description |
|--------|------|-------------|
| chain_step_key | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| tool_key | VARCHAR | FK to dim_tool |
| chain_position | INTEGER | Position in chain |
| tool_use_id | VARCHAR | Tool use identifier |
| next_tool_key | VARCHAR | FK to dim_tool for next step |
| is_error | BOOLEAN | Whether this step errored |
| timestamp | TIMESTAMP | Step timestamp |

### Granular Dimensions

#### dim_file
One row per file path.

| Column | Type | Description |
|--------|------|-------------|
| file_key | VARCHAR | PK (hash of file_path) |
| file_path | VARCHAR | Full file path |
| file_name | VARCHAR | File name only |
| file_extension | VARCHAR | Extension (e.g., ".py") |
| directory | VARCHAR | Parent directory |
| language | VARCHAR | Inferred programming language |

#### dim_session_chain
One row per chain (group of resumed sessions sharing a slug).

| Column | Type | Description |
|--------|------|-------------|
| chain_key | VARCHAR | PK (hash of slug) |
| slug | VARCHAR | Shared slug |
| session_count | INTEGER | Number of sessions in chain |
| first_session_id | VARCHAR | Earliest session |
| last_session_id | VARCHAR | Latest session |

### Granular Facts

#### fact_content_blocks
One row per content block within a message.

| Column | Type | Description |
|--------|------|-------------|
| content_block_key | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| message_key | VARCHAR | FK to fact_messages |
| block_type | VARCHAR | Degenerate: text, tool_use, tool_result, thinking, image |
| block_index | INTEGER | Position within message |
| content_text | VARCHAR | Block text |
| timestamp | TIMESTAMP | Block timestamp |

#### fact_code_blocks
One row per code block extracted from messages.

| Column | Type | Description |
|--------|------|-------------|
| code_block_key | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| language | VARCHAR | Degenerate: python, javascript, etc. |
| code_text | VARCHAR | Code content |
| line_count | INTEGER | Number of lines |
| source_context | VARCHAR | Where the code appeared |
| timestamp | TIMESTAMP | Extraction timestamp |

#### fact_entity_mentions
One row per entity mention (file, function, class, etc.).

| Column | Type | Description |
|--------|------|-------------|
| mention_key | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| entity_type | VARCHAR | Degenerate: file, function, class, variable, module, url, error, command |
| entity_name | VARCHAR | Entity identifier |
| mention_count | INTEGER | Occurrences |
| timestamp | TIMESTAMP | First mention timestamp |

### Agent/Bridge/Staging

#### fact_agent_delegations
One row per agent delegation (Task tool invocation).

| Column | Type | Description |
|--------|------|-------------|
| delegation_key | VARCHAR | PK |
| parent_session_key | VARCHAR | FK to dim_session (parent) |
| agent_session_key | VARCHAR | FK to dim_session (agent) |
| tool_use_id | VARCHAR | Task tool_use_id |
| subagent_type | VARCHAR | Agent type |
| task_description | VARCHAR | Task description |
| task_prompt | VARCHAR | Task prompt text |
| match_confidence | FLOAT | Link confidence (1.0=deterministic, 0.5-0.8=heuristic) |
| agent_tool_calls | INTEGER | Denormalized: tool calls in agent session |
| agent_errors | INTEGER | Denormalized: errors in agent session |
| agent_duration_seconds | FLOAT | Denormalized: agent session duration |
| timestamp | TIMESTAMP | Delegation timestamp |

#### bridge_session_file
One row per (session, file) pair for cross-session file tracking.

| Column | Type | Description |
|--------|------|-------------|
| bridge_key | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| file_key | VARCHAR | FK to dim_file |
| read_count | INTEGER | Read operations |
| write_count | INTEGER | Write operations |
| edit_count | INTEGER | Edit operations |

#### stg_task_agent_map
Staging table for deterministic agent delegation linking from progress records.

| Column | Type | Description |
|--------|------|-------------|
| tool_use_id | VARCHAR | Task tool_use_id |
| agent_session_id | VARCHAR | Agent session UUID |
| session_key | VARCHAR | FK to dim_session (parent) |

### Optional (require pylate)

#### fact_session_embeddings
ColBERT embeddings for semantic similarity. Requires `pip install ccutils[colbert]`.

| Column | Type | Description |
|--------|------|-------------|
| embedding_key | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| content_type | VARCHAR | What was embedded (default: first_user_message) |
| embedding_model | VARCHAR | Model used |
| embedding_dim | INTEGER | Embedding dimensions |
| mean_embedding | FLOAT[] | Mean-pooled embedding vector |
| embedded_at | TIMESTAMP | When embedding was computed |
| content_hash | VARCHAR | MD5 of embedded content |

#### fact_tool_input_params
Key-value extraction of tool input parameters.

| Column | Type | Description |
|--------|------|-------------|
| param_key | VARCHAR | PK |
| tool_call_key | VARCHAR | FK to fact_tool_calls |
| session_key | VARCHAR | FK to dim_session |
| param_name | VARCHAR | Parameter name |
| param_value | VARCHAR | Parameter value |

---

## Semantic Views

All views use the `semantic_` prefix and join facts with dimensions for easy querying.

### semantic_sessions
Sessions enriched with project info, summary metrics, and heuristic classifications.

```sql
SELECT * FROM semantic_sessions WHERE intent = 'bug_fix' AND complexity = 'complex';
```

### semantic_messages
Messages with session, model, and time context.

```sql
SELECT * FROM semantic_messages WHERE model_family = 'opus' AND time_of_day = 'night';
```

### semantic_tool_calls
Tool calls with tool info, session context, and duration.

```sql
SELECT * FROM semantic_tool_calls WHERE tool_category = 'file_edit' ORDER BY duration_seconds DESC;
```

### semantic_file_operations
File operations with file info, tool, and session context.

```sql
SELECT * FROM semantic_file_operations WHERE language = 'python';
```

### semantic_session_chains
Session chains with aggregate metrics across all member sessions.

```sql
SELECT * FROM semantic_session_chains WHERE session_count > 3;
```

### semantic_agent_delegations
Agent delegations with parent/agent session details and denormalized metrics.

```sql
SELECT * FROM semantic_agent_delegations WHERE agent_errors > 0;
```

### semantic_file_evolution
Cross-session file activity aggregation.

```sql
SELECT * FROM semantic_file_evolution WHERE sessions_touched > 5 ORDER BY total_modifications DESC;
```

### semantic_tool_patterns
Common tool sequences with frequency and error rates.

```sql
SELECT * FROM semantic_tool_patterns WHERE error_rate > 0.3 ORDER BY frequency DESC;
```

---

## Heuristic Classification

The star schema runs heuristic classification during ETL with zero external dependencies. Results are stored on `dim_session`.

### Intent
Classified from the first user message using score-based keyword matching. Each intent's keywords are checked against the message; the intent with the most keyword hits wins. On ties, priority order is used as tiebreaker. This correctly handles compound messages like "implement new error handling" (feature, not bug_fix).

| Intent | Triggers |
|--------|----------|
| bug_fix | fix, bug, broken, error, crash, wrong, issue, failing |
| debug | debug, investigate, why, trace, diagnose |
| test | test, spec, coverage |
| docs | doc, readme, comment, explain, documentation |
| review | review, check, audit, look at |
| feature | add, new, feature, implement, create, build |
| refactor | refactor, clean, reorganize, restructure, simplify |
| explore | (fallback) |

### Complexity
Scored from session metrics:

| Factor | Condition | Points |
|--------|-----------|--------|
| Tool calls | > 20 | +2 |
| Tool calls | > 2 | +1 |
| Messages | > 30 | +2 |
| Messages | > 8 | +1 |
| Agent depth | > 0 | +2 |
| Errors | > 3 | +1 |

| Score | Classification |
|-------|----------------|
| >= 5 | complex |
| >= 3 | moderate |
| >= 1 | simple |
| 0 | trivial |

### Outcome
Inferred from the last assistant message and error rate:

- **success**: Last message contains "done", "completed", "fixed", "finished", etc.
- **failure**: Last message contains "error", "failed", "couldn't", or error rate > 50%
- **unknown**: (fallback)

### Domain
Inferred from file extensions touched during the session:

| Domain | Extensions |
|--------|-----------|
| web | .tsx, .jsx, .css, .scss, .html, .vue, .svelte |
| backend | .py, .rs, .go, .java, .rb |
| data | .sql, .parquet, .csv |
| devops | .yaml, .yml, .tf, .dockerfile, .sh |
| docs | .md, .rst, .txt |
| mixed | (tie between domains) |
| unknown | (no files or unrecognized extensions) |

### Error Type
Classified from error message text on `fact_errors`:

| Type | Triggers |
|------|----------|
| permission_denied | "permission denied", "EACCES" |
| file_not_found | "not found", "ENOENT", "no such file" |
| syntax_error | "syntax error", "SyntaxError" |
| timeout | "timeout", "ETIMEDOUT" |
| import_error | "ImportError", "ModuleNotFoundError" |
| tool_error | (fallback) |

---

## What This Enables (vs Simple Schema)

| Question | Star Schema | Simple Schema |
|----------|-------------|---------------|
| What kinds of sessions do I have? | `SELECT intent, COUNT(*) FROM dim_session GROUP BY intent` | Not possible |
| Which tools are slowest? | `AVG(duration_seconds)` on fact_tool_calls | No duration tracking |
| Most-touched file across sessions? | bridge_session_file aggregation | Per-session only |
| Productive times of day? | dim_time join + fact_session_summary | No time dimension |
| Tool sequences that lead to errors? | fact_tool_chain_steps with next_tool_key + is_error | No chain tracking |
| Find similar sessions? | fact_session_embeddings with pylate | No embeddings |

---

## JSON Export Structure

When using `--format json-star`, the output is a directory:

```
output_dir/
  meta.json              # Schema metadata, version "2.0", relationships
  dimensions/            # One JSON file per dimension table
    dim_session.json
    dim_project.json
    ...
  facts/                 # One JSON file per fact table
    fact_messages.json
    fact_tool_calls.json
    ...
```

---

## Embedding Pipeline (Optional)

Requires the `colbert` extra: `pip install ccutils[colbert]`

```python
from ccutils import EmbeddingPipeline

pipeline = EmbeddingPipeline()  # lazy model loading (mxbai-edge-colbert-v0-32m)
pipeline.embed_sessions(conn)   # embed first user message per session
pipeline.match_delegations(conn)  # re-score agent delegation confidence
pipeline.cluster_sessions(conn)   # cluster sessions by similarity -> dim_session.domain
```

CLI integration:

```bash
ccutils all --format duckdb-star --embed -o ./analytics
```
