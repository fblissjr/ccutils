---
name: test-schema
description: Run the star schema DDL test plus all per-fact/per-dimension v15 test files, with verbose output
---

The star schema tests are `tests/test_star_schema_ddl.py` (DDL/column
assertions) plus one `tests/test_<fact_or_dim>_v15.py` per populator (24
files as of this writing -- `test_fact_messages_v15.py`,
`test_dim_session_heuristics_v15.py`, `test_fact_plan_revisions_v15.py`,
etc.). There is no single monolithic "ETL" or "analytics" or "advanced"
file; each populator gets its own file.

Run everything:

```bash
uv run pytest tests/test_star_schema_ddl.py tests/test_*_v15.py -v --tb=short
```

If a specific area is mentioned, run only the matching file(s):

```bash
uv run pytest tests/test_star_schema_ddl.py -v --tb=short          # "ddl" or "schema"
uv run pytest -k "<fact_or_dim_name>" tests/test_*_v15.py -v --tb=short   # e.g. "plan_revisions", "agent_delegations", "session_heuristics"
```
