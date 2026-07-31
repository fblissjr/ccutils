---
name: test-schema
description: Run the 4 star schema test files (DDL, ETL, analytics, advanced) with verbose output
---

Run the star schema tests:

```bash
uv run pytest tests/test_star_schema_ddl.py tests/test_star_schema_etl.py tests/test_star_schema_analytics.py tests/test_star_schema_advanced.py -v --tb=short
```

If a specific area is mentioned, run only that file:
- "ddl" or "schema" -> test_star_schema_ddl.py
- "etl" or "pipeline" -> test_star_schema_etl.py
- "analytics" or "queries" -> test_star_schema_analytics.py
- "advanced" or "agents" or "embeddings" -> test_star_schema_advanced.py
