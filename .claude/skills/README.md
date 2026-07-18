# Project skills

Skills for working on and with ccutils, organized as a progressive-disclosure
hierarchy so each task loads only the context it needs:

1. **Skill descriptions** (always visible) -- routing only: which skill fires when.
2. **SKILL.md bodies** (loaded on trigger) -- the decision router for that scenario,
   plus the handful of rules that must never be missed.
3. **`references/*.md`** (read on demand) -- worked recipes and full contracts for
   one specific sub-task. Read the one the SKILL.md routes you to; skip the rest.
4. **Repo docs and source** (`docs/STAR_SCHEMA.md`, `docs/FACET_CLUSTER_PIPELINE.md`,
   module docstrings) -- ground truth. References point into these with section
   anchors instead of duplicating them; when a reference and a doc disagree, the
   doc and the code win, and the reference should be fixed.

| Skill | Scenario |
|---|---|
| `query-warehouse` | Answering questions against a ccutils DuckDB warehouse (usage, cost, tools, files, plans, agents, ETL health) |
| `etl-dev` | Extending the ETL: new facts, new columns, facets, migrations, releases |
| `render-exports` | HTML/markdown exporters, Jinja2 templates, search UI, `--private` |

Keep this drift-free: a new fact table touches `query-warehouse` routing +
recipes; a schema change touches `etl-dev` references only if the *contract*
changed (not just the column list -- that lives in `docs/STAR_SCHEMA.md`).
