---
name: render-exports
description: Work on the ccutils HTML/markdown transcript exporters — Jinja2 templates, transcript.css, the search UI, snapshot tests, or the --private privacy flag. Use for any change under src/ccutils/export/html.py, export/markdown.py, templates/, or static/. The HTML security model (nh3 sanitization, autoescape, CSP) is load-bearing against XSS from untrusted transcript content; this skill states what must not be weakened and how to keep snapshot tests honest.
---

# HTML / markdown export development

Transcript content is **untrusted input** — session JSONL can contain
arbitrary HTML/JS pasted by users or emitted by tools. The security model has
three layers; do not remove or weaken any without understanding the XSS
implications:

1. `render_markdown_text` sanitizes via
   `nh3.clean(raw, attributes={"code": {"class"}})` — the single attribute
   carve-out keeps fenced-code highlighting working. Don't widen it.
2. Jinja2 runs `autoescape=True`. Every `|safe` in the macros is safe ONLY
   because the content is pre-sanitized (nh3) or pre-escaped (`html.escape`).
   Adding a new `|safe` requires proving which of those two applies.
3. `base.html` ships a CSP meta tag blocking external scripts/iframes. Keep it.

## Non-obvious mechanics

- `search.js` / `global_search.js` are **Jinja2 templates** rendered via
  `_jinja_env`, not static files — `{{ }}` in them is template syntax, and
  edits must survive rendering.
- Template variables render **empty, not error** — a typo'd variable silently
  disappears. Assert on rendered output in tests, never on "no exception".
- CSS classes referenced in templates MUST exist in `transcript.css` —
  Jinja2 won't warn about dangling classes.
- html/markdown are render-only formats: they skip warmup/no-summary sessions
  on purpose (curated archive), while duckdb/json ingest everything. Don't
  "fix" that asymmetry.

## Snapshot tests

Any change to `transcript.css`, `macros.html`, or `base.html` requires
regenerating snapshots: `uv run pytest tests/ --confcutdir=tests --snapshot-update`,
then review the diff — a snapshot update is a claim that every pixel-level
change is intended.

## `--private`

- Best-effort on html/markdown only; NOT wired into the duckdb/json ETL (loud
  `UsageError` there). It masks cwd/home-prefixed paths in a subset of
  channels only — it is not a sharing guarantee.
- It fails LOUD when cwd is unresolvable instead of no-opping. The
  silent-privacy-no-op class shipped three times; never reintroduce it. Tests
  must assert the sanitized output itself (paths actually masked), not just
  flag acceptance.
- Comprehensive channel-walking plan: `internal/plans/private_hardening.md`.
- `--no-thinking` IS wired through the facts; raw `message_json` never
  survives staging; the Parquet lake intentionally retains everything.
