---
name: render-exports
description: Work on the ccutils HTML/markdown transcript exporters — Jinja2 templates, transcript.css, the filter/search UI, or the --private and --no-thinking flags on render formats. Use for any change under src/ccutils/export/html.py, export/markdown.py, templates/, or static/. The HTML security model (autoescape, nh3 sanitization, a hash-pinned CSP) is load-bearing against XSS from untrusted transcript content; this skill states what must not be weakened and which tests hold it.
---

# HTML / markdown export development

Transcript content is **untrusted input**: session JSONL can contain
arbitrary HTML/JS pasted by users or emitted by tools.

## Two channels into HTML, guarded by different things

1. **Markdown** (`export/html.py::render_markdown_text`). Safe because the
   markdown renderer runs with raw HTML off and escapes it itself.
   `nh3.clean(raw, attributes={"code": {"class"}})` on top is defence in
   depth; the one attribute carve-out keeps fenced-code highlighting. Don't
   widen it.
2. **Template interpolation** (session id, cwd, git branch, model, tool name,
   tool input). Guarded ONLY by Jinja2 `autoescape=True`. Turning it off puts
   live `<script>` into exports.
   `tests/test_generate_html.py::test_template_interpolation_escapes_transcript_content`
   holds it. Every `|safe` in the macros is safe only because its content was
   pre-sanitized (nh3) or pre-escaped (`html.escape`); a new `|safe` needs
   you to say which.

Every document also ships a strict CSP built by `export/html.py::_build_csp`:
`default-src 'none'`, with the inline `<style>` and `<script>` blocks pinned
by sha256. Never `unsafe-inline` or `unsafe-hashes`. A hash covers a BLOCK,
so `style=` and `on*=` ATTRIBUTES stay forbidden everywhere
(`tests/test_html_css_coverage.py::TestNoInlineConstructsForCsp`). The hash
must be computed over the exact bytes emitted: autoescape once corrupted an
inline script and its hash together, silently.

## Non-obvious mechanics

- Templates are `templates/session.html`, `index.html` and `macros.html`.
  `templates/filter.js` is a **Jinja2 template** rendered via `_jinja_env`,
  not a static file: `{{ }}` in it is template syntax, and edits must survive
  rendering. `static/transcript.js` is read as text and inlined. Both are
  hashed into the CSP.
- Template variables render **empty, not error**. A typo'd variable silently
  disappears. Assert on rendered output in tests, never on "no exception".
- CSS classes referenced in templates MUST exist in `static/transcript.css`.
  Jinja2 won't warn about dangling classes; `tests/test_html_css_coverage.py`
  does.
- html/markdown are render-only formats: they skip warmup/no-summary sessions
  on purpose (curation, `parsers/discovery.py::is_curated_out`), while
  duckdb/json ingest everything. Don't "fix" that asymmetry. An agent
  transcript's task prompt is an `isMeta` user entry, so skipping isMeta
  there makes the agent summarise to nothing and get curated out.

## Tests

There are no snapshot tests (they were removed). Coverage is explicit
assertions and invariants in `tests/test_generate_html.py` and
`tests/test_html_css_coverage.py`. A template or CSS change means asserting
the rendered effect you intended.

## `--private`

- Best-effort on html/markdown only; NOT wired into the duckdb/json ETL (loud
  `UsageError` there). It masks cwd/home-prefixed paths in a subset of
  channels only. It is not a sharing guarantee: the boundary for anything
  shared is scope (which projects are in the file), never redaction.
- It fails LOUD when cwd is unresolvable instead of no-opping. The
  silent-privacy-no-op class shipped three times; never reintroduce it. Tests
  must assert the sanitized output itself (paths actually masked), not just
  flag acceptance.
- Comprehensive channel-walking plan: `internal/plans/private_hardening.md`.

## `--no-thinking`

Enforced separately on every surface, and a new render surface has to wire
it again and assert the effect. On HTML it is
`export/html.py::_strip_thinking_blocks`, which filters loglines before any
consumer sees them; on markdown it is a per-block skip in
`export/markdown.py`. It once shipped ignored entirely by HTML (flag
accepted, exit 0, output byte-identical) while a test named after the flag
stayed green because it covered the ETL tier. Thinking is real content:
Claude Code keeps the text for a share of blocks
(`docs/JSONL_CONTRACT.md` claim 5).
