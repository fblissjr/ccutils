"""Class-coverage tests: every CSS class emitted by the HTML exporter must have a rule.

Keystone defense against template/CSS drift. Two complementary checks:

1. **Runtime:** render a comprehensive fixture, parse the output, assert every
   `class="..."` token has a matching `.foo` selector in the inlined CSS.
2. **Static:** scan every Jinja template in src/ccutils/templates/ for literal
   class tokens and assert each has a matching selector in transcript.css.
   This catches classes emitted only by rarely-exercised templates.

Without these, classes like `.tool-error` (defined in macros.html, missing from
CSS for months) silently render unstyled.
"""

import re
from pathlib import Path

import pytest

from ccutils import CSS, generate_html


# Classes emitted by templates but intentionally without a direct CSS rule.
# Each entry must document why, so additions are a deliberate choice.
KNOWN_SAFE_UNSTYLED = {
    # JS-toggled state; styling lives on compound selectors like
    # `.truncatable.truncated`, which the regex catches as the base class:
    "expanded",
    "truncated",
    # Modifier classes that intentionally inherit from a styled parent:
    "write-tool",       # on .file-tool
    "edit-tool",        # on .file-tool
    "write-header",     # on .file-tool-header
    "edit-header",      # on .file-tool-header
    # Wrapper divs with no current styling needs (future extension points):
    "assistant-text",
    "user-content",
    "truncatable-content",          # inner div of .truncatable
    "index-item-long-text-content", # inner of .index-item-long-text truncatable
    "index-link",                   # pagination link, currently uses default <a>
    # Todo status classes on .todo-item; only .todo-completed has styling today:
    "todo-pending",
    "todo-in-progress",
}

# Prefix allowlist: any class token starting with one of these is accepted.
# Useful for library-injected classes whose full set is open-ended.
KNOWN_SAFE_UNSTYLED_PREFIXES = (
    # markdown library injects `language-<lang>` on <pre>/<code> for fenced blocks.
    # Parent <pre> is styled; the language-* class itself has no rule and never needs one.
    "language-",
)


_CSS_CLASS_TOKEN = re.compile(r"\.([a-zA-Z_][\w-]*)")
_HTML_CLASS_ATTR = re.compile(r"""class\s*=\s*["']([^"']+)["']""")
# Match class attributes in Jinja templates; skip any that contain `{` (which
# means they interpolate a Jinja expression -- not a literal class list).
_JINJA_LITERAL_CLASS = re.compile(r"""class\s*=\s*["']([^"'{]+)["']""")


def _extract_css_block(html: str) -> str:
    match = re.search(r"<style[^>]*>(.*?)</style>", html, re.DOTALL)
    return match.group(1) if match else ""


def _classes_defined_in_css(css: str) -> set[str]:
    defined: set[str] = set()
    for chunk in css.split("{"):
        selector_part = chunk.rsplit("}", 1)[-1]
        for cls in _CSS_CLASS_TOKEN.findall(selector_part):
            defined.add(cls)
    return defined


def _filter_known_safe(classes: set[str]) -> set[str]:
    remaining = classes - KNOWN_SAFE_UNSTYLED
    return {c for c in remaining if not c.startswith(KNOWN_SAFE_UNSTYLED_PREFIXES)}


def _classes_used_in_html(html: str) -> set[str]:
    body = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    body = re.sub(r"<script[^>]*>.*?</script>", "", body, flags=re.DOTALL)
    used: set[str] = set()
    for match in _HTML_CLASS_ATTR.finditer(body):
        for token in match.group(1).split():
            used.add(token)
    return used


def _literal_classes_in_template(text: str) -> set[str]:
    """Extract class tokens from a Jinja template's static class= attributes.

    Attributes that include `{{ ... }}` interpolation are skipped entirely
    (the rendered value is runtime-dependent and caught by the runtime test).
    """
    literals: set[str] = set()
    for match in _JINJA_LITERAL_CLASS.finditer(text):
        for token in match.group(1).split():
            literals.add(token)
    return literals


@pytest.fixture(scope="class")
def rendered_sample_outputs(tmp_path_factory):
    """Render the comprehensive sample once per test class; return (index, page) HTML."""
    fixture = Path(__file__).parent / "sample_session.json"
    out = tmp_path_factory.mktemp("css_coverage_render")
    generate_html(fixture, out, github_repo="example/project")
    pages = sorted(out.glob("page-*.html"))
    assert pages, "No page-*.html generated"
    return {
        "index": (out / "index.html").read_text(encoding="utf-8"),
        "page": pages[0].read_text(encoding="utf-8"),
        "out_dir": out,
    }


class TestRenderedCssCoverage:
    """Runtime check: every class in the rendered sample must have a CSS rule."""

    @pytest.mark.parametrize("doc_name", ["index", "page"])
    def test_rendered_classes_have_rules(self, rendered_sample_outputs, doc_name):
        html = rendered_sample_outputs[doc_name]
        out_dir = rendered_sample_outputs["out_dir"]
        css_content = (out_dir / "transcript.css").read_text(encoding="utf-8")
        defined = _classes_defined_in_css(css_content)
        missing = _filter_known_safe(_classes_used_in_html(html) - defined)
        assert not missing, (
            f"Classes in {doc_name}.html without CSS rules: {sorted(missing)}. "
            f"Add a rule to transcript.css or extend KNOWN_SAFE_UNSTYLED with a justification."
        )


class TestTemplateStaticCssCoverage:
    """Static check: every literal class token across all .html templates has a CSS rule.

    Catches classes emitted only by rarely-exercised templates (multi_session_index,
    master_index, project_index) that the runtime fixture doesn't hit.
    """

    def test_all_template_classes_have_rules(self):
        defined = _classes_defined_in_css(CSS)
        templates_dir = Path(__file__).parent.parent / "src" / "ccutils" / "templates"

        missing_by_file: dict[str, set[str]] = {}
        for tpl in sorted(templates_dir.glob("*.html")):
            used = _literal_classes_in_template(tpl.read_text(encoding="utf-8"))
            missing = _filter_known_safe(used - defined)
            if missing:
                missing_by_file[tpl.name] = missing

        assert not missing_by_file, (
            f"Templates emit class tokens with no CSS rule: {missing_by_file}. "
            f"Add rules to transcript.css, or extend KNOWN_SAFE_UNSTYLED with justification."
        )
