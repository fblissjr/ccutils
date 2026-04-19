"""Class-coverage tests: every CSS class emitted by the HTML exporter must have a rule.

Keystone defense against template/CSS drift. Two complementary checks:

1. **Runtime:** render a comprehensive fixture, parse the output, assert every
   `class="..."` token has a matching `.foo` selector in the inlined CSS.
2. **Static:** scan every Jinja template in src/ccutils/templates/ for literal
   class tokens and assert each has a matching selector in transcript.css.
   This catches classes emitted only by rarely-exercised templates.

Without these, classes like `.tool-error` (defined in macros.html, missing from
CSS for months) silently render unstyled. Adding any new panel -- like plan
revisions -- without this test means manual eyeballing for every new class.
"""

import re
from importlib.resources import files
from pathlib import Path

import pytest

from ccutils import generate_html


# Classes intentionally emitted without a direct CSS rule. Keep small and justified.
KNOWN_SAFE_UNSTYLED = {
    # JS-toggled state classes (styled via .truncatable.truncated / .truncatable.expanded,
    # which are caught as "truncatable" in the base-class pass, not as standalone rules):
    "expanded",
    "truncated",
    # Markdown-library-injected classes on fenced code blocks -- pygments-style.
    # Accept the family under any language; the parent <pre> is styled.
    "language-python",
    "language-bash",
    "language-sh",
    "language-javascript",
    "language-typescript",
    "language-json",
    "language-yaml",
    "language-sql",
    "language-html",
    "language-css",
    "language-rust",
    "language-go",
    "language-c",
    "language-cpp",
    "language-java",
    "language-text",
}


_CSS_CLASS_TOKEN = re.compile(r"\.([a-zA-Z_][\w-]*)")
_HTML_CLASS_ATTR = re.compile(r"""class\s*=\s*["']([^"']+)["']""")
# Match Jinja expressions that set class values including dynamic concatenation.
_JINJA_CLASS_IN_TEMPLATE = re.compile(r"""class\s*=\s*["']([^"'{}]*(?:\{\{[^}]+\}\}[^"'{}]*)*)["']""")


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


def _classes_used_in_html(html: str) -> set[str]:
    body = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    # Also strip <script> blocks so JS string literals don't count
    body = re.sub(r"<script[^>]*>.*?</script>", "", body, flags=re.DOTALL)
    used: set[str] = set()
    for match in _HTML_CLASS_ATTR.finditer(body):
        for token in match.group(1).split():
            used.add(token)
    return used


def _literal_classes_in_template(text: str) -> set[str]:
    """Extract class tokens from a Jinja template, skipping dynamic expressions.

    For class="foo {{ bar }} baz" we only collect "foo" and "baz", because
    the interpolated value is runtime-dependent.
    """
    literals: set[str] = set()
    for match in _JINJA_CLASS_IN_TEMPLATE.finditer(text):
        value = match.group(1)
        # Remove Jinja expressions -- anything between {{ and }}
        cleaned = re.sub(r"\{\{[^}]+\}\}", " ", value)
        for token in cleaned.split():
            # Skip Jinja control tokens that may have leaked in (conditionals):
            if token.startswith("{%") or token.startswith("%}"):
                continue
            literals.add(token)
    return literals


def _all_template_paths() -> list[Path]:
    templates_dir = Path(str(files("ccutils") / "templates"))
    return sorted(p for p in templates_dir.glob("*.html") if p.is_file())


def _read_css() -> str:
    return (Path(str(files("ccutils") / "static")) / "transcript.css").read_text(
        encoding="utf-8"
    )


# --- fixtures ----------------------------------------------------------------


def _render_sample(tmp_path):
    fixture = Path(__file__).parent / "sample_session.json"
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    generate_html(fixture, out, github_repo="example/project")
    return out


@pytest.fixture
def rendered_index_html(tmp_path):
    out = _render_sample(tmp_path)
    return (out / "index.html").read_text(encoding="utf-8")


@pytest.fixture
def rendered_page_html(tmp_path):
    out = _render_sample(tmp_path)
    pages = sorted(out.glob("page-*.html"))
    assert pages, "No page-*.html generated"
    return pages[0].read_text(encoding="utf-8")


# --- tests -------------------------------------------------------------------


class TestRenderedCssCoverage:
    """Runtime check: every class in the rendered sample must have a CSS rule."""

    def test_index_html_classes_have_rules(self, rendered_index_html):
        css = _extract_css_block(rendered_index_html)
        defined = _classes_defined_in_css(css)
        used = _classes_used_in_html(rendered_index_html)
        missing = used - defined - KNOWN_SAFE_UNSTYLED
        assert not missing, (
            f"Classes in index.html without CSS rules: {sorted(missing)}. "
            f"Add a rule to transcript.css or add to KNOWN_SAFE_UNSTYLED."
        )

    def test_page_html_classes_have_rules(self, rendered_page_html):
        css = _extract_css_block(rendered_page_html)
        defined = _classes_defined_in_css(css)
        used = _classes_used_in_html(rendered_page_html)
        missing = used - defined - KNOWN_SAFE_UNSTYLED
        assert not missing, (
            f"Classes in page-NNN.html without CSS rules: {sorted(missing)}."
        )


class TestTemplateStaticCssCoverage:
    """Static check: every literal class token across all .html templates has a CSS rule.

    Catches classes emitted only by rarely-exercised templates (multi_session_index,
    master_index, project_index) that the runtime fixture doesn't hit.
    """

    def test_all_template_classes_have_rules(self):
        css = _read_css()
        defined = _classes_defined_in_css(css)

        missing_by_file: dict[str, set[str]] = {}
        for tpl in _all_template_paths():
            text = tpl.read_text(encoding="utf-8")
            used = _literal_classes_in_template(text)
            missing = used - defined - KNOWN_SAFE_UNSTYLED
            if missing:
                missing_by_file[tpl.name] = missing

        assert not missing_by_file, (
            f"Templates emit class tokens with no CSS rule: {missing_by_file}. "
            f"Add rules to transcript.css, or add to KNOWN_SAFE_UNSTYLED with justification."
        )
