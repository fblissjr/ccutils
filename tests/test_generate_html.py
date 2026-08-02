"""Tests for HTML generation from Claude Code session JSON."""

import json
import re
import tempfile
from pathlib import Path

import pytest

from ccutils import (
    generate_html,
    detect_github_repo,
    render_markdown_text,
    format_json,
    is_json_like,
    render_todo_write,
    render_write_tool,
    render_edit_tool,
    render_bash_tool,
    render_content_block,
    analyze_conversation,
    format_tool_stats,
    is_tool_result_message,
    parse_session_file,
    get_session_summary,
    find_local_sessions,
    extract_session_slug,
)


def read_transcript(out_dir):
    """Read the single self-contained transcript a session renders to.

    Since C2 there is exactly one .html file per session -- no index.html, no
    page-NNN.html. Tests read through this helper so the next layout change
    touches one function instead of two dozen assertions.
    """
    files = sorted(Path(out_dir).glob("*.html"))
    assert len(files) == 1, f"expected exactly one transcript, got {[f.name for f in files]}"
    return files[0].read_text(encoding="utf-8")


@pytest.fixture
def sample_session():
    """Load the sample session fixture."""
    fixture_path = Path(__file__).parent / "sample_session.json"
    with open(fixture_path) as f:
        return json.load(f)


@pytest.fixture
def output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def _reset_github_repo():
    """Prevent _github_repo state from bleeding between tests in either direction."""
    from ccutils import set_github_repo

    set_github_repo(None)
    yield
    set_github_repo(None)


class TestGenerateHtml:
    """Tests for the main generate_html function."""

    def test_generates_index_html(self, output_dir):
        """Test index.html generation."""
        fixture_path = Path(__file__).parent / "sample_session.json"
        generate_html(fixture_path, output_dir, github_repo="example/project")

        index_html = read_transcript(output_dir)
        assert "<!DOCTYPE html>" in index_html
        assert "Content-Security-Policy" in index_html
        assert 'class="prompt-list"' in index_html

    def test_csp_header_in_generated_html(self, output_dir):
        """Test that Content-Security-Policy meta tag is present in generated HTML."""
        fixture_path = Path(__file__).parent / "sample_session.json"
        generate_html(fixture_path, output_dir, github_repo="example/project")

        for html_content in [read_transcript(output_dir)]:
            assert "Content-Security-Policy" in html_content, (
                f"CSP meta tag missing from {html_file}"
            )
            assert "script-src" in html_content
            assert "frame-src 'none'" in html_content

    def test_generates_page_001_html(self, output_dir):
        """Test page-001.html generation."""
        fixture_path = Path(__file__).parent / "sample_session.json"
        generate_html(fixture_path, output_dir, github_repo="example/project")

        page_html = read_transcript(output_dir)
        assert "<!DOCTYPE html>" in page_html
        assert 'class="message' in page_html
        # Stable anchor: bookmarked message links must survive any layout change.
        assert 'id="msg-' in page_html

    def test_generates_page_002_html(self, output_dir):
        """Test page-002.html generation (continuation page)."""
        fixture_path = Path(__file__).parent / "sample_session.json"
        generate_html(fixture_path, output_dir, github_repo="example/project")

        page_html = read_transcript(output_dir)
        assert "<!DOCTYPE html>" in page_html
        assert 'id="msg-' in page_html

    def test_github_repo_autodetect(self, sample_session):
        """Test GitHub repo auto-detection from git push output."""
        loglines = sample_session["loglines"]
        repo = detect_github_repo(loglines)
        assert repo == "example/project"

    def test_github_repo_does_not_leak_between_calls(self, tmp_path):
        """generate_html() must restore _github_repo to its pre-call value.

        Previously the module-level _github_repo was set but never reset, so a
        second call without an explicit github_repo would inherit the first
        call's value. The _reset_github_repo autouse fixture guarantees the
        starting state.
        """
        from ccutils import get_github_repo

        jsonl_file = tmp_path / "session.jsonl"
        jsonl_file.write_text(
            '{"type":"user","message":{"role":"user","content":"hi"}}\n'
        )

        out1 = tmp_path / "out1"
        out1.mkdir()
        generate_html(jsonl_file, out1, github_repo="owner/first")
        assert get_github_repo() is None, "github_repo leaked across generate_html calls"

        out2 = tmp_path / "out2"
        out2.mkdir()
        generate_html(jsonl_file, out2)
        assert get_github_repo() is None

    def test_handles_array_content_format(self, tmp_path):
        """Test that user messages with array content format are recognized.

        Claude Code v2.0.76+ uses array content format like:
        {"type": "user", "message": {"content": [{"type": "text", "text": "..."}]}}
        instead of the simpler string format:
        {"type": "user", "message": {"content": "..."}}
        """
        jsonl_file = tmp_path / "session.jsonl"
        jsonl_file.write_text(
            '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"Hello from array format"}]}}\n'
            '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hi there!"}]}}\n'
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        generate_html(jsonl_file, output_dir)

        index_html = read_transcript(output_dir)
        # Should have 1 prompt, not 0
        assert "1 prompts" in index_html or "1 prompt" in index_html
        assert "0 prompts" not in index_html
        # The page file should exist
        assert len(list(output_dir.glob("*.html"))) == 1


class TestRenderFunctions:
    """Tests for individual render functions."""

    def test_render_markdown_text(self):
        """Test markdown rendering."""
        result = render_markdown_text("**bold** and `code`\n\n- item 1\n- item 2")
        assert "<strong>bold</strong>" in result
        assert "<code>code</code>" in result
        assert "<li>item 1</li>" in result
    def test_render_markdown_text_empty(self):
        """Test markdown rendering with empty input."""
        assert render_markdown_text("") == ""
        assert render_markdown_text(None) == ""

    def test_render_markdown_text_neutralises_script_tags(self):
        """Raw `<script>` must never reach the output as live markup.

        Claim: the old version of this test asserted `"alert(" not in result`,
        which encoded STRIP semantics -- it passed only because the renderer
        deleted the tag and its body outright. That is wrong for an archive: a
        prompt discussing `<script>` should still show what was written. The
        durable claim is "never markup", not "never present".
        """
        result = render_markdown_text("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_render_markdown_text_neutralises_event_handlers(self):
        """An `onerror` attribute must never appear as markup.

        Claim: escaped, `onerror` is inert text and SHOULD be visible. A test
        asserting the substring is absent fails on correct behaviour -- delete
        this and the next rewrite of `render_markdown_text` loses its only
        statement of what "safe" means here.
        """
        result = render_markdown_text('<img src=x onerror="alert(1)">')
        assert "<img" not in result
        assert "&lt;img" in result

    def test_render_markdown_text_preserves_script_body_as_text(self):
        """Transcript content inside a `<script>` survives as visible text.

        Claim: this is the archive-fidelity requirement. Under the previous
        strip-based renderer `Try <script>console.log(1)</script> here`
        rendered as `Try  here` -- the body silently deleted. Delete this test
        and that regression is invisible.
        """
        result = render_markdown_text("Try <script>console.log(1)</script> here")
        assert "console.log(1)" in result

    def test_render_markdown_text_preserves_block_html_as_text(self):
        """Block-level HTML in prose stays text and does not split the paragraph.

        Claim: the strip-based renderer turned one paragraph into
        `<p>Wrap it in </p><div> for layout<p></p></div>` -- text lost, structure
        mangled. One paragraph in, one paragraph out.
        """
        result = render_markdown_text('Wrap it in <div class="panel"> for layout')
        assert "panel" in result
        assert result.count("<p>") == 1

    def test_render_markdown_text_escapes_code_fence_once(self):
        """Fenced code is escaped exactly once.

        Claim: guards against the double-escaping trap -- pre-escaping input and
        then letting the markdown renderer escape again yields `a &amp;lt; b`,
        which displays literally as `a &lt; b`. Deleting this test removes the
        only check that the escaping happens in one place.
        """
        result = render_markdown_text("```python\nif a < b and c & d:\n```")
        assert "a &lt; b" in result
        assert "&amp;lt;" not in result

    def test_render_markdown_text_strips_iframe(self):
        """Test that iframe tags are stripped from markdown output."""
        result = render_markdown_text('<iframe src="https://evil.com"></iframe>')
        assert "<iframe" not in result

    def test_render_markdown_text_preserves_safe_html(self):
        """Test that safe markdown-generated HTML is preserved."""
        result = render_markdown_text("**bold** and `code`")
        assert "<strong>bold</strong>" in result
        assert "<code>code</code>" in result

    def test_render_markdown_text_preserves_code_blocks(self):
        """Test that fenced code blocks are preserved with language class."""
        result = render_markdown_text("```python\nprint('hello')\n```")
        assert "<code" in result
        assert "print" in result
        assert 'class="language-python"' in result

    def test_format_json(self):
        """Test JSON formatting."""
        result = format_json({"key": "value", "number": 42, "nested": {"a": 1}})
        assert '<pre class="json">' in result
        # Escaped, not raw -- format_json output is inserted with |safe.
        assert "&quot;key&quot;" in result
    def test_is_json_like(self):
        """Test JSON-like string detection."""
        assert is_json_like('{"key": "value"}')
        assert is_json_like("[1, 2, 3]")
        assert not is_json_like("plain text")
        assert not is_json_like("")
        assert not is_json_like(None)

    def test_render_todo_write(self):
        """Test TodoWrite rendering."""
        tool_input = {
            "todos": [
                {"content": "First task", "status": "completed", "activeForm": "First"},
                {
                    "content": "Second task",
                    "status": "in_progress",
                    "activeForm": "Second",
                },
                {"content": "Third task", "status": "pending", "activeForm": "Third"},
            ]
        }
        result = render_todo_write(tool_input, "tool-123")
        assert 'class="todo-list"' in result
        assert 'data-tool-id="tool-123"' in result
        assert "todo-completed" in result and "todo-in-progress" in result
        assert "First task" in result
    def test_render_todo_write_empty(self):
        """Test TodoWrite with no todos."""
        result = render_todo_write({"todos": []}, "tool-123")
        assert result == ""

    def test_render_write_tool(self):
        """Test Write tool rendering."""
        tool_input = {
            "file_path": "/project/src/main.py",
            "content": "def hello():\n    print('hello world')\n",
        }
        result = render_write_tool(tool_input, "tool-123")
        assert "write-tool" in result
        assert "/project/src/main.py" in result
    def test_render_edit_tool(self):
        """Test Edit tool rendering."""
        tool_input = {
            "file_path": "/project/file.py",
            "old_string": "old code here",
            "new_string": "new code here",
        }
        result = render_edit_tool(tool_input, "tool-123")
        assert "edit-tool" in result
        assert "/project/file.py" in result
        assert "replace all" not in result

    def test_render_edit_tool_replace_all(self):
        """Test Edit tool with replace_all flag."""
        tool_input = {
            "file_path": "/project/file.py",
            "old_string": "old",
            "new_string": "new",
            "replace_all": True,
        }
        result = render_edit_tool(tool_input, "tool-123")
        assert "edit-replace-all" in result
        assert "(replace all)" in result

    def test_render_bash_tool(self):
        """Test Bash tool rendering."""
        tool_input = {
            "command": "pytest tests/ -v",
            "description": "Run tests with verbose output",
        }
        result = render_bash_tool(tool_input, "tool-123")
        assert "bash-tool" in result
        assert "pytest tests/ -v" in result
        assert "Run tests with verbose output" in result
class TestRenderContentBlock:
    """Tests for render_content_block function."""

    def test_image_block(self):
        """Test image block rendering with base64 data URL."""
        # 200x200 black GIF - minimal valid GIF with black pixels
        # Generated with: from PIL import Image; img = Image.new('RGB', (200, 200), (0, 0, 0)); img.save('black.gif')
        import base64
        import io

        # Create a minimal 200x200 black GIF using raw bytes
        # GIF89a header + logical screen descriptor + global color table + image data
        gif_data = (
            b"GIF89a"  # Header
            b"\xc8\x00\xc8\x00"  # Width 200, Height 200
            b"\x80"  # Global color table flag (1 color: 2^(0+1)=2 colors)
            b"\x00"  # Background color index
            b"\x00"  # Pixel aspect ratio
            b"\x00\x00\x00"  # Color 0: black
            b"\x00\x00\x00"  # Color 1: black (padding)
            b","  # Image separator
            b"\x00\x00\x00\x00"  # Left, Top
            b"\xc8\x00\xc8\x00"  # Width 200, Height 200
            b"\x00"  # No local color table
            b"\x08"  # LZW minimum code size
            b"\x02\x04\x01\x00"  # Compressed data (minimal)
            b";"  # GIF trailer
        )
        black_gif_base64 = base64.b64encode(gif_data).decode("ascii")

        block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/gif",
                "data": black_gif_base64,
            },
        }
        result = render_content_block(block)
        # The result should contain an img tag with data URL
        assert 'src="data:image/gif;base64,' in result
        # Image sizing lives in CSS (.image-block img) now, not an inline
        # style= attr -- the tightened CSP forbids 'unsafe-inline' styles.
        assert 'class="image-block"' in result
    def test_thinking_block(self):
        """Test thinking block rendering."""
        block = {
            "type": "thinking",
            "thinking": "Let me think about this...\n\n1. First consideration\n2. Second point",
        }
        result = render_content_block(block)
        assert 'class="thinking"' in result
        assert "Thinking" in result
        # CommonMark lets an ordered list interrupt a paragraph; the old
        # Python-Markdown renderer left these as run-on text.
        assert "<ol>" in result and "First consideration" in result

    def test_text_block(self):
        """Test text block rendering."""
        block = {"type": "text", "text": "Here is my response with **markdown**."}
        result = render_content_block(block)
        assert 'class="assistant-text"' in result
        assert "<strong>markdown</strong>" in result

    def test_tool_result_block(self):
        """Test tool result rendering."""
        block = {
            "type": "tool_result",
            "content": "Command completed successfully\nOutput line 1\nOutput line 2",
            "is_error": False,
        }
        result = render_content_block(block)
        assert 'class="tool-result"' in result
        assert "Command completed successfully" in result
        assert "tool-error" not in result

    def test_tool_result_error(self):
        """Test tool result error rendering."""
        block = {
            "type": "tool_result",
            "content": "Error: file not found\nTraceback follows...",
            "is_error": True,
        }
        result = render_content_block(block)
        assert "tool-error" in result
        assert "Error: file not found" in result

    def test_tool_result_with_commit(self):
        """Test tool result with git commit output."""
        # Need to set the global github_repo for commit link rendering
        from ccutils import get_github_repo, set_github_repo

        old_repo = get_github_repo()
        set_github_repo("example/repo")
        try:
            block = {
                "type": "tool_result",
                "content": "[main abc1234] Add new feature\n 2 files changed, 10 insertions(+)",
                "is_error": False,
            }
            result = render_content_block(block)
        finally:
            set_github_repo(old_repo)

        assert "abc1234" in result
        # github_repo is set, so the commit hash becomes a link.
        assert "example/repo" in result

    def test_tool_result_with_image(self):
        """Test tool result containing image blocks in content array.

        This tests the case where a tool (like a screenshot tool) returns
        both text and image content in the same tool_result.
        """
        import base64

        # Create a minimal GIF image
        gif_data = (
            b"GIF89a"  # Header
            b"\xc8\x00\xc8\x00"  # Width 200, Height 200
            b"\x80"  # Global color table flag
            b"\x00"  # Background color index
            b"\x00"  # Pixel aspect ratio
            b"\x00\x00\x00"  # Color 0: black
            b"\x00\x00\x00"  # Color 1: black
            b","  # Image separator
            b"\x00\x00\x00\x00"  # Left, Top
            b"\xc8\x00\xc8\x00"  # Width 200, Height 200
            b"\x00"  # No local color table
            b"\x08"  # LZW minimum code size
            b"\x02\x04\x01\x00"  # Compressed data
            b";"  # GIF trailer
        )
        gif_base64 = base64.b64encode(gif_data).decode("ascii")

        block = {
            "type": "tool_result",
            "content": [
                {
                    "type": "text",
                    "text": "Successfully captured screenshot (807x782, jpeg) - ID: ss_123",
                },
                {
                    "type": "text",
                    "text": "\n\nTab Context:\n- Executed on tabId: 12345",
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/gif",
                        "data": gif_base64,
                    },
                },
            ],
            "is_error": False,
        }
        result = render_content_block(block)

        # The result should contain the text content
        assert "Successfully captured screenshot" in result
        assert "Tab Context" in result

        # The result should contain an img tag with data URL for the image
        assert 'src="data:image/gif;base64,' in result
        # Image sizing lives in CSS (.image-block img) now, not an inline
        # style= attr -- the tightened CSP forbids 'unsafe-inline' styles.
        assert 'class="image-block"' in result

        # Tool results with images should NOT be truncatable
        assert "truncatable" not in result
class TestAnalyzeConversation:
    """Tests for conversation analysis."""

    def test_counts_tools(self):
        """Test that tool usage is counted."""
        messages = [
            (
                "assistant",
                json.dumps(
                    {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "id": "1",
                                "input": {},
                            },
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "id": "2",
                                "input": {},
                            },
                            {
                                "type": "tool_use",
                                "name": "Write",
                                "id": "3",
                                "input": {},
                            },
                        ]
                    }
                ),
                "2025-01-01T00:00:00Z",
            ),
        ]
        result = analyze_conversation(messages)
        assert result["tool_counts"]["Bash"] == 2
        assert result["tool_counts"]["Write"] == 1

    def test_extracts_commits(self):
        """Test that git commits are extracted."""
        messages = [
            (
                "user",
                json.dumps(
                    {
                        "content": [
                            {
                                "type": "tool_result",
                                "content": "[main abc1234] Add new feature\n 1 file changed",
                            }
                        ]
                    }
                ),
                "2025-01-01T00:00:00Z",
            ),
        ]
        result = analyze_conversation(messages)
        assert len(result["commits"]) == 1
        assert result["commits"][0][0] == "abc1234"
        assert "Add new feature" in result["commits"][0][1]


class TestFormatToolStats:
    """Tests for tool stats formatting."""

    def test_formats_counts(self):
        """Test tool count formatting."""
        counts = {"Bash": 5, "Read": 3, "Write": 1}
        result = format_tool_stats(counts)
        assert "5 bash" in result
        assert "3 read" in result
        assert "1 write" in result

    def test_empty_counts(self):
        """Test empty tool counts."""
        assert format_tool_stats({}) == ""


class TestIsToolResultMessage:
    """Tests for tool result message detection."""

    def test_detects_tool_result_only(self):
        """Test detection of tool-result-only messages."""
        message = {"content": [{"type": "tool_result", "content": "result"}]}
        assert is_tool_result_message(message) is True

    def test_rejects_mixed_content(self):
        """Test rejection of mixed content messages."""
        message = {
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_result", "content": "result"},
            ]
        }
        assert is_tool_result_message(message) is False

    def test_rejects_empty(self):
        """Test rejection of empty content."""
        assert is_tool_result_message({"content": []}) is False
        assert is_tool_result_message({"content": "string"}) is False


class TestContinuationLongTexts:
    """Tests for long text extraction from continuation conversations."""

    def test_long_text_in_continuation_appears_in_index(self, output_dir):
        """Test that long texts from continuation conversations appear in index.

        This is a regression test for a bug where conversations marked as
        continuations (isCompactSummary=True) were completely skipped when
        building the index, causing their long_texts to be lost.
        """
        # Create a session with:
        # 1. An initial user prompt
        # 2. Some messages
        # 3. A continuation prompt (isCompactSummary=True)
        # 4. An assistant message with a long text summary (>300 chars)
        session_data = {
            "loglines": [
                # Initial user prompt
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {
                        "content": "Build a Redis JavaScript module",
                        "role": "user",
                    },
                },
                # Some assistant work
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "I'll start working on this."}
                        ],
                    },
                },
                # Continuation prompt (context was summarized)
                {
                    "type": "user",
                    "timestamp": "2025-01-01T11:00:00.000Z",
                    "isCompactSummary": True,
                    "message": {
                        "content": "This session is being continued from a previous conversation...",
                        "role": "user",
                    },
                },
                # More assistant work after continuation
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T11:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Continuing the work..."}],
                    },
                },
                # Final summary - this is a LONG text (>300 chars) that should appear in index
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T12:00:00.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "All tasks completed successfully. Here's a summary of what was built:\n\n"
                                    "## Redis JavaScript Module\n\n"
                                    "A loadable Redis module providing JavaScript scripting via the mquickjs engine.\n\n"
                                    "### Commands Implemented\n"
                                    "- JS.EVAL - Execute JavaScript with KEYS/ARGV arrays\n"
                                    "- JS.LOAD / JS.CALL - Cache and call scripts by SHA1\n"
                                    "- JS.EXISTS / JS.FLUSH - Manage script cache\n\n"
                                    "All 41 tests pass. Changes pushed to branch."
                                ),
                            }
                        ],
                    },
                },
            ]
        }

        # Write the session to a temp file
        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        # Generate HTML
        generate_html(session_file, output_dir)

        # Read the index.html
        index_html = read_transcript(output_dir)

        # The long text summary should appear in the index
        # This is the bug: currently it doesn't because the continuation
        # conversation is skipped entirely
        assert (
            "All tasks completed successfully" in index_html
        ), "Long text from continuation conversation should appear in index"
        assert "Redis JavaScript Module" in index_html


class TestVersionOption:
    """Tests for the --version option."""

    def test_version_long_flag(self):
        """Test that --version shows version info."""
        import importlib.metadata
        from click.testing import CliRunner
        from ccutils import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        expected_version = importlib.metadata.version("ccutils")
        assert result.exit_code == 0
        assert expected_version in result.output

    def test_version_short_flag(self):
        """Test that -v shows version info."""
        import importlib.metadata
        from click.testing import CliRunner
        from ccutils import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["-v"])

        expected_version = importlib.metadata.version("ccutils")
        assert result.exit_code == 0
        assert expected_version in result.output


class TestOpenOption:
    """Tests for the --open option."""

    def test_session_open_calls_webbrowser(self, output_dir, monkeypatch):
        """Test that session --open opens the browser."""
        from click.testing import CliRunner
        from ccutils import cli
        import webbrowser

        fixture_path = Path(__file__).parent / "sample_session.json"

        # Track webbrowser.open calls
        opened_urls = []

        def mock_open(url):
            opened_urls.append(url)
            return True

        monkeypatch.setattr(webbrowser, "open", mock_open)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["convert", str(fixture_path), "-o", str(output_dir), "--open"],
        )

        assert result.exit_code == 0
        assert len(opened_urls) == 1
        assert "index.html" in opened_urls[0]
        assert opened_urls[0].startswith("file://")

    def test_import_open_calls_webbrowser(self, httpx_mock, output_dir, monkeypatch):
        """Test that import --open opens the browser."""
        from click.testing import CliRunner
        from ccutils import cli
        import webbrowser

        # Load sample session to mock API response
        fixture_path = Path(__file__).parent / "sample_session.json"
        with open(fixture_path) as f:
            session_data = json.load(f)

        httpx_mock.add_response(
            url="https://api.anthropic.com/v1/session_ingress/session/test-session-id",
            json=session_data,
        )

        # Track webbrowser.open calls
        opened_urls = []

        def mock_open(url):
            opened_urls.append(url)
            return True

        monkeypatch.setattr(webbrowser, "open", mock_open)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "web",
                "test-session-id",
                "--token",
                "test-token",
                "--org-uuid",
                "test-org",
                "-o",
                str(output_dir),
                "--open",
            ],
        )

        assert result.exit_code == 0
        assert len(opened_urls) == 1
        assert "index.html" in opened_urls[0]
        assert opened_urls[0].startswith("file://")


class TestParseSessionFile:
    """Tests for parse_session_file which abstracts both JSON and JSONL formats."""

    def test_parses_json_format(self):
        """Test that standard JSON format is parsed correctly."""
        fixture_path = Path(__file__).parent / "sample_session.json"
        result = parse_session_file(fixture_path)

        assert "loglines" in result
        assert len(result["loglines"]) > 0
        # Check first entry
        first = result["loglines"][0]
        assert first["type"] == "user"
        assert "timestamp" in first
        assert "message" in first

    def test_parses_jsonl_format(self):
        """Test that JSONL format is parsed and converted to standard format."""
        fixture_path = Path(__file__).parent / "sample_session.jsonl"
        result = parse_session_file(fixture_path)

        assert "loglines" in result
        assert len(result["loglines"]) > 0
        # Check structure matches JSON format
        for entry in result["loglines"]:
            assert "type" in entry
            # Skip summary entries which don't have message
            if entry["type"] in ("user", "assistant"):
                assert "timestamp" in entry
                assert "message" in entry

    def test_jsonl_preserves_contextual_entries(self):
        """v0.15: non-message entries (system, attachment, meta, summary,
        file-history-snapshot, queue-operation, pr-link, last-prompt) are
        preserved so the HTML renderer can dispatch on them. Progress
        entries are still skipped (too high-volume to render inline)."""
        fixture_path = Path(__file__).parent / "sample_session.jsonl"
        result = parse_session_file(fixture_path)

        valid_types = {
            "user",
            "assistant",
            "system",
            "attachment",
            "meta",
            "file-history-snapshot",
            "queue-operation",
            "pr-link",
            "summary",
            "last-prompt",
        }
        types_seen = {entry["type"] for entry in result["loglines"]}
        assert types_seen.issubset(valid_types)
        assert "progress" not in types_seen

    def test_jsonl_preserves_message_content(self):
        """Test that message content is preserved correctly."""
        fixture_path = Path(__file__).parent / "sample_session.jsonl"
        result = parse_session_file(fixture_path)

        # Find the first user message
        user_msg = next(e for e in result["loglines"] if e["type"] == "user")
        assert user_msg["message"]["content"] == "Create a hello world function"

    def test_jsonl_generates_html(self, output_dir):
        """Test that JSONL files can be converted to HTML."""
        fixture_path = Path(__file__).parent / "sample_session.jsonl"
        generate_html(fixture_path, output_dir)

        index_html = read_transcript(output_dir)
        assert "hello world" in index_html.lower()
class TestNonMessageEntryRendering:
    """v0.15 Phase 1: the renderer must dispatch on non-user/assistant
    entry types so contextual signals (permission mode transitions, hook
    durations, queued prompts, etc.) are visible in the HTML transcript
    instead of silently dropped."""

    def _render(self, tmp_path, entries):
        """Write JSONL entries to a temp file and return rendered page-001 HTML."""
        # Always anchor with one user message so a conversation exists for
        # contextual entries to attach to.
        all_entries = [
            {
                "type": "user",
                "timestamp": "2026-04-19T10:00:00Z",
                "message": {"role": "user", "content": "kick off"},
            }
        ] + entries
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text("\n".join(json.dumps(e) for e in all_entries))
        out_dir = tmp_path / "out"
        generate_html(jsonl, out_dir)
        return read_transcript(out_dir)

    def test_permission_mode_renders_styled_banner(self, tmp_path):
        html_out = self._render(
            tmp_path,
            [
                {
                    "type": "permission-mode",
                    "permissionMode": "plan",
                    "timestamp": "2026-04-19T10:00:01Z",
                }
            ],
        )
        assert '<span class="entry-banner-label">Permission mode</span>' in html_out
        assert ">plan</pre>" in html_out

    def test_last_prompt_renders_styled_banner(self, tmp_path):
        html_out = self._render(
            tmp_path,
            [
                {
                    "type": "last-prompt",
                    "lastPrompt": "queue this up",
                    "timestamp": "2026-04-19T10:00:02Z",
                }
            ],
        )
        assert '<span class="entry-banner-label">Queued prompt</span>' in html_out
        assert "queue this up" in html_out

    def test_system_turn_duration_renders(self, tmp_path):
        html_out = self._render(
            tmp_path,
            [
                {
                    "type": "system",
                    "subtype": "turn_duration",
                    "durationMs": 1234,
                    "messageCount": 5,
                    "timestamp": "2026-04-19T10:00:03Z",
                }
            ],
        )
        assert '<span class="entry-banner-label">Turn duration</span>' in html_out
        assert "1234 ms / 5 messages" in html_out

    def test_attachment_hook_success_renders(self, tmp_path):
        html_out = self._render(
            tmp_path,
            [
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "hook_success",
                        "hookName": "PreToolUse:Bash",
                        "durationMs": 42,
                    },
                    "timestamp": "2026-04-19T10:00:04Z",
                }
            ],
        )
        assert '<span class="entry-banner-label">Hook</span>' in html_out
        assert "PreToolUse:Bash (42 ms)" in html_out

    def test_unknown_subtype_falls_back_to_details(self, tmp_path):
        """Entry types we don't have a styled renderer for should still
        appear -- via a collapsed <details> -- never silently dropped."""
        html_out = self._render(
            tmp_path,
            [
                {
                    "type": "system",
                    "subtype": "bridge_status",  # not in our styled set
                    "someField": "some-value",
                    "timestamp": "2026-04-19T10:00:05Z",
                }
            ],
        )
        assert '<details class="entry-fallback">' in html_out
        assert "bridge_status" in html_out

    def test_progress_entries_are_skipped(self, tmp_path):
        """Progress entries are intentionally dropped -- too high-volume
        for inline rendering. v0.15 captures them in fact_progress_events."""
        html_out = self._render(
            tmp_path,
            [
                {
                    "type": "progress",
                    "data": {"type": "hook_progress", "hookName": "PreToolUse:Bash"},
                    "timestamp": "2026-04-19T10:00:06Z",
                }
            ],
        )
        assert "hook_progress" not in html_out
        assert "PreToolUse:Bash" not in html_out


class TestGetSessionSummary:
    """Tests for get_session_summary which extracts summary from session files."""

    def test_gets_summary_from_jsonl(self):
        """Test extracting summary from JSONL file."""
        fixture_path = Path(__file__).parent / "sample_session.jsonl"
        summary = get_session_summary(fixture_path)
        assert summary == "Test session for JSONL parsing"

    def test_gets_first_user_message_if_no_summary(self, tmp_path):
        """Test falling back to first user message when no summary entry."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hello world test"}}\n'
        )
        summary = get_session_summary(jsonl_file)
        assert summary == "Hello world test"

    def test_returns_no_summary_for_empty_file(self, tmp_path):
        """Test handling empty or invalid files."""
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("", encoding="utf-8")
        summary = get_session_summary(jsonl_file)
        assert summary == "(no summary)"

    def test_truncates_long_summaries(self, tmp_path):
        """Test that long summaries are truncated."""
        jsonl_file = tmp_path / "long.jsonl"
        long_text = "x" * 300
        jsonl_file.write_text(f'{{"type":"summary","summary":"{long_text}"}}\n')
        summary = get_session_summary(jsonl_file, max_length=100)
        assert len(summary) <= 100
        assert summary.endswith("...")


class TestFindLocalSessions:
    """Tests for find_local_sessions which discovers local JSONL files."""

    def test_finds_jsonl_files(self, tmp_path):
        """Test finding JSONL files in projects directory."""
        # Create mock .claude/projects structure
        projects_dir = tmp_path / ".claude" / "projects" / "test-project"
        projects_dir.mkdir(parents=True)

        # Create a session file
        session_file = projects_dir / "session-123.jsonl"
        session_file.write_text(
            '{"type":"summary","summary":"Test session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
        )

        results = find_local_sessions(tmp_path / ".claude" / "projects", limit=10)
        assert len(results) == 1
        assert results[0][0] == session_file
        assert results[0][1] == "Test session"

    def test_excludes_agent_files(self, tmp_path):
        """Test that agent- prefixed files are excluded."""
        projects_dir = tmp_path / ".claude" / "projects" / "test-project"
        projects_dir.mkdir(parents=True)

        # Create agent file (should be excluded)
        agent_file = projects_dir / "agent-123.jsonl"
        agent_file.write_text('{"type":"user","message":{"content":"test"}}\n')

        # Create regular file (should be included)
        session_file = projects_dir / "session-123.jsonl"
        session_file.write_text(
            '{"type":"summary","summary":"Real session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
        )

        results = find_local_sessions(tmp_path / ".claude" / "projects", limit=10)
        assert len(results) == 1
        assert "agent-" not in results[0][0].name

    def test_excludes_warmup_sessions(self, tmp_path):
        """Test that warmup sessions are excluded."""
        projects_dir = tmp_path / ".claude" / "projects" / "test-project"
        projects_dir.mkdir(parents=True)

        # Create warmup file (should be excluded)
        warmup_file = projects_dir / "warmup-session.jsonl"
        warmup_file.write_text('{"type":"summary","summary":"warmup"}\n')

        # Create regular file
        session_file = projects_dir / "session-123.jsonl"
        session_file.write_text(
            '{"type":"summary","summary":"Real session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
        )

        results = find_local_sessions(tmp_path / ".claude" / "projects", limit=10)
        assert len(results) == 1
        assert results[0][1] == "Real session"

    def test_sorts_by_modification_time(self, tmp_path):
        """Test that results are sorted by modification time, newest first."""
        import time

        projects_dir = tmp_path / ".claude" / "projects" / "test-project"
        projects_dir.mkdir(parents=True)

        # Create files with different mtimes
        file1 = projects_dir / "older.jsonl"
        file1.write_text(
            '{"type":"summary","summary":"Older"}\n{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"test"}}\n'
        )

        time.sleep(0.1)  # Ensure different mtime

        file2 = projects_dir / "newer.jsonl"
        file2.write_text(
            '{"type":"summary","summary":"Newer"}\n{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"test"}}\n'
        )

        results = find_local_sessions(tmp_path / ".claude" / "projects", limit=10)
        assert len(results) == 2
        assert results[0][1] == "Newer"  # Most recent first
        assert results[1][1] == "Older"

    def test_respects_limit(self, tmp_path):
        """Test that limit parameter is respected."""
        projects_dir = tmp_path / ".claude" / "projects" / "test-project"
        projects_dir.mkdir(parents=True)

        # Create 5 files
        for i in range(5):
            f = projects_dir / f"session-{i}.jsonl"
            f.write_text(
                f'{{"type":"summary","summary":"Session {i}"}}\n{{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{{"role":"user","content":"test"}}}}\n'
            )

        results = find_local_sessions(tmp_path / ".claude" / "projects", limit=3)
        assert len(results) == 3

    def test_project_filter_matches_project(self, tmp_path):
        """Test filtering sessions by project name."""
        projects_dir = tmp_path / ".claude" / "projects"

        # Create two project directories
        project_a = projects_dir / "-home-user-projects-project-alpha"
        project_a.mkdir(parents=True)
        project_b = projects_dir / "-home-user-projects-project-beta"
        project_b.mkdir(parents=True)

        # Create sessions in each project
        session_a = project_a / "session-a.jsonl"
        session_a.write_text(
            '{"type":"summary","summary":"Alpha session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"test"}}\n'
        )
        session_b = project_b / "session-b.jsonl"
        session_b.write_text(
            '{"type":"summary","summary":"Beta session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"test"}}\n'
        )

        # Filter by "alpha" - should only get alpha project sessions
        results = find_local_sessions(projects_dir, limit=10, project_filter="alpha")
        assert len(results) == 1
        assert results[0][1] == "Alpha session"

    def test_project_filter_case_insensitive(self, tmp_path):
        """Test that project filter is case-insensitive."""
        projects_dir = tmp_path / ".claude" / "projects"
        project = projects_dir / "-home-user-projects-MyProject"
        project.mkdir(parents=True)

        session = project / "session.jsonl"
        session.write_text(
            '{"type":"summary","summary":"Test session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"test"}}\n'
        )

        # Filter with different case
        results = find_local_sessions(
            projects_dir, limit=10, project_filter="MYPROJECT"
        )
        assert len(results) == 1

    def test_project_filter_partial_match(self, tmp_path):
        """Test that partial project names match."""
        projects_dir = tmp_path / ".claude" / "projects"
        project = projects_dir / "-home-user-projects-claude-code-transcripts"
        project.mkdir(parents=True)

        session = project / "session.jsonl"
        session.write_text(
            '{"type":"summary","summary":"Transcripts session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"test"}}\n'
        )

        # Filter with partial name
        results = find_local_sessions(
            projects_dir, limit=10, project_filter="transcript"
        )
        assert len(results) == 1

    def test_project_filter_no_match(self, tmp_path):
        """Test that non-matching filter returns empty."""
        projects_dir = tmp_path / ".claude" / "projects"
        project = projects_dir / "-home-user-projects-myproject"
        project.mkdir(parents=True)

        session = project / "session.jsonl"
        session.write_text(
            '{"type":"summary","summary":"Test session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"test"}}\n'
        )

        results = find_local_sessions(
            projects_dir, limit=10, project_filter="nonexistent"
        )
        assert len(results) == 0


class TestLocalSessionCLI:
    """Tests for CLI behavior with local sessions."""

    def test_local_shows_sessions_and_converts(self, tmp_path, monkeypatch):
        """Test that 'local' command shows sessions and converts selected one."""
        from click.testing import CliRunner
        from ccutils import cli
        import questionary

        # Create mock .claude/projects structure
        projects_dir = tmp_path / ".claude" / "projects" / "test-project"
        projects_dir.mkdir(parents=True)

        session_file = projects_dir / "session-123.jsonl"
        session_file.write_text(
            '{"type":"summary","summary":"Test local session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
        )

        # Mock Path.home() to return our tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Mock questionary.checkbox to return the session file (as list)
        class MockCheckbox:
            def __init__(self, *args, **kwargs):
                pass

            def ask(self):
                return [session_file]

        monkeypatch.setattr(questionary, "checkbox", MockCheckbox)

        runner = CliRunner()
        result = runner.invoke(cli, ["local"])

        assert result.exit_code == 0
        assert (
            "Scanning sessions" in result.output
            or "Loading local sessions" in result.output
        )
        assert "Selected 1 session" in result.output

    def test_no_args_runs_local_command(self, tmp_path, monkeypatch):
        """Test that running with no arguments runs local command."""
        from click.testing import CliRunner
        from ccutils import cli
        import questionary

        # Create mock .claude/projects structure
        projects_dir = tmp_path / ".claude" / "projects" / "test-project"
        projects_dir.mkdir(parents=True)

        session_file = projects_dir / "session-123.jsonl"
        session_file.write_text(
            '{"type":"summary","summary":"Test default session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
        )

        # Mock Path.home() to return our tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Mock questionary.checkbox to return the session file (as list)
        class MockCheckbox:
            def __init__(self, *args, **kwargs):
                pass

            def ask(self):
                return [session_file]

        monkeypatch.setattr(questionary, "checkbox", MockCheckbox)

        runner = CliRunner()
        result = runner.invoke(cli, [])

        assert result.exit_code == 0
        assert (
            "Scanning sessions" in result.output
            or "Loading local sessions" in result.output
        )

    def test_local_handles_cancelled_selection(self, tmp_path, monkeypatch):
        """Test that local command handles cancelled selection gracefully."""
        from click.testing import CliRunner
        from ccutils import cli
        import questionary

        # Create mock .claude/projects structure
        projects_dir = tmp_path / ".claude" / "projects" / "test-project"
        projects_dir.mkdir(parents=True)

        session_file = projects_dir / "session-123.jsonl"
        session_file.write_text(
            '{"type":"summary","summary":"Test session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
        )

        # Mock Path.home() to return our tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Mock questionary.checkbox to return empty list (cancelled)
        class MockCheckbox:
            def __init__(self, *args, **kwargs):
                pass

            def ask(self):
                return []

        monkeypatch.setattr(questionary, "checkbox", MockCheckbox)

        runner = CliRunner()
        result = runner.invoke(cli, ["local"])

        assert result.exit_code == 0
        assert "No sessions selected" in result.output


class TestFilterUI:
    """The in-document filter that replaced fetch-based search.

    Claim: the old search fetched every page-NNN.html and parsed it with
    DOMParser. Browsers block fetch() to file:// URLs, and the .catch()
    swallowed the failure, so opening an export from disk and searching
    reported "Found 0 result(s)" -- indistinguishable from no matches. The
    replacement filters messages already in the document, so it cannot fail
    that way. These assertions exist to stop a network dependency coming back.
    """

    def _html(self, output_dir):
        fixture_path = Path(__file__).parent / "sample_session.json"
        generate_html(fixture_path, output_dir, github_repo="example/project")
        return read_transcript(output_dir)

    def test_filter_input_present(self, output_dir):
        html_out = self._html(output_dir)
        assert 'id="q"' in html_out
        assert 'id="shown"' in html_out

    def test_filter_script_is_inline_and_never_fetches(self, output_dir):
        html_out = self._html(output_dir)
        assert "addEventListener" in html_out
        assert "fetch(" not in html_out
        assert "DOMParser" not in html_out
        assert "XMLHttpRequest" not in html_out

    def test_filter_matches_data_search_not_rendered_text(self, output_dir):
        """Matching the DOM would diverge from data-search the first time CSS
        changed what is displayed."""
        html_out = self._html(output_dir)
        assert "data-search" in html_out

    def test_no_sibling_search_assets(self, output_dir):
        fixture_path = Path(__file__).parent / "sample_session.json"
        generate_html(fixture_path, output_dir, github_repo="example/project")
        names = {p.name for p in output_dir.iterdir()}
        assert "search.js" not in names
        assert "transcript.css" not in names
        assert "transcript.js" not in names


class TestExtractSessionSlug:
    """Tests for extract_session_slug which extracts the slug field from session files."""

    def test_extracts_slug_from_first_line(self, tmp_path):
        """Test extracting slug from session file first line."""
        project = tmp_path / "project"
        project.mkdir()

        session = project / "session.jsonl"
        session.write_text(
            '{"type":"summary","slug":"cozy-imagining-karp","sessionId":"abc123"}\n'
            '{"type":"user","message":{"content":"Hello"}}\n'
        )

        slug = extract_session_slug(session)
        assert slug == "cozy-imagining-karp"

    def test_returns_none_for_no_slug(self, tmp_path):
        """Test returning None when no slug is present."""
        project = tmp_path / "project"
        project.mkdir()

        session = project / "session.jsonl"
        session.write_text(
            '{"type":"user","message":{"content":"Hello"}}\n'
            '{"type":"assistant","message":{"content":"Hi"}}\n'
        )

        slug = extract_session_slug(session)
        assert slug is None

    def test_handles_empty_file(self, tmp_path):
        """Test handling empty files gracefully."""
        project = tmp_path / "project"
        project.mkdir()

        session = project / "session.jsonl"
        session.write_text("")

        slug = extract_session_slug(session)
        assert slug is None

    def test_handles_invalid_json(self, tmp_path):
        """Test handling files with invalid JSON."""
        project = tmp_path / "project"
        project.mkdir()

        session = project / "session.jsonl"
        session.write_text("not valid json\n")

        slug = extract_session_slug(session)
        assert slug is None

    def test_handles_nonexistent_file(self, tmp_path):
        """Test handling nonexistent files."""
        session = tmp_path / "nonexistent.jsonl"

        slug = extract_session_slug(session)
        assert slug is None

    def test_finds_slug_in_any_line(self, tmp_path):
        """Test that slug can be found even if not in first line."""
        project = tmp_path / "project"
        project.mkdir()

        # Slug might appear in a later line in some formats
        session = project / "session.jsonl"
        session.write_text(
            '{"type":"user","message":{"content":"Hello"}}\n'
            '{"type":"summary","slug":"later-appearing-slug"}\n'
        )

        slug = extract_session_slug(session)
        assert slug == "later-appearing-slug"


class TestFindLocalSessionsWithSlugs:
    """Tests for find_local_sessions returning slug information."""

    def test_returns_slug_in_tuple(self, tmp_path):
        """Test that find_local_sessions returns (filepath, summary, slug) tuples."""
        projects_dir = tmp_path / ".claude" / "projects"
        project = projects_dir / "test-project"
        project.mkdir(parents=True)

        session = project / "session.jsonl"
        session.write_text(
            '{"type":"summary","summary":"Test session","slug":"my-test-slug"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
        )

        results = find_local_sessions(projects_dir, limit=10)
        assert len(results) == 1
        assert len(results[0]) == 3  # (filepath, summary, slug)
        assert results[0][0] == session
        assert results[0][1] == "Test session"
        assert results[0][2] == "my-test-slug"

    def test_returns_none_slug_when_missing(self, tmp_path):
        """Test that sessions without slug return None for slug."""
        projects_dir = tmp_path / ".claude" / "projects"
        project = projects_dir / "test-project"
        project.mkdir(parents=True)

        session = project / "session.jsonl"
        session.write_text(
            '{"type":"summary","summary":"No slug session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
        )

        results = find_local_sessions(projects_dir, limit=10)
        assert len(results) == 1
        assert results[0][2] is None

    def test_groups_sessions_by_slug(self, tmp_path):
        """Test that sessions with same slug can be identified as related."""
        import time

        projects_dir = tmp_path / ".claude" / "projects"
        project = projects_dir / "test-project"
        project.mkdir(parents=True)

        # Create two sessions with the same slug (conversation chain)
        session1 = project / "session1.jsonl"
        session1.write_text(
            '{"type":"summary","summary":"First session","slug":"shared-conversation"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
        )
        time.sleep(0.1)  # Ensure different mtime

        session2 = project / "session2.jsonl"
        session2.write_text(
            '{"type":"summary","summary":"Second session","slug":"shared-conversation"}\n'
            '{"type":"user","timestamp":"2025-01-02T00:00:00Z","message":{"role":"user","content":"Continue"}}\n'
        )
        time.sleep(0.1)

        # Create a session with a different slug
        session3 = project / "session3.jsonl"
        session3.write_text(
            '{"type":"summary","summary":"Different conversation","slug":"another-slug"}\n'
            '{"type":"user","timestamp":"2025-01-03T00:00:00Z","message":{"role":"user","content":"New"}}\n'
        )

        results = find_local_sessions(projects_dir, limit=10)
        assert len(results) == 3

        # Group by slug manually to verify
        slugs = [r[2] for r in results]
        assert slugs.count("shared-conversation") == 2
        assert slugs.count("another-slug") == 1


class TestBuildSessionChoices:
    """Tests for build_flat_choices which creates questionary choices."""

    def _make_meta(
        self,
        path,
        summary,
        slug=None,
        project_name="test-project",
        project_path="test-project",
    ):
        from ccutils.parsers.metadata import SessionMetadata

        stat = path.stat()
        return SessionMetadata(
            path=path,
            project_name=project_name,
            project_path=project_path,
            model_short="",
            summary=summary,
            mtime=stat.st_mtime,
            size=stat.st_size,
            slug=slug,
        )

    def test_collapsed_chains_groups_sessions_by_slug(self, tmp_path):
        """Test that collapsed mode groups sessions with same slug into single choice."""
        from ccutils.tui import build_flat_choices

        projects_dir = tmp_path / ".claude" / "projects"
        project = projects_dir / "test-project"
        project.mkdir(parents=True)

        # Create chain of 3 sessions with same slug
        for i in range(3):
            session = project / f"session{i}.jsonl"
            session.write_text(
                f'{{"type":"summary","summary":"Session {i}","slug":"my-chain"}}\n'
                f'{{"type":"user","timestamp":"2025-01-0{i+1}T00:00:00Z","message":{{"role":"user","content":"Hello"}}}}\n'
            )

        # Create a standalone session (no slug)
        standalone = project / "standalone.jsonl"
        standalone.write_text(
            '{"type":"summary","summary":"Standalone session"}\n'
            '{"type":"user","timestamp":"2025-01-04T00:00:00Z","message":{"role":"user","content":"Hi"}}\n'
        )

        grouped = {"test-project": []}
        for f in sorted(project.glob("*.jsonl")):
            summary = f"Summary for {f.stem}"
            slug = "my-chain" if "session" in f.stem else None
            grouped["test-project"].append(self._make_meta(f, summary, slug))

        choices = build_flat_choices(grouped, expand_chains=False)

        # Filter out Separator objects
        import questionary

        value_choices = [c for c in choices if not isinstance(c, questionary.Separator)]

        # Should have 2 choices: 1 chain (with 3 sessions) + 1 standalone
        assert len(value_choices) == 2

        # Find the chain choice - its value should be a list
        chain_choice = None
        standalone_choice = None
        for c in value_choices:
            if isinstance(c.value, list):
                chain_choice = c
            else:
                standalone_choice = c

        assert chain_choice is not None, "Should have a chain choice with list value"
        assert len(chain_choice.value) == 3, "Chain should contain 3 session paths"
        assert standalone_choice is not None, "Should have standalone choice"

    def test_collapsed_chain_shows_metadata(self, tmp_path):
        """Test that collapsed chain shows resumed count in FormattedText label."""
        from ccutils.tui import build_flat_choices
        import questionary

        projects_dir = tmp_path / ".claude" / "projects"
        project = projects_dir / "test-project"
        project.mkdir(parents=True)

        # Create chain of 2 sessions
        session1 = project / "session1.jsonl"
        session1.write_text(
            '{"type":"summary","summary":"First","slug":"test-chain"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
        )
        session2 = project / "session2.jsonl"
        session2.write_text(
            '{"type":"summary","summary":"Second","slug":"test-chain"}\n'
            '{"type":"user","timestamp":"2025-01-02T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
        )

        grouped = {
            "test-project": [
                self._make_meta(session1, "First", "test-chain"),
                self._make_meta(session2, "Second", "test-chain"),
            ]
        }

        choices = build_flat_choices(grouped, expand_chains=False)
        value_choices = [c for c in choices if not isinstance(c, questionary.Separator)]

        assert len(value_choices) == 1
        chain_choice = value_choices[0]

        # Title is now FormattedText (list of tuples) with chain indicator
        assert isinstance(chain_choice.title, list)
        text = "".join(t[1] for t in chain_choice.title)
        assert "[2 resumed]" in text

    def test_collapsed_chain_shows_latest_summary(self, tmp_path):
        """Test that collapsed chain shows the summary from the most recent session."""
        from ccutils.tui import build_flat_choices
        import questionary
        import time

        projects_dir = tmp_path / ".claude" / "projects"
        project = projects_dir / "test-project"
        project.mkdir(parents=True)

        # Create older session
        session1 = project / "session1.jsonl"
        session1.write_text(
            '{"type":"summary","summary":"Old summary from first session","slug":"test-chain"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
        )
        time.sleep(0.1)  # Ensure different mtime

        # Create newer session (should be displayed)
        session2 = project / "session2.jsonl"
        session2.write_text(
            '{"type":"summary","summary":"Latest summary from recent session","slug":"test-chain"}\n'
            '{"type":"user","timestamp":"2025-01-02T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
        )

        grouped = {
            "test-project": [
                self._make_meta(
                    session1, "Old summary from first session", "test-chain"
                ),
                self._make_meta(
                    session2, "Latest summary from recent session", "test-chain"
                ),
            ]
        }

        choices = build_flat_choices(grouped, expand_chains=False)
        value_choices = [c for c in choices if not isinstance(c, questionary.Separator)]

        chain_choice = value_choices[0]

        # Title is FormattedText -- extract text from tuples
        text = "".join(t[1] for t in chain_choice.title)

        # Should show the latest (most recent) summary, not the old one
        assert "Latest summary from recent" in text
        assert "Old summary from first" not in text

    def test_expanded_chains_shows_individual_sessions(self, tmp_path):
        """Test that expanded mode shows individual sessions with inline project markers."""
        from ccutils.tui import build_flat_choices
        import questionary

        projects_dir = tmp_path / ".claude" / "projects"
        project = projects_dir / "test-project"
        project.mkdir(parents=True)

        # Create chain of 2 sessions
        session1 = project / "session1.jsonl"
        session1.write_text(
            '{"type":"summary","summary":"First","slug":"test-chain"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
        )
        session2 = project / "session2.jsonl"
        session2.write_text(
            '{"type":"summary","summary":"Second","slug":"test-chain"}\n'
            '{"type":"user","timestamp":"2025-01-02T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
        )

        grouped = {
            "test-project": [
                self._make_meta(session1, "First", "test-chain"),
                self._make_meta(session2, "Second", "test-chain"),
            ]
        }

        choices = build_flat_choices(grouped, expand_chains=True)

        # No separators - using inline project markers instead
        separators = [c for c in choices if isinstance(c, questionary.Separator)]
        value_choices = [c for c in choices if not isinstance(c, questionary.Separator)]

        assert len(separators) == 0, "Should not use Separator objects"

        # Should have 2 individual session choices (not grouped)
        assert len(value_choices) == 2

        # Each choice value should be a single Path, not a list
        for c in value_choices:
            assert not isinstance(
                c.value, list
            ), "Expanded mode should have individual paths"
            # Title is FormattedText -- project marker should be in the text
            text = "".join(t[1] for t in c.title)
            assert "[" in text and "]" in text


class TestFlattenSelectedSessions:
    """Tests for flatten_selected_sessions helper."""

    def test_flattens_mixed_selections(self, tmp_path):
        """Test flattening mixed list and single path selections."""
        from ccutils import flatten_selected_sessions

        path1 = tmp_path / "session1.jsonl"
        path2 = tmp_path / "session2.jsonl"
        path3 = tmp_path / "session3.jsonl"

        # Mixed: one chain (list) and one standalone (path)
        selected = [[path1, path2], path3]

        result = flatten_selected_sessions(selected)

        assert len(result) == 3
        assert path1 in result
        assert path2 in result
        assert path3 in result

    def test_handles_all_single_selections(self, tmp_path):
        """Test with all single path selections."""
        from ccutils import flatten_selected_sessions

        path1 = tmp_path / "session1.jsonl"
        path2 = tmp_path / "session2.jsonl"

        selected = [path1, path2]

        result = flatten_selected_sessions(selected)

        assert result == [path1, path2]

    def test_handles_all_chain_selections(self, tmp_path):
        """Test with all chain (list) selections."""
        from ccutils import flatten_selected_sessions

        path1 = tmp_path / "session1.jsonl"
        path2 = tmp_path / "session2.jsonl"
        path3 = tmp_path / "session3.jsonl"

        selected = [[path1, path2], [path3]]

        result = flatten_selected_sessions(selected)

        assert len(result) == 3

    def test_handles_empty_selection(self):
        """Test with empty selection."""
        from ccutils import flatten_selected_sessions

        result = flatten_selected_sessions([])

        assert result == []


class TestDynamicTruncation:
    """Tests for dynamic session display truncation based on terminal width."""

    def test_get_terminal_width_returns_sensible_default(self):
        """Test terminal width helper returns sensible default."""
        from ccutils.tui import get_terminal_width

        width = get_terminal_width()

        # Should return a reasonable width (not 0 or negative)
        assert width >= 80
        assert width <= 500  # Reasonable upper bound


class TestInlineProjectMarkers:
    """Tests for inline project markers replacing Separator objects."""

    def _make_meta(
        self,
        path,
        summary,
        slug=None,
        project_name="test-project",
        project_path="test-project",
    ):
        from ccutils.parsers.metadata import SessionMetadata

        stat = path.stat()
        return SessionMetadata(
            path=path,
            project_name=project_name,
            project_path=project_path,
            model_short="",
            summary=summary,
            mtime=stat.st_mtime,
            size=stat.st_size,
            slug=slug,
        )

    def test_build_session_choices_no_separators(self, tmp_path):
        """Test that build_flat_choices no longer uses Separator objects."""
        from ccutils.tui import build_flat_choices
        import questionary

        project = tmp_path / "test-project"
        project.mkdir()

        session = project / "session.jsonl"
        session.write_text(
            '{"type":"summary","summary":"Test session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hi"}}\n'
        )

        grouped = {"test-project": [self._make_meta(session, "Test session")]}

        choices = build_flat_choices(grouped, expand_chains=False)

        # Should NOT contain any Separator objects
        separators = [c for c in choices if isinstance(c, questionary.Separator)]
        assert len(separators) == 0, "Should not use Separator objects"

    def test_build_session_choices_includes_project_prefix(self, tmp_path):
        """Test that each choice includes inline project name prefix."""
        from ccutils.tui import build_flat_choices
        import questionary

        project = tmp_path / "test-project"
        project.mkdir()

        session = project / "session.jsonl"
        session.write_text(
            '{"type":"summary","summary":"Test session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hi"}}\n'
        )

        grouped = {"test-project": [self._make_meta(session, "Test session")]}

        choices = build_flat_choices(grouped, expand_chains=False)
        value_choices = [c for c in choices if not isinstance(c, questionary.Separator)]

        # Each choice title is FormattedText with project name in brackets
        for choice in value_choices:
            text = "".join(t[1] for t in choice.title)
            assert "[" in text and "]" in text
            # Project name "test-project" should appear
            assert "test-project" in text.lower() or "test" in text.lower()

    def test_multiple_projects_inline_markers(self, tmp_path):
        """Test multiple projects show inline markers for each."""
        from ccutils.tui import build_flat_choices
        import questionary

        # Create two projects
        project1 = tmp_path / "project-alpha"
        project1.mkdir()
        session1 = project1 / "session1.jsonl"
        session1.write_text(
            '{"type":"summary","summary":"Alpha session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hi"}}\n'
        )

        project2 = tmp_path / "project-beta"
        project2.mkdir()
        session2 = project2 / "session2.jsonl"
        session2.write_text(
            '{"type":"summary","summary":"Beta session"}\n'
            '{"type":"user","timestamp":"2025-01-02T00:00:00Z","message":{"role":"user","content":"Hi"}}\n'
        )

        grouped = {
            "project-alpha": [
                self._make_meta(
                    session1,
                    "Alpha session",
                    project_name="project-alpha",
                    project_path="project-alpha",
                ),
            ],
            "project-beta": [
                self._make_meta(
                    session2,
                    "Beta session",
                    project_name="project-beta",
                    project_path="project-beta",
                ),
            ],
        }

        choices = build_flat_choices(grouped, expand_chains=False)
        value_choices = [c for c in choices if not isinstance(c, questionary.Separator)]

        # Should have 2 choices (no separators)
        assert len(value_choices) == 2

        # Titles (FormattedText) should include respective project names
        texts = ["".join(t[1] for t in c.title) for c in value_choices]
        assert any("alpha" in t.lower() for t in texts)
        assert any("beta" in t.lower() for t in texts)

    def test_project_name_in_flat_choices(self, tmp_path):
        """Test project names appear in FormattedText labels."""
        from ccutils.tui import build_flat_choices
        import questionary

        # Create projects with different name lengths
        short_proj = tmp_path / "api"
        short_proj.mkdir()
        session1 = short_proj / "session1.jsonl"
        session1.write_text('{"type":"user","message":{"content":"Hi"}}\n')

        long_proj = tmp_path / "my-very-long-project-name"
        long_proj.mkdir()
        session2 = long_proj / "session2.jsonl"
        session2.write_text('{"type":"user","message":{"content":"Hi"}}\n')

        grouped = {
            "api": [
                self._make_meta(
                    session1,
                    "Short project session",
                    project_name="api",
                    project_path="api",
                ),
            ],
            "my-very-long-project-name": [
                self._make_meta(
                    session2,
                    "Long project session",
                    project_name="my-very-long-project-name",
                    project_path="my-very-long-project-name",
                ),
            ],
        }

        choices = build_flat_choices(grouped, expand_chains=False)
        value_choices = [c for c in choices if not isinstance(c, questionary.Separator)]

        # Each title should be FormattedText with project name in brackets
        for choice in value_choices:
            assert isinstance(choice.title, list)
            text = "".join(t[1] for t in choice.title)
            assert "[" in text and "]" in text


class TestFlatMode:
    """Tests for flat mode session display (build_flat_choices merges all projects)."""

    def _make_meta(
        self,
        path,
        summary,
        slug=None,
        project_name="test-project",
        project_path="test-project",
    ):
        from ccutils.parsers.metadata import SessionMetadata

        stat = path.stat()
        return SessionMetadata(
            path=path,
            project_name=project_name,
            project_path=project_path,
            model_short="",
            summary=summary,
            mtime=stat.st_mtime,
            size=stat.st_size,
            slug=slug,
        )

    def test_flat_mode_merges_projects(self, tmp_path):
        """Test that build_flat_choices shows all sessions in one list."""
        from ccutils.tui import build_flat_choices
        import questionary
        import time

        # Create two projects
        project1 = tmp_path / "project-alpha"
        project1.mkdir()
        session1 = project1 / "session1.jsonl"
        session1.write_text(
            '{"type":"summary","summary":"Alpha session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hi"}}\n'
        )
        time.sleep(0.05)  # Ensure different mtimes

        project2 = tmp_path / "project-beta"
        project2.mkdir()
        session2 = project2 / "session2.jsonl"
        session2.write_text(
            '{"type":"summary","summary":"Beta session"}\n'
            '{"type":"user","timestamp":"2025-01-02T00:00:00Z","message":{"role":"user","content":"Hi"}}\n'
        )

        grouped = {
            "project-alpha": [
                self._make_meta(
                    session1,
                    "Alpha session",
                    project_name="project-alpha",
                    project_path="project-alpha",
                ),
            ],
            "project-beta": [
                self._make_meta(
                    session2,
                    "Beta session",
                    project_name="project-beta",
                    project_path="project-beta",
                ),
            ],
        }

        choices = build_flat_choices(grouped, expand_chains=False)
        value_choices = [c for c in choices if not isinstance(c, questionary.Separator)]

        # Should have 2 choices (no separators)
        assert len(value_choices) == 2

        # Should include project markers in FormattedText
        for choice in value_choices:
            text = "".join(t[1] for t in choice.title)
            assert "[" in text and "]" in text

    def test_flat_mode_preserves_sort_order(self, tmp_path):
        """Test that flat mode preserves modification time sort order."""
        from ccutils.tui import build_flat_choices
        import questionary
        import time

        # Create sessions across multiple projects with specific mtimes
        project1 = tmp_path / "project-alpha"
        project1.mkdir()
        oldest = project1 / "oldest.jsonl"
        oldest.write_text('{"type":"user","message":{"content":"Oldest"}}\n')
        time.sleep(0.05)

        project2 = tmp_path / "project-beta"
        project2.mkdir()
        middle = project2 / "middle.jsonl"
        middle.write_text('{"type":"user","message":{"content":"Middle"}}\n')
        time.sleep(0.05)

        newest = project1 / "newest.jsonl"
        newest.write_text('{"type":"user","message":{"content":"Newest"}}\n')

        grouped = {
            "project-alpha": [
                self._make_meta(
                    newest,
                    "Newest session",
                    project_name="project-alpha",
                    project_path="project-alpha",
                ),
                self._make_meta(
                    oldest,
                    "Oldest session",
                    project_name="project-alpha",
                    project_path="project-alpha",
                ),
            ],
            "project-beta": [
                self._make_meta(
                    middle,
                    "Middle session",
                    project_name="project-beta",
                    project_path="project-beta",
                ),
            ],
        }

        choices = build_flat_choices(grouped, expand_chains=False)
        flat_value_choices = [
            c for c in choices if not isinstance(c, questionary.Separator)
        ]

        # Should be sorted by mtime (newest first)
        assert len(flat_value_choices) == 3
        texts = ["".join(t[1] for t in c.title) for c in flat_value_choices]

        # First should be newest, last should be oldest
        assert "Newest" in texts[0]
        assert "Oldest" in texts[-1]


class TestMasterIndex:
    """Tests for the archive index.

    Per-project index pages were deleted in C3 -- a project is a filter
    over one list, not a page, so `_generate_project_index` no longer
    exists and the test that exercised it went with it.
    """

    def test_index_renders_totals_and_dates(self, tmp_path):
        """The index states how much it covers.

        Claim: replaces a test of the deleted _generate_master_index. An index
        that does not say how many sessions it holds cannot be checked against
        the directory it describes -- which is how a silently-truncated archive
        would look identical to a complete one.
        """
        from ccutils.export.html import _render_archive_index

        out = tmp_path / "out"
        out.mkdir()
        _render_archive_index(out, [], session_count=7, project_count=3,
                              date_range="2026-01-01 - 2026-04-19")
        html_out = (out / "index.html").read_text()
        assert "7 sessions" in html_out
        assert "3 projects" in html_out
        assert "2026-01-01 - 2026-04-19" in html_out


class TestPrivateModeSanitizesJsonl:
    """--private must actually sanitize JSONL sessions, not just exit 0.

    Normalized JSONL loglines carry only type/timestamp/message -- no cwd --
    so generate_html must resolve cwd from the session file itself before
    calling the sanitizer. Effect-asserting per project convention.
    """

    def _session(self, tmp_path, leading_entries=()):
        cwd = "/Users/dev/workspace/secretproj"  # path-privacy: ignore
        entries = list(leading_entries) + [
            {
                "type": "user",
                "cwd": cwd,
                "sessionId": "priv-1",
                "timestamp": "2026-04-19T10:00:00Z",
                "message": {"role": "user", "content": "edit the file"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-04-19T10:00:05Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu-1",
                            "name": "Read",
                            "input": {"file_path": cwd + "/src/main.py"},
                        }
                    ],
                },
            },
        ]
        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text("\n".join(json.dumps(e) for e in entries))
        return jsonl, cwd

    def _rendered(self, tmp_path, jsonl, private):
        out_dir = tmp_path / ("out-private" if private else "out-plain")
        generate_html(jsonl, out_dir, private=private)
        return read_transcript(out_dir)

    def test_private_strips_cwd_prefix(self, tmp_path):
        jsonl, cwd = self._session(tmp_path)
        html_out = self._rendered(tmp_path, jsonl, private=True)
        assert cwd not in html_out
        assert "src/main.py" in html_out

    def test_private_works_with_leading_summary_line(self, tmp_path):
        jsonl, cwd = self._session(
            tmp_path,
            leading_entries=[{"type": "summary", "summary": "recap"}],
        )
        html_out = self._rendered(tmp_path, jsonl, private=True)
        assert cwd not in html_out
        assert "src/main.py" in html_out

    def test_non_private_keeps_paths(self, tmp_path):
        jsonl, cwd = self._session(tmp_path)
        html_out = self._rendered(tmp_path, jsonl, private=False)
        assert cwd in html_out


class TestPrivateFailsLoudWhenCwdUnresolvable:
    """--private must warn loudly (not silently no-op) when it cannot
    resolve a working directory to sanitize against."""

    def _loglines_no_cwd(self):
        # Normalized loglines carry no cwd -- the web/import shape.
        return [{
            "type": "user", "timestamp": "2026-04-19T10:00:00Z",
            "message": {"role": "user",
                        # synthetic placeholder -- the string this test asserts --private must scrub
                        "content": "see /Users/dev/x/secret.txt"},  # path-privacy: ignore
        }]

    def test_loglines_path_warns_when_no_cwd(self, tmp_path, capsys):
        generate_html(
            loglines=self._loglines_no_cwd(),
            output_dir=tmp_path / "out", private=True,
        )
        err = capsys.readouterr().err
        assert "private" in err.lower()
        assert "not sanitized" in err.lower() or "not be sanitized" in err.lower()

    def test_no_warning_when_not_private(self, tmp_path, capsys):
        generate_html(
            loglines=self._loglines_no_cwd(),
            output_dir=tmp_path / "out", private=False,
        )
        assert "not sanitized" not in capsys.readouterr().err.lower()

    def test_no_warning_when_cwd_resolves(self, tmp_path, capsys):
        jsonl = tmp_path / "s.jsonl"
        jsonl.write_text(json.dumps({
            "type": "user", "sessionId": "s1",
            "cwd": "/Users/dev/workspace/p",  # path-privacy: ignore
            "timestamp": "2026-04-19T10:00:00Z",
            "message": {"role": "user", "content": "hi"},
        }))
        generate_html(jsonl, tmp_path / "out", private=True)
        assert "not sanitized" not in capsys.readouterr().err.lower()


class TestNoThinkingIsHonouredByHtml:
    """`--no-thinking` must actually remove thinking from HTML output.

    Claim: this asserts the flag's EFFECT, not its acceptance. It shipped
    broken precisely because nothing did -- `generate_html` had no
    `include_thinking` parameter at all, so `ccutils local --format html
    --no-thinking` exited 0 and produced output byte-identical to the
    default, with every thinking block rendered. `export/markdown.py`
    honoured the same flag, so the two renderers silently disagreed.

    CLAUDE.md names this exact failure: "Exit-code-only CLI flag tests are
    insufficient -- assert the flag's actual effect (sanitized paths, absent
    thinking), not just acceptance."

    Delete these and the flag can go back to being decorative on the format
    most people use.
    """

    SECRET = "deliberating about the secret plan"

    def _session(self, tmp_path):
        jsonl = tmp_path / "thinking.jsonl"
        jsonl.write_text("\n".join(json.dumps(d) for d in [
            {"type": "user", "sessionId": "s1",
             "cwd": "/Users/dev/workspace/p",  # path-privacy: ignore
             "timestamp": "2026-04-19T10:00:00Z",
             "message": {"role": "user", "content": "go"}},
            {"type": "assistant", "sessionId": "s1",
             "timestamp": "2026-04-19T10:00:01Z",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [
                             {"type": "thinking", "thinking": self.SECRET},
                             {"type": "text", "text": "visible answer"},
                         ]}},
        ]))
        return jsonl

    def _rendered(self, out_dir):
        return "\n".join(
            p.read_text() for p in Path(out_dir).rglob("*.html")
        )

    def test_thinking_absent_when_excluded(self, tmp_path):
        out = tmp_path / "out"
        generate_html(self._session(tmp_path), out, include_thinking=False)
        assert self.SECRET not in self._rendered(out)

    def test_thinking_present_by_default(self, tmp_path):
        """Non-vacuity: if thinking never rendered, the test above passes
        for the wrong reason."""
        out = tmp_path / "out"
        generate_html(self._session(tmp_path), out)
        assert self.SECRET in self._rendered(out)

    def test_rest_of_transcript_survives(self, tmp_path):
        """Excluding thinking must not blank the message it came from."""
        out = tmp_path / "out"
        generate_html(self._session(tmp_path), out, include_thinking=False)
        assert "visible answer" in self._rendered(out)

    def test_cli_no_thinking_changes_html_output(self, tmp_path):
        """End to end through the CLI -- the path that actually shipped
        broken. Asserts the two runs DIFFER, which is what byte-identical
        output disproved."""
        from click.testing import CliRunner

        from ccutils.cli.local import local_cmd

        src = tmp_path / "projects" / "-Users-dev-p"
        src.mkdir(parents=True)
        session = self._session(tmp_path)
        session.rename(src / "s1.jsonl")

        runner = CliRunner()
        for name, args in (("with", []), ("without", ["--no-thinking"])):
            res = runner.invoke(local_cmd, [
                str(src / "s1.jsonl"), "--format", "html",
                "-o", str(tmp_path / name), *args,
            ])
            assert res.exit_code == 0, res.output

        assert self.SECRET in self._rendered(tmp_path / "with")
        assert self.SECRET not in self._rendered(tmp_path / "without")


class TestExportersAgreeOnThinking:
    """The HTML and markdown exporters must agree on what `--no-thinking` does.

    Claim: this is the invariant the shipped bug violated. `export/markdown.py`
    honoured the flag with a per-block skip; `generate_html` had no
    `include_thinking` parameter at all. Each exporter's own tests passed --
    the defect lived BETWEEN them, where nothing was looking.

    A per-exporter test cannot catch that by construction: it only ever sees
    one side. Delete this and the next flag added to one renderer and forgotten
    in the other is invisible again until someone reads the output by hand.
    """

    SECRET = "deliberating about the secret plan"

    def _loglines(self):
        return [
            {"type": "user", "sessionId": "s1",
             "cwd": "/Users/dev/workspace/p",  # path-privacy: ignore
             "timestamp": "2026-04-19T10:00:00Z",
             "message": {"role": "user", "content": "go"}},
            {"type": "assistant", "sessionId": "s1",
             "timestamp": "2026-04-19T10:00:01Z",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [
                             {"type": "thinking", "thinking": self.SECRET},
                             {"type": "text", "text": "visible answer"},
                         ]}},
        ]

    def _both(self, tmp_path, *, include_thinking):
        from ccutils.export.markdown import render_session_markdown

        out = tmp_path / f"out-{include_thinking}"
        generate_html(loglines=self._loglines(), output_dir=out,
                      include_thinking=include_thinking)
        html = "\n".join(p.read_text() for p in Path(out).rglob("*.html"))
        md = render_session_markdown(self._loglines(), title="t",
                                     include_thinking=include_thinking)
        return html, md

    def test_both_exclude_thinking_together(self, tmp_path):
        html, md = self._both(tmp_path, include_thinking=False)
        assert self.SECRET not in html, "HTML leaked thinking the markdown dropped"
        assert self.SECRET not in md, "markdown leaked thinking the HTML dropped"

    def test_both_include_thinking_together(self, tmp_path):
        """Non-vacuity control: without this, the test above passes if either
        exporter simply rendered nothing."""
        html, md = self._both(tmp_path, include_thinking=True)
        assert self.SECRET in html
        assert self.SECRET in md

    def test_visible_content_survives_in_both(self, tmp_path):
        html, md = self._both(tmp_path, include_thinking=False)
        assert "visible answer" in html
        assert "visible answer" in md


class TestSingleFileSessionOutput:
    """A session renders to ONE self-contained file. No siblings, no fetch.

    Claim: the multi-file layout is what made in-session search silently
    broken. `search.js` fetched `page-NNN.html` for every page, browsers
    block `fetch()` to file:// URLs, and the `.catch()` swallowed the error
    and incremented the counter -- so opening an export from disk and
    searching reported "Found 0 result(s) in N pages", indistinguishable
    from no matches.

    These assertions are layout-independent on purpose: they must survive
    any future restyling. Delete them and the export can quietly regain a
    dependency on being served over HTTP, which is not how anyone opens it.
    """

    def _session(self, tmp_path):
        jsonl = tmp_path / "sess.jsonl"
        jsonl.write_text("\n".join(json.dumps(d) for d in [
            {"type": "user", "sessionId": "s1",
             "cwd": "/Users/dev/workspace/p",  # path-privacy: ignore
             "timestamp": "2026-04-19T10:00:00Z",
             "message": {"role": "user", "content": "first prompt"}},
            {"type": "assistant", "sessionId": "s1",
             "timestamp": "2026-04-19T10:00:01Z",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [{"type": "text", "text": "an answer"}]}},
            {"type": "user", "sessionId": "s1",
             "timestamp": "2026-04-19T10:00:02Z",
             "message": {"role": "user", "content": "second prompt"}},
        ]))
        return jsonl

    def _render(self, tmp_path):
        out = tmp_path / "out"
        generate_html(self._session(tmp_path), out)
        files = sorted(p.name for p in out.glob("*"))
        return out, files

    def test_exactly_one_file_is_written(self, tmp_path):
        _, files = self._render(tmp_path)
        assert len([f for f in files if f.endswith(".html")]) == 1, files
        assert files == [f for f in files if f.endswith(".html")], (
            f"sibling assets written alongside the transcript: {files}")

    def test_no_pagination_files(self, tmp_path):
        _, files = self._render(tmp_path)
        assert not any(f.startswith("page-") for f in files), files

    def test_output_never_fetches(self, tmp_path):
        out, files = self._render(tmp_path)
        html = (out / files[0]).read_text()
        assert "fetch(" not in html
        assert "XMLHttpRequest" not in html
        assert "page-001.html" not in html

    def test_no_external_or_sibling_asset_references(self, tmp_path):
        out, files = self._render(tmp_path)
        html = (out / files[0]).read_text()
        assert "<script src=" not in html
        assert "<link rel=\"stylesheet\"" not in html
        assert "http://" not in html and "https://" not in html

    def test_every_message_carries_data_search(self, tmp_path):
        out, files = self._render(tmp_path)
        html = (out / files[0]).read_text()
        # Tight match: `class="message` alone also hits message-header and
        # message-content, which is how this assertion first passed for the
        # wrong reason.
        assert html.count('<div class="message ') == html.count("data-search="), (
            "filtering matches against data-search only; a message without "
            "one is invisible to the filter")

    def test_prompt_list_lists_every_user_prompt(self, tmp_path):
        out, files = self._render(tmp_path)
        html = (out / files[0]).read_text()
        assert "first prompt" in html and "second prompt" in html
        assert 'class="prompt-list"' in html

    def test_csp_allows_inline_only_via_hash(self, tmp_path):
        """Self-contained means inline blocks; inline must be hash-pinned.

        Claim: `unsafe-inline` would permit ANY inline script, including one
        an attacker buried in a transcript. A sha256 hash permits exactly
        the block we emitted and nothing else.
        """
        out, files = self._render(tmp_path)
        html = (out / files[0]).read_text()
        # Assert on the POLICY, not the document. transcript.css carries a
        # comment mentioning 'unsafe-inline' to explain why inline styles were
        # removed, and inlining the stylesheet puts that prose in the output --
        # a whole-document search matches the explanation, not the policy.
        m = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]+)"', html)
        assert m, "no CSP meta tag in the rendered document"
        csp = m.group(1)
        assert "unsafe-inline" not in csp
        assert "unsafe-hashes" not in csp
        assert csp.count("sha256-") == 2, f"expected style+script hashes, got: {csp}"

    def test_emitted_hashes_match_the_emitted_blocks(self, tmp_path):
        """The CSP hash must match the bytes actually in the document.

        Claim: this is the assertion that catches a silently dead page. Jinja
        autoescape turned `'` into `&#39;` inside the inline script -- the JS
        was corrupt AND the hash had been computed over the pre-escape text,
        so the browser would have hashed different bytes and blocked the
        script. The document still rendered, so every structural test passed;
        only the filter was dead. Checking "a sha256- appears in the CSP" is
        not enough -- it must be the RIGHT sha256.
        """
        import base64
        import hashlib

        out, files = self._render(tmp_path)
        doc = (out / files[0]).read_text()
        csp = re.search(r'content="([^"]*default-src[^"]*)"', doc).group(1)

        for tag in ("style", "script"):
            body = re.search(rf"<{tag}>(.*?)</{tag}>", doc, re.S).group(1)
            digest = base64.b64encode(
                hashlib.sha256(body.encode("utf-8")).digest()).decode("ascii")
            assert f"sha256-{digest}" in csp, (
                f"the {tag} block in the document does not match any hash in "
                f"the CSP -- the browser would block it")


class TestBatchIndexIsOneFilterableList:
    """Batch output is a flat directory: one index plus one file per session.

    Claim: the old layout was master index -> per-project index -> per-session
    directory -> paginated pages, with a separate search-index.js carrying the
    full text of every session. Five templates and two JS files to answer
    "which session was that". A project is a filter over one list, not a page,
    and cross-session full text belongs in the warehouse where DuckDB's fts
    already does it.

    Delete these and the archive can silently regrow a per-project page tree,
    which is what made `--private` leak through the search index (that build
    re-parsed every session from disk, unsanitized).
    """

    def _archive(self, tmp_path):
        src = tmp_path / "projects" / "proj-a"
        src.mkdir(parents=True)
        for name, text in (("sess-one", "alpha prompt"), ("sess-two", "beta prompt")):
            (src / f"{name}.jsonl").write_text("\n".join(json.dumps(d) for d in [
                {"type": "user", "sessionId": name,
                 "timestamp": "2026-04-19T10:00:00Z",
                 "message": {"role": "user", "content": text}},
                {"type": "assistant", "sessionId": name,
                 "timestamp": "2026-04-19T10:00:01Z",
                 "message": {"role": "assistant", "model": "claude-opus-5",
                             "content": [{"type": "text", "text": "answer"}]}},
                {"type": "summary", "summary": f"summary for {name}"},
            ]))
        out = tmp_path / "archive"
        from ccutils import generate_batch_html
        generate_batch_html(tmp_path / "projects", out)
        return out

    def test_flat_layout_index_plus_one_file_per_session(self, tmp_path):
        out = self._archive(tmp_path)
        names = sorted(p.name for p in out.iterdir())
        assert "index.html" in names
        assert not any((out / n).is_dir() for n in names), (
            f"batch output should be flat, found directories: {names}")
        assert len([n for n in names if n.endswith(".html")]) == 3  # index + 2

    def test_no_search_index_or_sibling_assets(self, tmp_path):
        out = self._archive(tmp_path)
        names = {p.name for p in out.iterdir()}
        assert "search-index.js" not in names
        assert "global_search.js" not in names
        assert "transcript.css" not in names
        assert "transcript.js" not in names

    def test_index_lists_every_session_with_data_search(self, tmp_path):
        out = self._archive(tmp_path)
        index = (out / "index.html").read_text()
        assert index.count("data-search=") == 2
        assert "sess-one" in index and "sess-two" in index
        assert "proj-a" in index

    def test_index_never_fetches(self, tmp_path):
        out = self._archive(tmp_path)
        index = (out / "index.html").read_text()
        assert "fetch(" not in index
        assert "<script src=" not in index
