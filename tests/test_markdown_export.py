"""Tests for the markdown exporter (--format markdown)."""

import json
from pathlib import Path

from click.testing import CliRunner

from ccutils.cli import cli
from ccutils.export.markdown import generate_batch_markdown, generate_markdown


def _read_md(md_path):
    return Path(md_path).read_text(encoding="utf-8")


class TestGenerateMarkdown:
    """Unit tests for generate_markdown on a single session file."""

    def test_creates_md_file_in_output_dir(self, sample_session_file, output_dir):
        md_path = generate_markdown(sample_session_file, output_dir)
        assert md_path.exists()
        assert md_path.suffix == ".md"
        assert md_path.parent == output_dir

    def test_explicit_md_output_path(self, sample_session_file, output_dir):
        target = output_dir / "transcript.md"
        md_path = generate_markdown(sample_session_file, target)
        assert md_path == target
        assert target.exists()

    def test_session_header(self, sample_session_file, output_dir):
        content = _read_md(generate_markdown(sample_session_file, output_dir))
        # Title heading present
        assert content.startswith("# ")
        # Session id and date from the first logline
        assert "session-123" in content
        assert "2025-01-15" in content

    def test_user_heading_and_text(self, sample_session_file, output_dir):
        content = _read_md(generate_markdown(sample_session_file, output_dir))
        assert "## User" in content
        assert "Help me write a hello world program" in content

    def test_tool_result_only_messages_get_no_user_heading(
        self, sample_session_file, output_dir
    ):
        content = _read_md(generate_markdown(sample_session_file, output_dir))
        # The fixture has exactly one real user prompt; the two tool_result
        # carrier messages must not produce their own "## User" headings.
        assert content.count("## User") == 1

    def test_assistant_heading_and_text(self, sample_session_file, output_dir):
        content = _read_md(generate_markdown(sample_session_file, output_dir))
        assert "## Assistant" in content
        assert "I'll create that for you." in content
        assert "Done! I've created hello.py with a hello world program." in content

    def test_tool_use_details_block(self, sample_session_file, output_dir):
        content = _read_md(generate_markdown(sample_session_file, output_dir))
        assert "<details>" in content
        assert "</details>" in content
        assert "<summary>Write: /home/user/project/hello.py</summary>" in content
        # Tool input rendered as fenced JSON
        assert '"file_path": "/home/user/project/hello.py"' in content

    def test_tool_result_inside_details_block(self, sample_session_file, output_dir):
        content = _read_md(generate_markdown(sample_session_file, output_dir))
        # The Write tool's result is paired into the same details block
        details_start = content.index("<summary>Write:")
        details_end = content.index("</details>", details_start)
        assert "File written successfully" in content[details_start:details_end]

    def test_thinking_included_by_default(self, sample_session_file, output_dir):
        content = _read_md(generate_markdown(sample_session_file, output_dir))
        assert "The file was created. Let me verify it." in content
        # Rendered as a blockquote
        assert "> The file was created. Let me verify it." in content

    def test_thinking_excluded_with_flag(self, sample_session_file, output_dir):
        md_path = generate_markdown(
            sample_session_file, output_dir, include_thinking=False
        )
        content = _read_md(md_path)
        assert "The file was created. Let me verify it." not in content

    def test_private_sanitizes_paths(self, sample_session_file, output_dir):
        md_path = generate_markdown(sample_session_file, output_dir, private=True)
        content = _read_md(md_path)
        # cwd is /home/user/project, so paths under it become relative
        assert "/home/user/project/hello.py" not in content
        assert "hello.py" in content

    def test_header_and_private_survive_leading_summary_line(self, output_dir):
        """A leading summary entry (no sessionId/cwd) must not blank the
        header metadata or disable --private sanitization."""
        session_file = output_dir / "summary-first.jsonl"
        session_file.write_text(
            json.dumps({"type": "summary", "summary": "A test session"})
            + "\n"
            + json.dumps(
                {
                    "type": "user",
                    "uuid": "u1",
                    "sessionId": "session-summary-first",
                    "timestamp": "2025-02-01T00:00:00.000Z",
                    "cwd": "/home/user/project",
                    "message": {"role": "user", "content": "read the file"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "assistant",
                    "uuid": "a1",
                    "sessionId": "session-summary-first",
                    "timestamp": "2025-02-01T00:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "Read",
                                "input": {
                                    "file_path": "/home/user/project/notes.txt"
                                },
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        content = _read_md(generate_markdown(session_file, output_dir, private=True))
        assert "session-summary-first" in content
        assert "/home/user/project/notes.txt" not in content
        assert "notes.txt" in content

    def test_long_tool_result_truncated(self, output_dir):
        session_file = output_dir / "long-result.jsonl"
        long_output = "x" * 10000
        session_file.write_text(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "u1",
                    "sessionId": "s-long",
                    "timestamp": "2025-01-01T00:00:00.000Z",
                    "cwd": "/home/user/project",
                    "message": {"role": "user", "content": "run it"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "assistant",
                    "uuid": "a1",
                    "sessionId": "s-long",
                    "timestamp": "2025-01-01T00:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "Bash",
                                "input": {"command": "yes | head"},
                            }
                        ],
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "user",
                    "uuid": "u2",
                    "sessionId": "s-long",
                    "timestamp": "2025-01-01T00:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": long_output,
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        content = _read_md(generate_markdown(session_file, output_dir))
        assert "truncated" in content
        assert long_output not in content

    def test_fences_grow_past_backticks_in_content(self, output_dir):
        session_file = output_dir / "backticks.jsonl"
        tricky = "```python\nprint('hi')\n```"
        session_file.write_text(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "u1",
                    "sessionId": "s-bt",
                    "timestamp": "2025-01-01T00:00:00.000Z",
                    "message": {"role": "user", "content": "show me"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "assistant",
                    "uuid": "a1",
                    "sessionId": "s-bt",
                    "timestamp": "2025-01-01T00:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "Bash",
                                "input": {"command": "cat file.md"},
                            }
                        ],
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "user",
                    "uuid": "u2",
                    "sessionId": "s-bt",
                    "timestamp": "2025-01-01T00:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": tricky,
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        content = _read_md(generate_markdown(session_file, output_dir))
        # The fence around a result containing ``` must be longer than ```
        assert "````" in content


class TestGenerateBatchMarkdown:
    """Batch export mirrors the per-project directory tree, no index pages."""

    def test_writes_per_project_md_files(self, mock_projects_dir, output_dir):
        stats = generate_batch_markdown(mock_projects_dir, output_dir)
        md_files = list(output_dir.rglob("*.md"))
        assert len(md_files) == 1
        # Same per-project layout as HTML batch export
        assert md_files[0].parent.parent == output_dir
        assert stats["total_sessions"] == 1
        assert stats["failed_sessions"] == []

    def test_no_index_pages(self, mock_projects_dir, output_dir):
        generate_batch_markdown(mock_projects_dir, output_dir)
        assert list(output_dir.rglob("index.html")) == []
        assert list(output_dir.rglob("index.md")) == []


class TestMarkdownCli:
    """CLI-level wiring of --format markdown."""

    def test_local_format_markdown(self, sample_session_file, output_dir):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [str(sample_session_file), "--format", "markdown", "-o", str(output_dir)],
        )
        assert result.exit_code == 0, result.output
        md_files = list(output_dir.rglob("*.md"))
        assert len(md_files) == 1
        assert "## User" in md_files[0].read_text(encoding="utf-8")

    def test_local_format_markdown_no_thinking(self, sample_session_file, output_dir):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(sample_session_file),
                "--format",
                "markdown",
                "-o",
                str(output_dir),
                "--no-thinking",
            ],
        )
        assert result.exit_code == 0, result.output
        md_files = list(output_dir.rglob("*.md"))
        content = md_files[0].read_text(encoding="utf-8")
        assert "The file was created. Let me verify it." not in content

    def test_local_format_markdown_private_works(
        self, sample_session_file, output_dir
    ):
        """--private must NOT be rejected for markdown (render-only format,
        sanitized on the render path like html) -- and it must actually
        sanitize, not just be accepted."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(sample_session_file),
                "--format",
                "markdown",
                "-o",
                str(output_dir),
                "--private",
            ],
        )
        assert result.exit_code == 0, result.output
        md_files = list(output_dir.rglob("*.md"))
        content = md_files[0].read_text(encoding="utf-8")
        assert "/home/user/project/hello.py" not in content
        assert "hello.py" in content

    def test_all_format_markdown_private_works(self, mock_projects_dir, output_dir):
        """The all command's --private guard must exempt markdown too."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "all",
                "-s",
                str(mock_projects_dir),
                "-o",
                str(output_dir),
                "--format",
                "markdown",
                "--private",
            ],
        )
        assert result.exit_code == 0, result.output
        md_files = list(output_dir.rglob("*.md"))
        assert len(md_files) == 1
        content = md_files[0].read_text(encoding="utf-8")
        assert "/home/user/project/hello.py" not in content

    def test_all_format_markdown(self, mock_projects_dir, output_dir):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "all",
                "-s",
                str(mock_projects_dir),
                "-o",
                str(output_dir),
                "--format",
                "markdown",
            ],
        )
        assert result.exit_code == 0, result.output
        md_files = list(output_dir.rglob("*.md"))
        assert len(md_files) == 1
        assert list(output_dir.rglob("index.html")) == []
