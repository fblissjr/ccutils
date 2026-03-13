"""Tests for the convert CLI command."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from ccutils.cli import cli


class TestConvertCommandHTML:
    """Tests for convert command with HTML output (default)."""

    def test_converts_jsonl_to_html(self, sample_session_file, output_dir):
        """Convert JSONL to HTML output."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["convert", str(sample_session_file), "-o", str(output_dir)]
        )

        assert result.exit_code == 0
        assert "Output:" in result.output

    def test_missing_file_errors(self, tmp_path):
        """Missing input file gives error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["convert", str(tmp_path / "nonexistent.jsonl")])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Error" in result.output

    def test_include_json_flag(self, sample_session_file, output_dir):
        """--json flag copies original JSON to output."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "convert",
                str(sample_session_file),
                "-o",
                str(output_dir),
                "--json",
            ],
        )

        assert result.exit_code == 0
        assert "JSON:" in result.output

    def test_private_flag(self, sample_session_file, output_dir):
        """--private flag is accepted."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "convert",
                str(sample_session_file),
                "-o",
                str(output_dir),
                "--private",
            ],
        )

        assert result.exit_code == 0


class TestConvertCommandDuckDB:
    """Tests for convert command with DuckDB output."""

    def test_duckdb_simple(self, sample_session_file, output_dir):
        """Convert to simple DuckDB."""
        db_path = output_dir / "test.duckdb"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "convert",
                str(sample_session_file),
                "--format",
                "duckdb",
                "-o",
                str(db_path),
            ],
        )

        assert result.exit_code == 0
        assert "Exported to" in result.output
        assert db_path.exists()

    def test_duckdb_star(self, sample_session_file, output_dir):
        """Convert to star DuckDB."""
        db_path = output_dir / "test.duckdb"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "convert",
                str(sample_session_file),
                "--format",
                "duckdb-star",
                "-o",
                str(db_path),
            ],
        )

        assert result.exit_code == 0
        assert "Exported to" in result.output
        assert db_path.exists()

    def test_include_thinking_flag(self, sample_session_file, output_dir):
        """--include-thinking flag is accepted for DuckDB."""
        db_path = output_dir / "test.duckdb"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "convert",
                str(sample_session_file),
                "--format",
                "duckdb",
                "-o",
                str(db_path),
                "--include-thinking",
            ],
        )

        assert result.exit_code == 0


class TestConvertCommandJSON:
    """Tests for convert command with JSON output."""

    def test_json_simple(self, sample_session_file, output_dir):
        """Convert to simple JSON export."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "convert",
                str(sample_session_file),
                "--format",
                "json",
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert "Exported to" in result.output
        # Simple JSON creates sessions.json in the directory
        json_file = output_dir / "sessions.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text())
        assert data["schema_type"] == "simple"
        assert "tables" in data

    def test_json_star(self, sample_session_file, output_dir):
        """Convert to star JSON export."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "convert",
                str(sample_session_file),
                "--format",
                "json-star",
                "-o",
                str(output_dir / "star_out"),
            ],
        )

        assert result.exit_code == 0
        assert "Exported to" in result.output
        # Star JSON creates a directory structure
        star_dir = output_dir / "star_out"
        assert star_dir.exists()
        assert (star_dir / "meta.json").exists()

    def test_schema_override(self, sample_session_file, output_dir):
        """--schema flag overrides format inference."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "convert",
                str(sample_session_file),
                "--format",
                "json",
                "--schema",
                "star",
                "-o",
                str(output_dir / "star_out"),
            ],
        )

        assert result.exit_code == 0
        # Should use star schema even though format was "json" not "json-star"
        star_dir = output_dir / "star_out"
        assert star_dir.exists()


class TestConvertCommandURL:
    """Tests for convert command with URL input."""

    def test_url_input_fetches(self, output_dir, sample_session_file):
        """URL input triggers fetch."""
        session_content = sample_session_file.read_text()

        mock_response = MagicMock()
        mock_response.text = session_content
        mock_response.raise_for_status = MagicMock()

        with patch("ccutils.cli.utils.httpx.get", return_value=mock_response):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "convert",
                    "https://example.com/session.jsonl",
                    "-o",
                    str(output_dir),
                ],
            )

        assert result.exit_code == 0
        assert "Fetching" in result.output

    def test_url_input_uses_url_name_as_project(self, output_dir, sample_session_file):
        """URL input should use URL filename as project_name, not temp dir."""
        import duckdb

        session_content = sample_session_file.read_text()

        mock_response = MagicMock()
        mock_response.text = session_content
        mock_response.raise_for_status = MagicMock()

        db_path = output_dir / "url_test.duckdb"
        with patch("ccutils.cli.utils.httpx.get", return_value=mock_response):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "convert",
                    "https://example.com/my-session.jsonl",
                    "--format",
                    "duckdb",
                    "-o",
                    str(db_path),
                ],
            )

        assert result.exit_code == 0
        conn = duckdb.connect(str(db_path))
        project_name = conn.execute("SELECT project_name FROM sessions").fetchone()[0]
        conn.close()
        # Should be "my-session" (from URL), not a temp dir name
        assert project_name == "my-session"

    def test_url_network_error(self, output_dir):
        """URL fetch network error gives friendly message."""
        import httpx

        with patch(
            "ccutils.cli.utils.httpx.get",
            side_effect=httpx.RequestError("Connection failed"),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "convert",
                    "https://example.com/session.jsonl",
                    "-o",
                    str(output_dir),
                ],
            )

        assert result.exit_code != 0
