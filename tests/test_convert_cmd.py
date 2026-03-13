"""Tests for single-file conversion (positional arg to default command)."""

import json
from pathlib import Path

from click.testing import CliRunner

from ccutils.cli import cli


class TestFileConversionHTML:
    """Tests for single-file conversion with HTML output (default)."""

    def test_converts_jsonl_to_html(self, sample_session_file, output_dir):
        """Convert JSONL to HTML output via positional arg."""
        runner = CliRunner()
        result = runner.invoke(
            cli, [str(sample_session_file), "-o", str(output_dir)]
        )

        assert result.exit_code == 0
        assert "Output:" in result.output

    def test_convert_alias_still_works(self, sample_session_file, output_dir):
        """The 'convert' alias still works for backwards compatibility."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["convert", str(sample_session_file), "-o", str(output_dir)]
        )

        assert result.exit_code == 0
        assert "Output:" in result.output

    def test_missing_file_errors(self, tmp_path):
        """Missing input file gives error."""
        runner = CliRunner()
        result = runner.invoke(cli, [str(tmp_path / "nonexistent.jsonl")])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Error" in result.output

    def test_private_flag(self, sample_session_file, output_dir):
        """--private flag is accepted."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [str(sample_session_file), "-o", str(output_dir), "--private"],
        )

        assert result.exit_code == 0


class TestFileConversionDuckDB:
    """Tests for single-file conversion with DuckDB output."""

    def test_duckdb_simple(self, sample_session_file, output_dir):
        """Convert to simple DuckDB."""
        db_path = output_dir / "test.duckdb"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [str(sample_session_file), "--format", "duckdb", "-o", str(db_path)],
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
            [str(sample_session_file), "--format", "duckdb-star", "-o", str(db_path)],
        )

        assert result.exit_code == 0
        assert "Exported to" in result.output
        assert db_path.exists()

    def test_thinking_included_by_default(self, sample_session_file, output_dir):
        """Thinking blocks are included by default in DuckDB export."""
        db_path = output_dir / "test.duckdb"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [str(sample_session_file), "--format", "duckdb", "-o", str(db_path)],
        )

        assert result.exit_code == 0

    def test_no_thinking_flag(self, sample_session_file, output_dir):
        """--no-thinking flag is accepted for DuckDB."""
        db_path = output_dir / "test.duckdb"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(sample_session_file),
                "--format", "duckdb",
                "-o", str(db_path),
                "--no-thinking",
            ],
        )

        assert result.exit_code == 0


class TestFileConversionJSON:
    """Tests for single-file conversion with JSON output."""

    def test_json_simple(self, sample_session_file, output_dir):
        """Convert to simple JSON export."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [str(sample_session_file), "--format", "json", "-o", str(output_dir)],
        )

        assert result.exit_code == 0
        assert "Exported to" in result.output
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
                str(sample_session_file),
                "--format", "json-star",
                "-o", str(output_dir / "star_out"),
            ],
        )

        assert result.exit_code == 0
        assert "Exported to" in result.output
        star_dir = output_dir / "star_out"
        assert star_dir.exists()
        assert (star_dir / "meta.json").exists()
