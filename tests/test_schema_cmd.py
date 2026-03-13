"""Tests for the schema CLI command."""

import json
from pathlib import Path

from click.testing import CliRunner

from ccutils.cli import cli


class TestSchemaCommandSingleFile:
    """Tests for schema command with single file input."""

    def test_json_file_shows_format(self, tmp_path):
        """Schema command displays format for a JSON file."""
        data = {"name": "test", "count": 42, "items": [1, 2, 3]}
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data))

        runner = CliRunner()
        result = runner.invoke(cli, ["schema", str(json_file)])

        assert result.exit_code == 0
        assert "Format: JSON" in result.output

    def test_json_file_shows_size(self, tmp_path):
        """Schema command displays file size."""
        data = {"key": "value"}
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data))

        runner = CliRunner()
        result = runner.invoke(cli, ["schema", str(json_file)])

        assert result.exit_code == 0
        assert "Size:" in result.output
        assert "KB" in result.output

    def test_json_file_shows_keys(self, tmp_path):
        """Schema command shows object keys in output."""
        data = {"name": "test", "count": 42}
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data))

        runner = CliRunner()
        result = runner.invoke(cli, ["schema", str(json_file)])

        assert result.exit_code == 0
        assert "name" in result.output
        assert "count" in result.output

    def test_jsonl_file_shows_line_count(self, tmp_path):
        """Schema command shows line count for JSONL files."""
        jsonl_file = tmp_path / "test.jsonl"
        lines = [json.dumps({"id": i, "val": f"item-{i}"}) for i in range(5)]
        jsonl_file.write_text("\n".join(lines) + "\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["schema", str(jsonl_file)])

        assert result.exit_code == 0
        assert "Lines: 5" in result.output
        assert "Format: JSONL" in result.output

    def test_json_flag_outputs_raw_json(self, tmp_path):
        """--json flag outputs parseable JSON."""
        data = {"key": "value"}
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data))

        runner = CliRunner()
        result = runner.invoke(cli, ["schema", str(json_file), "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "schema" in parsed
        assert parsed["format"] == "json"

    def test_samples_option(self, tmp_path):
        """--samples option controls array sampling depth."""
        data = [{"id": i} for i in range(100)]
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data))

        runner = CliRunner()
        result = runner.invoke(cli, ["schema", str(json_file), "-s", "2"])

        assert result.exit_code == 0
        assert "100 items" in result.output


class TestSchemaCommandDirectory:
    """Tests for schema command with directory input."""

    def test_directory_inspects_all_json_files(self, tmp_path):
        """Schema command inspects all JSON files in a directory."""
        (tmp_path / "a.json").write_text(json.dumps({"x": 1}))
        (tmp_path / "b.json").write_text(json.dumps({"y": 2}))

        runner = CliRunner()
        result = runner.invoke(cli, ["schema", str(tmp_path)])

        assert result.exit_code == 0
        assert "a.json" in result.output
        assert "b.json" in result.output

    def test_directory_json_flag(self, tmp_path):
        """--json flag outputs combined JSON for directory."""
        (tmp_path / "a.json").write_text(json.dumps({"x": 1}))
        (tmp_path / "b.json").write_text(json.dumps({"y": 2}))

        runner = CliRunner()
        result = runner.invoke(cli, ["schema", str(tmp_path), "--json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "a.json" in parsed
        assert "b.json" in parsed

    def test_file_flag_inspects_specific_file(self, tmp_path):
        """--file flag inspects only the specified file in a directory."""
        (tmp_path / "a.json").write_text(json.dumps({"x": 1}))
        (tmp_path / "b.json").write_text(json.dumps({"y": 2}))

        runner = CliRunner()
        result = runner.invoke(cli, ["schema", str(tmp_path), "-f", "a.json"])

        assert result.exit_code == 0
        assert "x" in result.output
        # Should not contain b.json's content
        assert "FILE: b.json" not in result.output

    def test_file_flag_nonexistent_file_errors(self, tmp_path):
        """--file flag with nonexistent file gives error."""
        (tmp_path / "a.json").write_text(json.dumps({"x": 1}))

        runner = CliRunner()
        result = runner.invoke(cli, ["schema", str(tmp_path), "-f", "missing.json"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Error" in result.output

    def test_empty_directory_errors(self, tmp_path):
        """Empty directory (no JSON files) gives error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["schema", str(tmp_path)])

        assert result.exit_code != 0
        assert "No JSON files" in result.output
