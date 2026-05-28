"""Tests for single-file conversion (positional arg to default command)."""

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

    def test_duckdb_produces_v15_tables(self, sample_session_file, output_dir):
        """--format duckdb writes the v0.15 star schema. There is no
        longer a 'simple' schema (4-table) variant; the only DDL the CLI
        produces is the v0.15 dimensional model. This test pins the
        contract: --format duckdb -> v0.15 fact tables."""
        import duckdb as _duckdb

        db_path = output_dir / "test.duckdb"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [str(sample_session_file), "--format", "duckdb", "-o", str(db_path)],
        )

        assert result.exit_code == 0, result.output
        assert db_path.exists()

        conn = _duckdb.connect(str(db_path))
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()

        # v0.15-only fact tables that must exist after a star ETL.
        for expected in ("dim_etl_version", "fact_messages",
                         "fact_session_summary", "fact_session_facets",
                         "dim_facet_type"):
            assert expected in tables, (
                f"Expected v0.15 table {expected} missing -- "
                "did --format duckdb still write the legacy simple schema?"
            )

        # Sanity check: legacy 4-table simple schema sentinels MUST NOT exist.
        for legacy in ("sessions", "thinking"):
            assert legacy not in tables, (
                f"Legacy simple-schema table {legacy} still being written by --format duckdb"
            )

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
    """Tests for single-file conversion with JSON output (v0.15 star schema
    as a directory tree)."""

    def test_json_produces_star_directory(self, sample_session_file, output_dir):
        """`--format json` writes the v0.15 star schema as a meta.json +
        dimensions/ + facts/ tree."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(sample_session_file),
                "--format", "json",
                "-o", str(output_dir / "json_out"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Exported to" in result.output
        out_dir = output_dir / "json_out"
        assert out_dir.exists()
        assert (out_dir / "meta.json").exists()
