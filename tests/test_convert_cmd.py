"""Tests for single-file conversion (positional arg to default command)."""

import json

from click.testing import CliRunner

from ccutils.cli import cli


def _session_with_subagents(tmp_path, n_agents=2):
    """A parent transcript with agents in the real on-disk layout.

    `<project>/<uuid>.jsonl` plus `<project>/<uuid>/subagents/agent-*.jsonl`
    -- see docs/JSONL_CONTRACT.md claim 6. Agent entries carry the PARENT's
    sessionId, which is why identity comes from the filename.
    """
    project = tmp_path / "-home-user-projects-demo"
    project.mkdir(parents=True, exist_ok=True)

    def entry(**over):
        base = {
            "type": "user", "uuid": "u1", "sessionId": "sess-parent",
            "timestamp": "2026-01-15T10:00:00.000Z", "cwd": "/home/user/demo",
            "message": {"role": "user", "content": "do the thing"},
        }
        base.update(over)
        return json.dumps(base) + "\n"

    parent = project / "sess-parent.jsonl"
    parent.write_text(entry())

    agents = project / "sess-parent" / "subagents"
    agents.mkdir(parents=True, exist_ok=True)
    for i in range(n_agents):
        (agents / f"agent-a{i}.jsonl").write_text(
            entry(uuid=f"au{i}", agentId=f"a{i}", isSidechain=True,
                  message={"role": "user", "content": f"subtask {i}"})
        )
    return parent


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

    def test_writes_an_index_the_browser_can_open(
        self, sample_session_file, output_dir
    ):
        """--open targets <output>/index.html, so one must exist.

        The single-file branch wrote only <stem>.html, while
        maybe_open_browser opens <output>/index.html unconditionally -- and
        with no -o the browser is opened automatically, so the default path
        was the broken one. This is the README's own quick-start example.

        The previous test for this path asserted exit code 0 and the string
        "Output:", which the broken behaviour satisfied exactly.
        """
        runner = CliRunner()
        result = runner.invoke(
            cli, [str(sample_session_file), "-o", str(output_dir)]
        )

        assert result.exit_code == 0, result.output
        assert (output_dir / "index.html").exists()

    def test_single_file_attaches_its_subagents(self, tmp_path, output_dir):
        """The headline invocation must not silently drop subagents.

        `_convert_file` built no agent_map, so the discovery fix in 0.19.1
        never reached this path: a session with agent transcripts beside it
        exported one file and said nothing.
        """
        parent = _session_with_subagents(tmp_path, n_agents=2)
        runner = CliRunner()
        result = runner.invoke(cli, [str(parent), "-o", str(output_dir)])

        assert result.exit_code == 0, result.output
        written = {f.name for f in output_dir.glob("*.html")}
        assert "sess-parent.html" in written
        assert "agent-a0.html" in written
        assert "agent-a1.html" in written

    def test_no_subagents_flag_actually_excludes_them(self, tmp_path, output_dir):
        """--no-subagents was accepted on this path and did nothing."""
        parent = _session_with_subagents(tmp_path, n_agents=2)
        runner = CliRunner()
        result = runner.invoke(
            cli, [str(parent), "-o", str(output_dir), "--no-subagents"]
        )

        assert result.exit_code == 0, result.output
        written = {f.name for f in output_dir.glob("*.html")}
        assert written == {"sess-parent.html", "index.html"}

    def test_missing_file_errors(self, tmp_path):
        """Missing input file gives error."""
        runner = CliRunner()
        result = runner.invoke(cli, [str(tmp_path / "nonexistent.jsonl")])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Error" in result.output

    def test_private_flag_accepted_on_html(self, sample_session_file, output_dir):
        """--private is honored on the HTML path (PathSanitizer is wired
        into generate_html)."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [str(sample_session_file), "-o", str(output_dir), "--private"],
        )

        assert result.exit_code == 0


class TestHonestyGuards:
    """v0.15 doesn't yet wire --private through the ETL. Rather than
    silently accepting the flag and producing a non-sanitized database,
    the CLI fails loud. --no-thinking IS wired -- see tests/test_no_thinking_v15.py."""

    def test_private_rejected_on_duckdb(self, sample_session_file, output_dir):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [str(sample_session_file), "--format", "duckdb",
             "-o", str(output_dir / "test.duckdb"), "--private"],
        )
        assert result.exit_code != 0
        assert "private" in result.output.lower()
        assert "v0.15" in result.output or "html" in result.output.lower()

    def test_private_rejected_on_json(self, sample_session_file, output_dir):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [str(sample_session_file), "--format", "json",
             "-o", str(output_dir / "json_out"), "--private"],
        )
        assert result.exit_code != 0
        assert "private" in result.output.lower()


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

    def test_empty_session_file_is_skipped_not_crashed(self, tmp_path, output_dir):
        """A session file with no valid entries must be skipped with a
        reported failure, not abort the export with an unhandled ValueError
        from write_session_to_parquet. The batch `all` path already isolates
        per-session failures; the single-file `local` path must too."""
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        db_path = output_dir / "out.duckdb"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [str(empty), "--format", "duckdb", "-o", str(db_path)],
        )

        # No unhandled exception should escape the command.
        assert result.exception is None, result.exception
        assert result.exit_code == 0, result.output
        assert db_path.exists()

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
