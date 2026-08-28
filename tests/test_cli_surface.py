"""The 0.20.0 command surface.

One command does the conversion work. `ccutils` with no positional and no
`--source` opens the picker; with paths it converts those; with `--source`
it walks everything under it. `import` and `open` remain as subcommands.

`local`, `all`, `convert`, `web` and `schema` are gone, with no aliases --
decided 2026-08-05, and 0.20.0 is the last release where breaking the CLI
is free. `web` and `schema` are deleted outright rather than renamed: one
depended on an undocumented Claude API plus a keychain token, the other was
a generic JSON introspector, and neither had a path to the warehouse.

These tests assert the surface. Behaviour of each mode is covered by
test_convert_cmd.py and test_all.py.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from ccutils.cli import cli


def _session(dir_path, name="sess-a"):
    dir_path.mkdir(parents=True, exist_ok=True)
    f = dir_path / f"{name}.jsonl"
    f.write_text(
        json.dumps({
            "type": "user", "uuid": "u1", "sessionId": name,
            "timestamp": "2026-01-15T10:00:00.000Z", "cwd": "/home/user/demo",
            "message": {"role": "user", "content": "hello there"},
        }) + "\n"
    )
    return f


class TestRemovedCommands:
    """A hard break means the old names fail, not silently redirect."""

    # `convert` is not in this list: the merged command is NAMED convert, so
    # it is no longer a redundant alias of `local` -- it is the one
    # conversion command. What the restructure removes is the duplicate, and
    # the local/all split that let the two paths drift apart.
    REMOVED = ("local", "all", "web", "schema")

    def test_removed_names_fail_loudly(self):
        """A removed command must not silently redirect.

        `--help` is the case that matters: a DefaultGroup forwards an
        unknown token to the default command, so `ccutils local --help`
        printed the conversion help and exited 0. That reads as "local still
        works" to anyone checking.
        """
        runner = CliRunner()
        for name in self.REMOVED:
            result = runner.invoke(cli, [name, "--help"])
            assert result.exit_code != 0, f"`ccutils {name} --help` still works"
            assert "removed in 0.20.0" in result.output, name

    def test_removed_names_say_what_to_use_instead(self):
        runner = CliRunner()
        assert "ccutils --source" in runner.invoke(cli, ["all"]).output
        assert "no arguments" in runner.invoke(cli, ["local"]).output
        # web and schema were deleted, not renamed -- say so rather than
        # pointing at something that does not do the same job.
        assert "no replacement" in runner.invoke(cli, ["web"]).output
        assert "no replacement" in runner.invoke(cli, ["schema"]).output

    def test_help_lists_only_the_surviving_subcommands(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        commands = result.output.split("Commands:")[-1]
        for kept in ("convert", "import", "open"):
            assert kept in commands
        for gone in self.REMOVED:
            assert f"  {gone}" not in commands, f"{gone} still advertised"


class TestModeDispatch:
    """Which mode runs is decided by the positional and --source."""

    def test_paths_convert_those_files(self, tmp_path):
        f = _session(tmp_path / "proj")
        out = tmp_path / "out"
        result = CliRunner().invoke(cli, [str(f), "-o", str(out)])

        assert result.exit_code == 0, result.output
        assert (out / "sess-a.html").exists()

    def test_source_walks_everything_under_it(self, tmp_path):
        _session(tmp_path / "projects" / "-home-user-projects-demo", "s1")
        _session(tmp_path / "projects" / "-home-user-projects-other", "s2")
        out = tmp_path / "out"
        result = CliRunner().invoke(
            cli, ["--source", str(tmp_path / "projects"), "-o", str(out)]
        )

        assert result.exit_code == 0, result.output
        written = {f.name for f in out.rglob("*.html")}
        assert "s1.html" in written and "s2.html" in written

    def test_paths_and_source_together_are_a_usage_error(self, tmp_path):
        f = _session(tmp_path / "proj")
        result = CliRunner().invoke(
            cli, [str(f), "--source", str(tmp_path), "-o", str(tmp_path / "o")]
        )

        assert result.exit_code != 0
        assert "--source" in result.output


class TestOpenCommand:
    """`ccutils open` launches a SQL UI; it never builds anything."""

    def test_open_exists_and_documents_itself(self):
        """Asserted on the usage line, not on the word "duckdb".

        A DefaultGroup forwards an unknown token to the default command, so
        `ccutils open --help` printed the CONVERSION help -- which mentions
        duckdb among its formats -- and a laxer assertion here passed while
        the command did not exist.
        """
        result = CliRunner().invoke(cli, ["open", "--help"])

        assert result.exit_code == 0
        assert "Usage: cli open" in result.output

    def test_open_reports_a_missing_warehouse_clearly(self, tmp_path):
        """Asserted on the warehouse wording specifically.

        "not found" alone was satisfied by the default command's
        "File not found: open" when `open` was being parsed as a path.
        """
        result = CliRunner().invoke(
            cli, ["open", "-o", str(tmp_path / "nothing-here")]
        )

        assert result.exit_code != 0
        combined = (result.output + str(result.exception or "")).lower()
        assert "warehouse" in combined

    def test_open_does_not_build_anything(self, tmp_path):
        """It launches a UI; the README's no-SQL-UI boundary holds."""
        target = tmp_path / "nothing-here"
        CliRunner().invoke(cli, ["open", "-o", str(target)])

        assert not target.exists()


class TestIgnoredFlagsFailLoudly:
    """A flag that cannot do anything must say so, not exit 0.

    `--private` already worked this way on duckdb/json and is the model.
    These three did not: `--embed` was accepted and ignored on every render
    format and on the single-file path in ANY format, and `--llm-facets`
    on a render format went further -- it demanded an ANTHROPIC_API_KEY and
    exited 2 before doing any work, blocking a valid HTML export on a
    credential that format would never use.
    """

    def test_embed_rejected_on_html(self, tmp_path):
        f = _session(tmp_path / "proj")
        result = CliRunner().invoke(
            cli, [str(f), "--format", "html", "--embed", "-o", str(tmp_path / "o")]
        )

        assert result.exit_code != 0
        assert "--embed" in result.output

    def test_embed_rejected_on_markdown(self, tmp_path):
        f = _session(tmp_path / "proj")
        result = CliRunner().invoke(
            cli, [str(f), "--format", "markdown", "--embed", "-o", str(tmp_path / "o")]
        )

        assert result.exit_code != 0
        assert "--embed" in result.output

    def test_llm_facets_rejected_on_html_without_asking_for_credentials(
        self, tmp_path, monkeypatch
    ):
        """The rejection must not depend on having a key.

        With a key present the old code sailed past the check and silently
        ignored the flag; without one it failed for the wrong reason. Assert
        the format is what is rejected, with a key in the environment.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-used-here")
        f = _session(tmp_path / "proj")
        result = CliRunner().invoke(
            cli, [str(f), "--format", "html", "--llm-facets", "-o", str(tmp_path / "o")]
        )

        assert result.exit_code != 0
        assert "--llm-facets" in result.output
        assert "api key" not in result.output.lower()

    def test_embed_still_accepted_on_duckdb(self, tmp_path):
        """No over-correction: the flag works where it means something."""
        f = _session(tmp_path / "proj")
        result = CliRunner().invoke(
            cli, [str(f), "--format", "duckdb", "-o", str(tmp_path / "o"),
                  "--embed", "--dry-run"]
        )

        assert "--embed" not in result.output


class TestOutputIsADirectory:
    """`-o` names a directory everywhere.

    It used to be a path stem for duckdb: `-o DIR/e` wrote `DIR/e.duckdb`
    and dropped `parquet_lake/` in DIR -- two things beside the path the
    user named rather than inside it.
    """

    def test_duckdb_lands_inside_the_named_directory(self, tmp_path):
        f = _session(tmp_path / "proj")
        out = tmp_path / "warehouse"
        result = CliRunner().invoke(
            cli, [str(f), "--format", "duckdb", "-o", str(out)]
        )

        assert result.exit_code == 0, result.output
        assert (out / "archive.duckdb").exists()
        assert not (tmp_path / "warehouse.duckdb").exists()

    def test_parquet_lake_lands_inside_too(self, tmp_path):
        f = _session(tmp_path / "proj")
        out = tmp_path / "warehouse"
        CliRunner().invoke(cli, [str(f), "--format", "duckdb", "-o", str(out)])

        assert (out / "parquet_lake").is_dir()
        assert not (tmp_path / "parquet_lake").exists()


class TestDuckdbPathEscapeHatch:
    """`-o foo.duckdb` is honoured, but must not put the lake in the cwd."""

    def test_a_bare_duckdb_name_does_not_drop_the_lake_in_cwd(
        self, tmp_path, monkeypatch
    ):
        f = _session(tmp_path / "proj")
        workdir = tmp_path / "cwd"
        workdir.mkdir()
        monkeypatch.chdir(workdir)

        result = CliRunner().invoke(
            cli, [str(f), "--format", "duckdb", "-o", "archive.duckdb"]
        )

        assert result.exit_code == 0, result.output
        assert (workdir / "archive.duckdb").exists()
        # The lake belongs beside the database, which here IS the cwd -- the
        # point is that it is resolved deliberately rather than landing there
        # because Path(".") happened to be the parent.
        assert (workdir / "parquet_lake").is_dir()


class TestOneWarehouseShape:
    """Every path that builds a warehouse builds the same one.

    Global post-loop sources ran only on the batch path, so a single-session
    build had `fact_messages.prompt_id` populated pointing at an empty
    `dim_prompt`. An external audit read that as an ETL defect; it was an
    entry-point defect.
    """

    def test_single_session_runs_the_global_sources(self, tmp_path):
        import duckdb

        f = _session(tmp_path / "proj")
        out = tmp_path / "warehouse"
        result = CliRunner().invoke(
            cli, [str(f), "--format", "duckdb", "-o", str(out)]
        )
        assert result.exit_code == 0, result.output

        conn = duckdb.connect(str(out / "archive.duckdb"), read_only=True)
        try:
            runs = conn.execute(
                "SELECT DISTINCT run_kind FROM fact_etl_runs "
                "WHERE run_kind IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()
        kinds = {r[0] for r in runs}
        assert "reconciliation" in kinds, (
            f"post-session reconciliation did not run; run_kinds={kinds}"
        )


class TestModeOnlyFlagsFailLoudly:
    """A flag that does nothing in the chosen mode must say so.

    Same rule the `--embed` / `--llm-facets` guards follow. These were
    missed: `ccutils session.jsonl --dry-run -q -j 4` wrote files and
    printed output -- a dry run that creates things and a quiet run that
    talks. `--dry-run` is the worst of them, because being ignored produces
    writes rather than a no-op.
    """

    def test_dry_run_rejected_without_source(self, tmp_path):
        f = _session(tmp_path / "proj")
        result = CliRunner().invoke(
            cli, [str(f), "--dry-run", "-o", str(tmp_path / "o")]
        )

        assert result.exit_code != 0
        assert "--dry-run" in result.output
        assert not (tmp_path / "o").exists(), "a rejected dry run wrote files"

    def test_batch_flags_rejected_without_source(self, tmp_path):
        f = _session(tmp_path / "proj")
        for flag in (["-j", "4"], ["--batch-size", "5"], ["--no-search-index"]):
            result = CliRunner().invoke(
                cli, [str(f), *flag, "-o", str(tmp_path / "o")]
            )
            assert result.exit_code != 0, f"{flag} accepted outside --source"

    def test_picker_flags_rejected_with_paths(self, tmp_path):
        f = _session(tmp_path / "proj")
        result = CliRunner().invoke(
            cli, [str(f), "--flat", "-o", str(tmp_path / "o")]
        )

        assert result.exit_code != 0
        assert "--flat" in result.output

    def test_the_same_flags_are_fine_with_source(self, tmp_path):
        """No over-correction: they work where they mean something."""
        _session(tmp_path / "projects" / "-home-user-projects-demo", "s1")
        result = CliRunner().invoke(
            cli, ["--source", str(tmp_path / "projects"), "--dry-run", "-q"]
        )

        assert result.exit_code == 0, result.output
