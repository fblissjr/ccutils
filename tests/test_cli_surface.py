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
