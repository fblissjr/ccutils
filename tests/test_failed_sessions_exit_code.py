"""A build in which a session failed exits non-zero, after writing what it could.

A DuckDB 2.0 install made every session run fail at its first fact
populator. The convert reported "65 of 65 session(s) failed and were
skipped", then printed "Exported to ..." and exited 0, leaving a warehouse
whose dimensions were full and whose facts were all empty. Nothing that
runs ccutils from a script or an agent could tell that build from a good
one without reading stderr.

Per-file isolation stays: one bad file still does not abort the others, and
the warehouse, its guide and the run metadata recording the failure are
still written. The exit status is what changes.

What breaks without it: a build that loaded nothing exits 0 and is
trusted.
"""

import duckdb
from click.testing import CliRunner
from helpers_ccutils import write_minimal_session

from ccutils.cli import cli


def _project(tmp_path):
    proj = tmp_path / "projects" / "-home-user-proj"
    proj.mkdir(parents=True)
    good = write_minimal_session(proj / "good-session.jsonl", "good-session")
    bad = proj / "bad-session.jsonl"
    bad.write_text("this is not a transcript line\n")
    return proj, good, bad


def _loaded_sessions(db_path):
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT session_id FROM fact_messages"
            ).fetchall()
        }
    finally:
        conn.close()


class TestNamedFiles:
    def test_a_failed_session_exits_non_zero(self, tmp_path):
        _, good, bad = _project(tmp_path)
        out = tmp_path / "out"
        result = CliRunner().invoke(
            cli, ["convert", str(good), str(bad), "--format", "duckdb",
                  "-o", str(out)],
        )
        assert result.exit_code == 1, result.output
        assert "1 of 2 session(s) failed" in result.output

        # Isolation still holds: the good session is in the warehouse and
        # the guide was written beside it.
        assert _loaded_sessions(out / "archive.duckdb") == {"good-session"}
        assert (out / "README.md").exists()

    def test_every_session_succeeding_exits_zero(self, tmp_path):
        _, good, _ = _project(tmp_path)
        out = tmp_path / "out"
        result = CliRunner().invoke(
            cli, ["convert", str(good), "--format", "duckdb", "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert _loaded_sessions(out / "archive.duckdb") == {"good-session"}


class TestSourceWalk:
    def test_a_failed_session_exits_non_zero(self, tmp_path):
        _project(tmp_path)
        out = tmp_path / "out"
        result = CliRunner().invoke(
            cli, ["convert", "--source", str(tmp_path / "projects"),
                  "--format", "duckdb", "-o", str(out), "-j", "1"],
        )
        assert result.exit_code == 1, result.output
        assert "bad-session" in result.output
        assert _loaded_sessions(out / "archive.duckdb") == {"good-session"}
