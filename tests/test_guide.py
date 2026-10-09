"""The reader's guide: a warehouse explains itself to whoever opens it cold.

Claim: the primary consumer of a ccutils warehouse is an agent, not a
person at a SQL prompt. Two external auditors reported empty tables as ETL
defects because the only explanation lived in a skill file they never saw.
The guide is generated FROM the warehouse (`etl.table_coverage`, `etl.steps`,
row counts, `etl.audit_exceptions`, the FK convention) so it describes what
this file holds, not what the code could hold, and it is written beside the
archive every time a warehouse is built.

Delete these and the guide can drift from the warehouse it sits next to,
which is the failure the stub list already had once.
"""

import duckdb
import pytest
from click.testing import CliRunner

from ccutils import cli, create_star_schema
from ccutils.etl.orchestrator import run_v15_etl
from ccutils.guide import render_guide
from ccutils.schemas.star.schema import TABLE_COVERAGE
from helpers_ccutils import write_minimal_session

FIXTURE_FILE = "sample_session.jsonl"


@pytest.fixture
def warehouse(tmp_path):
    db = tmp_path / "wh" / "archive.duckdb"
    db.parent.mkdir()
    conn = create_star_schema(db)
    run_v15_etl(
        conn, write_minimal_session(tmp_path / "s.jsonl", "guide-s"),
        project_name="guide-project", parquet_lake_root=tmp_path / "lake",
    )
    conn.execute(
        "INSERT INTO etl.audit_exceptions VALUES "
        "('null_column', 'fact_meta_events', 'timestamp', 'no source timestamp')"
    )
    conn.close()
    return db


class TestGuideDescribesThisWarehouse:
    def test_every_object_appears_with_its_reason(self, warehouse):
        conn = duckdb.connect(str(warehouse), read_only=True)
        text = render_guide(conn)
        conn.close()
        for name, (kind, status, by, reason) in TABLE_COVERAGE.items():
            assert name in text, name
            assert reason.split(" (")[0][:40] in text, name

    def test_row_counts_and_scope_are_measured_not_declared(self, warehouse):
        conn = duckdb.connect(str(warehouse), read_only=True)
        text = render_guide(conn)
        project = conn.execute("SELECT project_name FROM dim_project").fetchone()[0]
        conn.close()
        assert project in text
        assert "| fact_messages |" in text
        # one session, two entries: the count line must carry a real number
        assert "| dim_session | populated | 1 |" in text

    def test_accepted_findings_are_listed(self, warehouse):
        conn = duckdb.connect(str(warehouse), read_only=True)
        text = render_guide(conn)
        conn.close()
        assert "fact_meta_events.timestamp" in text
        assert "no source timestamp" in text

    def test_join_paths_are_listed(self, warehouse):
        conn = duckdb.connect(str(warehouse), read_only=True)
        text = render_guide(conn)
        conn.close()
        assert "fact_messages.session_key" in text and "dim_session.session_key" in text


class TestGuideIsWrittenBesideTheArchive:
    def test_duckdb_build_writes_the_guide(self, tmp_path):
        from pathlib import Path

        fixture = Path(__file__).parent / FIXTURE_FILE
        out = tmp_path / "out"
        result = CliRunner().invoke(cli, [str(fixture), "--format", "duckdb", "-o", str(out)])
        assert result.exit_code == 0, result.output
        guide = out / "README.md"
        assert guide.exists()
        text = guide.read_text()
        assert "archive.duckdb" in text
        assert "etl.table_coverage" in text

    def test_guide_command_regenerates_it(self, warehouse):
        result = CliRunner().invoke(cli, ["guide", "-o", str(warehouse.parent)])
        assert result.exit_code == 0, result.output
        assert (warehouse.parent / "README.md").exists()


THINKING_SESSION = [
    {"type": "user", "uuid": "u1", "sessionId": "guide-think",
     "timestamp": "2026-09-20T10:00:00.000Z", "cwd": "/w",
     "message": {"role": "user", "content": "go"}},
    {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
     "sessionId": "guide-think", "timestamp": "2026-09-20T10:00:01.000Z",
     "message": {"role": "assistant", "model": "claude-opus-5",
                 "content": [{"type": "thinking",
                              "thinking": "PERSISTED REASONING",
                              "signature": "sig"}]}},
    {"type": "assistant", "uuid": "a2", "parentUuid": "a1",
     "sessionId": "guide-think", "timestamp": "2026-09-20T10:00:02.000Z",
     "message": {"role": "assistant", "model": "claude-opus-5",
                 "content": [{"type": "text", "text": "the answer"}]}},
]


class TestGuideOnReasoningText:
    """The guide says what ccutils does with thinking, not what the source holds.

    The entry used to read "thinking blocks are persisted by Claude Code with
    an empty `thinking` field ... content the source never wrote". That was a
    statement about an unversioned upstream format, frozen into a file that
    ships beside every warehouse, and it went false when Claude Code started
    writing text into a share of those blocks. A reader was then told the
    data does not exist while it sat on disk.

    The pipeline's own behavior is what the guide can state and a test can
    hold: thinking text is not ingested, whatever the source wrote.
    """

    def _section(self, warehouse):
        conn = duckdb.connect(str(warehouse), read_only=True)
        text = render_guide(conn)
        conn.close()
        return text.split("## What is not in here")[1]

    def test_it_does_not_claim_the_source_never_wrote_reasoning(self, warehouse):
        section = self._section(warehouse)
        assert "Reasoning text" in section
        assert "never wrote" not in section
        assert "with an empty" not in section

    def test_it_says_reasoning_is_not_ingested_and_where_to_look(self, warehouse):
        section = self._section(warehouse)
        assert "not ingested" in section
        assert "content_text" in section and "has_thinking" in section
        assert "JSONL_CONTRACT.md" in section
        assert "last_assistant_message" in section and "--no-thinking" in section

    def test_what_it_says_is_true_of_the_pipeline(self, tmp_path):
        """A thinking block that carries text reaches no message's text."""
        import json

        src = tmp_path / "proj" / "think.jsonl"
        src.parent.mkdir(parents=True)
        src.write_text("\n".join(json.dumps(e) for e in THINKING_SESSION))
        conn = create_star_schema(tmp_path / "think.duckdb")
        run_v15_etl(conn, src, project_name="x",
                    parquet_lake_root=tmp_path / "lake")
        rows = conn.execute(
            "SELECT has_thinking, content_text FROM fact_messages "
            "WHERE message_type = 'assistant' ORDER BY timestamp"
        ).fetchall()
        conn.close()

        assert [r[0] for r in rows] == [True, False]
        assert not any("PERSISTED REASONING" in (r[1] or "") for r in rows)
        assert rows[1][1] == "the answer"

    def test_the_exception_it_names_is_real(self, tmp_path):
        """A session that ends on a thinking block keeps that text in dim_session.

        The guide names this as the one place reasoning text lands. If the
        default ever stops including it, the sentence goes with it.
        """
        import json

        def build(name, include_thinking):
            src = tmp_path / name / "ends-on-thinking.jsonl"
            src.parent.mkdir(parents=True)
            src.write_text("\n".join(json.dumps(e) for e in THINKING_SESSION[:2]))
            conn = create_star_schema(tmp_path / f"{name}.duckdb")
            run_v15_etl(conn, src, project_name="x",
                        parquet_lake_root=tmp_path / f"{name}-lake",
                        include_thinking=include_thinking)
            last = conn.execute(
                "SELECT last_assistant_message FROM dim_session"
            ).fetchone()[0]
            conn.close()
            return last or ""

        assert "PERSISTED REASONING" in build("default", True)
        assert "PERSISTED REASONING" not in build("no-thinking", False)
