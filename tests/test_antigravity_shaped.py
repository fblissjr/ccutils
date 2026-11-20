"""The shaped layer over the Antigravity lake: what it reads out of a step.

Fixtures are built from the measured field map (docs/ANTIGRAVITY_CONTRACT.md
claims 17 and 18); the corpus canaries in test_antigravity_contract.py hold
that map against the real store.
"""

from datetime import datetime, timezone

from ccutils.parsers.antigravity import shaped

from helpers_antigravity import (
    BASE_TS,
    CLEARED,
    CONV_A,
    CONV_B,
    GENERIC,
    PLANNER_RESPONSE,
    SYSTEM_MESSAGE,
    USER_INPUT,
    cli_body,
    cli_step_row,
    f_msg,
    f_str,
)


def read(row):
    return shaped.read_step(row[9])


class TestReadStep:
    def test_a_model_response(self):
        got = read(cli_step_row(
            4, PLANNER_RESPONSE, BASE_TS + 4, owner=CONV_A, text="I will look.", thinking="The file first.",
            calls=(("call_7", "read_file", '{"Path": "a.py"}'),),
            usage={2: 1200, 3: 90, 5: 40_000, 9: 30}, generation=2, model_enum=1417))
        assert got["step_type"] == 15 and got["status"] == 3
        assert got["created_at"] == datetime.fromtimestamp(BASE_TS + 4, tz=timezone.utc)
        assert got["source"] == 2 and got["execution_id"] == "exec-1"
        assert got["text"] == "I will look." and got["thinking"] == "The file first."
        assert (got["input_tokens"], got["output_tokens"], got["cache_read_tokens"], got["thinking_tokens"]) == (
            1200, 90, 40_000, 30)
        assert got["generation_idx"] == 2 and got["model_enum"] == 1417
        assert got["owner_conversation_id"] == CONV_A
        assert got["tool_call_id"] is None and got["tool_output"] is None

    def test_a_tool_step_carries_the_call_and_its_result(self):
        got = read(cli_step_row(
            5, GENERIC, BASE_TS + 5, owner=CONV_A, tool=("call_7", "read_file", '{"Path": "a.py"}'),
            output="print('hi')\n"))
        assert (got["tool_call_id"], got["tool_name"], got["tool_args"]) == ("call_7", "read_file", '{"Path": "a.py"}')
        assert got["tool_output"] == "print('hi')\n"
        assert got["text"] is None and got["thinking"] is None

    def test_user_input_and_system_message_text(self):
        assert read(cli_step_row(0, USER_INPUT, BASE_TS, owner=CONV_A, source=4, text="do the thing"))["text"] == "do the thing"
        assert read(cli_step_row(1, SYSTEM_MESSAGE, BASE_TS, owner=CONV_A, source=5, text="task done"))["text"] == "task done"

    def test_a_cleared_step_keeps_its_identity_and_usage_and_states_no_text(self):
        got = read(cli_step_row(4, PLANNER_RESPONSE, BASE_TS + 4, owner=CONV_A, status=CLEARED,
                                usage={2: 1200, 3: 90}, generation=2))
        assert got["status"] == 5 and got["created_at"] == datetime.fromtimestamp(BASE_TS + 4, tz=timezone.utc)
        assert got["input_tokens"] == 1200 and got["generation_idx"] == 2
        assert got["text"] is None and got["thinking"] is None

    def test_a_cleared_user_input_is_read_from_the_parts_it_keeps(self):
        keeps = f_msg(19, f_msg(3, f_str(1, "do the ")), f_msg(3, f_str(1, "thing")))
        got = read(cli_step_row(0, USER_INPUT, BASE_TS, owner=CONV_A, status=CLEARED, cleared_keeps=keeps))
        assert got["text"] == "do the thing"

    def test_a_replayed_step_names_the_conversation_it_came_from(self):
        got = read(cli_step_row(2, USER_INPUT, BASE_TS, owner=CONV_B, text="from the fork's source"))
        assert got["owner_conversation_id"] == CONV_B

    def test_an_absent_field_is_none_never_zero_or_empty(self):
        got = read(cli_step_row(3, PLANNER_RESPONSE, BASE_TS, owner=CONV_A))
        for name in ("text", "thinking", "input_tokens", "output_tokens", "cache_read_tokens",
                     "thinking_tokens", "generation_idx", "model_enum", "tool_call_id", "tool_output"):
            assert got[name] is None, name

    def test_a_payload_that_is_not_protobuf_raises(self):
        import pytest
        with pytest.raises(ValueError):
            shaped.read_step(b"\x0a\xff\x01")

    def test_the_body_shapes_do_not_cross(self):
        # A tool step's output is not a message's text, whatever else is in the payload.
        row = cli_step_row(5, GENERIC, BASE_TS, owner=CONV_A, tool=("c", "run_command", "{}"), output="ok")
        payload = row[9] + cli_body(PLANNER_RESPONSE, text="not this step's")
        assert shaped.read_step(payload)["text"] is None


# ---------------------------------------------------------------------------
# Shaping a lake
# ---------------------------------------------------------------------------

import json
import shutil
import sqlite3

import duckdb
import pyarrow.parquet as pq
import pytest

from ccutils.parsers.antigravity import AntigravitySource
from ccutils.parsers.lake import write_lake

from helpers_antigravity import (
    _STEP_COLUMNS,
    cli_transcript_line,
    step_payload,
    write_conversation_db,
    write_summaries_db,
)

T = BASE_TS
CALL = ("call_1", "run_command", '{"CommandLine": "ls"}')


def _rows_a():
    return [
        cli_step_row(0, USER_INPUT, T, owner=CONV_A, source=4, text="do the thing"),
        cli_step_row(1, PLANNER_RESPONSE, T + 1, owner=CONV_A, thinking="list first", calls=(CALL,),
                     usage={2: 100, 3: 20, 5: 5000, 9: 7}, generation=0, model_enum=1417),
        cli_step_row(2, GENERIC, T + 2, owner=CONV_A, tool=CALL, output="a.py\n"),
        cli_step_row(3, PLANNER_RESPONSE, T + 3, owner=CONV_A, text="One file.", usage={2: 130, 3: 4}, generation=1),
    ]


def _lines_a():
    return [
        cli_transcript_line(0, "USER_INPUT", T, content="<USER_REQUEST>\ndo the thing\n</USER_REQUEST>", source="USER_EXPLICIT"),
        cli_transcript_line(1, "PLANNER_RESPONSE", T + 1, thinking="list first",
                            tool_calls=[{"name": "run_command", "args": {"CommandLine": "ls"}}],
                            input_tokens=100, output_tokens=20, cache_read_tokens=5000),
        cli_transcript_line(2, "GENERIC", T + 2, content="Output:\na.py\n"),
        # A branch the user reverted: same index, an earlier second, gone from the database.
        cli_transcript_line(3, "GENERIC", T - 500, content="abandoned branch output"),
        cli_transcript_line(3, "PLANNER_RESPONSE", T + 3, content="One file.", input_tokens=130, output_tokens=4),
    ]


def _insert(db, rows):
    conn = sqlite3.connect(db)
    conn.executemany(f"INSERT INTO steps ({', '.join(_STEP_COLUMNS)}) VALUES ({', '.join('?' for _ in _STEP_COLUMNS)})", rows)
    conn.commit()
    conn.close()


@pytest.fixture
def cli_home(tmp_path):
    """A CLI-only data root: a root conversation with a tool call and a
    reverted branch in its transcript, and a child with no transcript that
    opens with two steps replayed from its parent."""
    root = tmp_path / "gemini"
    store = root / "antigravity-cli"
    for conv, rows in ((CONV_A, _rows_a()), (CONV_B, [
        # The first two are replayed from A: same creation time, A named as their owner.
        cli_step_row(0, USER_INPUT, T, owner=CONV_A, source=4, text="do the thing"),
        cli_step_row(1, PLANNER_RESPONSE, T + 1, owner=CONV_A, thinking="list first", calls=(CALL,),
                     usage={2: 100, 3: 20, 5: 5000, 9: 7}, generation=0),
        cli_step_row(2, SYSTEM_MESSAGE, T + 10, owner=CONV_B, source=5, text="You are a worker."),
        cli_step_row(3, PLANNER_RESPONSE, T + 11, owner=CONV_B, thinking="worker plan",
                     usage={2: 50, 3: 5}, generation=0),
    ])):
        write_conversation_db(store / "conversations" / f"{conv}.db", conv, [])
        _insert(store / "conversations" / f"{conv}.db", rows)
    logs = store / "brain" / CONV_A / ".system_generated" / "logs"
    logs.mkdir(parents=True)
    (logs / "transcript_full.jsonl").write_text("\n".join(_lines_a()) + "\nnot json\n")
    write_summaries_db(store / "conversation_summaries.db", [
        {"conversation_id": CONV_A, "step_count": 4, "title": "List files", "app_data_dir": "antigravity-cli"},
        {"conversation_id": CONV_B, "step_count": 2, "parent_conversation_id": CONV_A, "nesting_depth": 1,
         "agent_name": "worker", "app_data_dir": "antigravity-cli"},
    ])
    return root


@pytest.fixture
def lake_root(tmp_path):
    return tmp_path / "lake"


def lake(cli_home, lake_root):
    return write_lake(AntigravitySource(app_bundle=None), lake_root=lake_root, root=cli_home)


def unit_table(lake_root, conv, name):
    path = lake_root / shaped.SHAPED_DIR / "antigravity-cli" / "conversations" / conv / f"{name}.parquet"
    return pq.read_table(path).to_pylist()


def views(lake_root):
    conn = duckdb.connect()
    shaped.create_views(conn, lake_root)
    return conn


def _clear(cli_home, idx, conv=CONV_A):
    db = cli_home / "antigravity-cli" / "conversations" / f"{conv}.db"
    conn = sqlite3.connect(db)
    step_type, meta = conn.execute("SELECT step_type, metadata FROM steps WHERE idx = ?", (idx,)).fetchone()
    conn.execute("UPDATE steps SET status = 5, step_payload = ? WHERE idx = ?",
                 (step_payload(step_type, CLEARED, meta), idx))
    conn.commit()
    conn.close()


class TestShapeLake:
    def test_every_step_becomes_a_row_with_its_fields(self, cli_home, lake_root):
        lake(cli_home, lake_root)
        result = shaped.shape_lake(lake_root)
        assert result.errors == []
        steps = {r["idx"]: r for r in unit_table(lake_root, CONV_A, "step")}
        assert set(steps) == {0, 1, 2, 3}
        assert steps[0]["step_type_name"] == "USER_INPUT" and steps[0]["source_name"] == "USER_EXPLICIT"
        assert steps[0]["text"] == "do the thing"
        assert steps[1]["thinking"] == "list first" and steps[1]["input_tokens"] == 100
        assert steps[1]["cache_read_tokens"] == 5000 and steps[1]["thinking_tokens"] == 7
        assert steps[2]["tool_name"] == "run_command" and steps[2]["tool_output"] == "a.py\n"
        assert steps[3]["text"] == "One file." and steps[3]["generation_idx"] == 1
        assert all(r["source_present"] and not r["is_cleared"] and r["decode_error"] is None for r in steps.values())
        assert steps[3]["created_second"] == T + 3 and steps[3]["payload_bytes"] > 0

    def test_transcript_lines_keep_file_order_and_say_which_are_current(self, cli_home, lake_root):
        lake(cli_home, lake_root)
        result = shaped.shape_lake(lake_root)
        lines = unit_table(lake_root, CONV_A, "transcript_line")
        assert [r["line_no"] for r in lines] == [0, 1, 2, 3, 4]
        assert [(r["step_index"], r["is_current"]) for r in lines] == [
            (0, True), (1, True), (2, True), (3, False), (3, True)]
        assert json.loads(lines[1]["tool_calls"]) == [{"name": "run_command", "args": {"CommandLine": "ls"}}]
        assert lines[1]["input_tokens"] == 100 and lines[4]["content"] == "One file."
        assert result.unparsed_lines == 1          # counted, never silently dropped

    def test_a_replayed_step_is_marked_inherited(self, cli_home, lake_root):
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        steps = {r["idx"]: r for r in unit_table(lake_root, CONV_B, "step")}
        assert {i: r["is_inherited"] for i, r in steps.items()} == {0: True, 1: True, 2: False, 3: False}
        assert {r["idx"]: r["is_inherited"] for r in unit_table(lake_root, CONV_A, "step")}[0] is False

    def test_a_line_from_another_branch_is_never_a_steps_own_line(self, cli_home, lake_root):
        # Measured: a few steps have lines at their index and none at their
        # creation second. The reverted branch's line is all the index holds.
        log = cli_home / "antigravity-cli" / "brain" / CONV_A / ".system_generated" / "logs" / "transcript_full.jsonl"
        log.write_text("\n".join(_lines_a()[:4]) + "\n")           # drop step 3's own line
        _clear(cli_home, 3)
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        lines = unit_table(lake_root, CONV_A, "transcript_line")
        assert [(r["step_index"], r["is_current"]) for r in lines][-1] == (3, False)
        assert views(lake_root).execute(
            "SELECT text, text_source, tool_output FROM antigravity_steps WHERE conversation_id = ? AND idx = 3",
            [CONV_A]).fetchone() == (None, None, None)

    def test_a_second_run_rewrites_nothing_and_a_changed_unit_only_itself(self, cli_home, lake_root):
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        base = lake_root / shaped.SHAPED_DIR / "antigravity-cli" / "conversations"
        before = {c: (base / c / "step.parquet").stat().st_mtime_ns for c in (CONV_A, CONV_B)}
        again = shaped.shape_lake(lake_root)
        assert again.written == 0 and again.unchanged == 2
        _clear(cli_home, 3)
        lake(cli_home, lake_root)
        third = shaped.shape_lake(lake_root)
        assert third.written == 1
        after = {c: (base / c / "step.parquet").stat().st_mtime_ns for c in (CONV_A, CONV_B)}
        assert after[CONV_A] != before[CONV_A] and after[CONV_B] == before[CONV_B]

    def test_without_thinking_no_thinking_text_is_written_anywhere(self, cli_home, lake_root):
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root, include_thinking=False)
        steps = unit_table(lake_root, CONV_A, "step")
        lines = unit_table(lake_root, CONV_A, "transcript_line")
        assert all(r["thinking"] is None for r in steps + lines)
        assert steps[1]["thinking_tokens"] == 7                   # a count is not the text
        tree = lake_root / shaped.SHAPED_DIR
        assert not any(b"list first" in p.read_bytes() for p in tree.rglob("*.parquet"))
        # Asking for it afterwards rebuilds the units: the flag is part of what was derived.
        assert shaped.shape_lake(lake_root).written == 2
        assert unit_table(lake_root, CONV_A, "step")[1]["thinking"] == "list first"

    def test_a_payload_that_does_not_decode_is_a_row_with_its_error(self, cli_home, lake_root):
        db = cli_home / "antigravity-cli" / "conversations" / f"{CONV_A}.db"
        conn = sqlite3.connect(db)
        conn.execute("UPDATE steps SET step_payload = x'0aff01' WHERE idx = 2")
        conn.commit()
        conn.close()
        lake(cli_home, lake_root)
        result = shaped.shape_lake(lake_root)
        row = {r["idx"]: r for r in unit_table(lake_root, CONV_A, "step")}[2]
        assert row["decode_error"] and row["tool_output"] is None and row["step_type"] == 132
        assert result.undecoded_steps == 1


class TestViews:
    def test_a_cleared_step_reads_from_the_fuller_row_the_lake_kept(self, cli_home, lake_root):
        lake(cli_home, lake_root)
        _clear(cli_home, 1)
        _clear(cli_home, 2)
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        rows = {r[0]: r for r in views(lake_root).execute(
            "SELECT idx, is_cleared, thinking, thinking_source, tool_output, tool_output_source "
            "FROM antigravity_steps WHERE conversation_id = ? ORDER BY idx", [CONV_A]).fetchall()}
        assert len(rows) == 4                                      # one row per step, not per archived version
        assert rows[1] == (1, True, "list first", "archive", None, None)
        assert rows[2][4:] == ("a.py\n", "archive")
        assert rows[3][2:4] == (None, None)                         # a response that never had thinking

    def test_the_fuller_row_is_read_where_there_is_no_transcript(self, cli_home, lake_root):
        # The child has no transcript: the lake's earlier row is the only copy.
        lake(cli_home, lake_root)
        _clear(cli_home, 2, CONV_B)
        _clear(cli_home, 3, CONV_B)
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        rows = {r[0]: r[1:] for r in views(lake_root).execute(
            "SELECT idx, text, text_source, thinking, thinking_source FROM antigravity_steps "
            "WHERE conversation_id = ?", [CONV_B]).fetchall()}
        assert rows[2] == ("You are a worker.", "archive", None, None)
        assert rows[3] == (None, None, "worker plan", "archive")

    def test_a_step_cleared_before_it_was_archived_reads_from_the_transcript(self, cli_home, lake_root):
        _clear(cli_home, 1)
        _clear(cli_home, 2)
        _clear(cli_home, 3)
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        rows = {r[0]: r[1:] for r in views(lake_root).execute(
            "SELECT idx, text, text_source, thinking, thinking_source, tool_output, tool_output_source "
            "FROM antigravity_steps WHERE conversation_id = ?", [CONV_A]).fetchall()}
        assert rows[1][2:4] == ("list first", "transcript")
        assert rows[2][4:] == ("Output:\na.py\n", "transcript")    # the transcript's wrapped form, and says so
        # The reverted branch's line shares the index and is never the source.
        assert rows[3][:2] == ("One file.", "transcript")
        assert rows[0][:2] == ("do the thing", "database")

    def test_a_pruned_conversation_keeps_its_row_and_its_steps(self, cli_home, lake_root):
        lake(cli_home, lake_root)
        store = cli_home / "antigravity-cli"
        (store / "conversations" / f"{CONV_B}.db").unlink()
        conn = sqlite3.connect(store / "conversation_summaries.db")
        conn.execute("DELETE FROM conversation_summaries WHERE conversation_id = ?", (CONV_B,))
        conn.commit()
        conn.close()
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        conn = views(lake_root)
        conv = dict(conn.execute("SELECT conversation_id, is_pruned FROM antigravity_conversations").fetchall())
        assert conv == {CONV_A: False, CONV_B: True}
        parent, depth, agent = conn.execute(
            "SELECT parent_conversation_id, nesting_depth, agent_name FROM antigravity_conversations "
            "WHERE conversation_id = ?", [CONV_B]).fetchone()
        assert (parent, depth, agent) == (CONV_A, 1, "worker")
        assert conn.execute("SELECT count(*) FROM antigravity_steps WHERE conversation_id = ?", [CONV_B]).fetchone()[0] == 4

    def test_tool_calls_and_usage(self, cli_home, lake_root):
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        conn = views(lake_root)
        assert conn.execute(
            "SELECT idx, tool_call_id, tool_name, tool_args, tool_output, is_error FROM antigravity_tool_calls "
            "WHERE conversation_id = ?", [CONV_A]).fetchall() == [
                (2, "call_1", "run_command", '{"CommandLine": "ls"}', "a.py\n", False)]
        assert conn.execute(
            "SELECT idx, input_tokens, output_tokens, cache_read_tokens, thinking_tokens FROM antigravity_usage "
            "WHERE conversation_id = ? ORDER BY idx", [CONV_A]).fetchall() == [
                (1, 100, 20, 5000, 7), (3, 130, 4, None, None)]
        # The child's replayed response is its parent's usage, counted there and not twice.
        assert conn.execute("SELECT idx, input_tokens FROM antigravity_usage WHERE conversation_id = ?",
                            [CONV_B]).fetchall() == [(3, 50)]

    def test_inherited_steps_are_in_the_view_and_marked(self, cli_home, lake_root):
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        assert views(lake_root).execute(
            "SELECT idx, is_inherited FROM antigravity_steps WHERE conversation_id = ? ORDER BY idx",
            [CONV_B]).fetchall() == [(0, True), (1, True), (2, False), (3, False)]

    def test_views_over_an_empty_tree_fail_loudly(self, lake_root):
        with pytest.raises(FileNotFoundError):
            shaped.create_views(duckdb.connect(), lake_root)


# ---------------------------------------------------------------------------
# `ccutils lake antigravity --decode`: the effect, not the exit code
# ---------------------------------------------------------------------------

from click.testing import CliRunner

from ccutils.cli import cli


def decode_cmd(cli_home, out, *flags):
    return CliRunner().invoke(cli, ["lake", "antigravity", "-o", str(out), "--source", str(cli_home), *flags])


class TestDecodeCommand:
    def test_decode_writes_the_derived_tree_and_a_database_of_views(self, cli_home, tmp_path):
        out = tmp_path / "out"
        result = decode_cmd(cli_home, out, "--decode")
        assert result.exit_code == 0, result.output
        assert len(unit_table(out, CONV_A, "step")) == 4
        db = out / shaped.SHAPED_DIR / "antigravity.duckdb"
        conn = duckdb.connect(str(db), read_only=True)
        assert conn.execute("SELECT count(*) FROM antigravity_steps").fetchone()[0] == 8
        assert conn.execute("SELECT count(*) FROM antigravity_conversations").fetchone()[0] == 2
        conn.close()
        assert str(db) in result.output and "steps 8" in result.output

    def test_without_the_flag_nothing_is_derived(self, cli_home, tmp_path):
        out = tmp_path / "out"
        assert decode_cmd(cli_home, out).exit_code == 0
        assert not (out / shaped.SHAPED_DIR).exists()

    def test_a_rerun_refreshes_the_views_database(self, cli_home, tmp_path):
        out = tmp_path / "out"
        decode_cmd(cli_home, out, "--decode")
        _clear(cli_home, 3)
        result = decode_cmd(cli_home, out, "--decode")
        assert result.exit_code == 0, result.output
        conn = duckdb.connect(str(out / shaped.SHAPED_DIR / "antigravity.duckdb"), read_only=True)
        assert conn.execute("SELECT is_cleared, text, text_source FROM antigravity_steps "
                            "WHERE conversation_id = ? AND idx = 3", [CONV_A]).fetchone() == (True, "One file.", "archive")

    def test_no_thinking_keeps_thinking_text_out_of_everything_derived(self, cli_home, tmp_path):
        out = tmp_path / "out"
        result = decode_cmd(cli_home, out, "--decode", "--no-thinking")
        assert result.exit_code == 0, result.output
        conn = duckdb.connect(str(out / shaped.SHAPED_DIR / "antigravity.duckdb"), read_only=True)
        assert conn.execute("SELECT count(*) FROM antigravity_steps WHERE thinking IS NOT NULL").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM antigravity_steps WHERE thinking_tokens IS NOT NULL").fetchone()[0] > 0
        conn.close()
        derived = out / shaped.SHAPED_DIR
        assert not any(b"list first" in p.read_bytes() or b"worker plan" in p.read_bytes()
                       for p in derived.rglob("*") if p.is_file())
        # The lake is the archive and keeps everything; the output says so.
        assert "the lake itself keeps thinking text" in result.output

    def test_no_thinking_without_decode_is_refused(self, cli_home, tmp_path):
        result = decode_cmd(cli_home, tmp_path / "out", "--no-thinking")
        assert result.exit_code == 2
        assert not (tmp_path / "out").exists()


def test_the_readmes_antigravity_recipes_bind_against_the_views(cli_home, lake_root):
    """The README shows SQL over these views. tests/test_query_recipes.py binds
    every ```sql block against the warehouse, which does not have them, so
    the Antigravity blocks are fenced ```sql antigravity and held here."""
    import re
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    blocks = re.findall(r"^```sql antigravity\n(.*?)^```", readme, re.M | re.S)
    assert blocks, "no ```sql antigravity block left in README.md"
    lake(cli_home, lake_root)
    shaped.shape_lake(lake_root)
    conn = views(lake_root)
    for block in blocks:
        code = "\n".join(line.split("--", 1)[0] for line in block.splitlines())
        for statement in (s.strip() for s in code.split(";")):
            if statement:
                conn.execute(statement).fetchall()
