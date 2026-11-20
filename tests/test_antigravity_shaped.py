"""The shaped layer over the Antigravity lake: what it reads out of a step.

Fixtures are built from the measured field map (docs/ANTIGRAVITY_CONTRACT.md
claims 17 and 18); the corpus canaries in test_antigravity_contract.py hold
that map against the real store.
"""

import io
import re
from datetime import datetime, timezone
from pathlib import Path

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
    cli_metadata,
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
from ccutils.parsers.lake import read_manifest, write_lake

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


MACHINE = "m1"


@pytest.fixture
def lake_root(tmp_path):
    """One machine's root inside a lake base: the base holds a directory per machine."""
    return tmp_path / "lake" / f"machine={MACHINE}"


def lake(cli_home, lake_root):
    return write_lake(AntigravitySource(app_bundle=None), lake_root=lake_root, root=cli_home)


def unit_table(lake_root, conv, name):
    path = lake_root / shaped.SHAPED_DIR / "antigravity-cli" / "conversations" / conv / f"{name}.parquet"
    return pq.read_table(path).to_pylist()


def views(lake_root):
    conn = duckdb.connect()
    shaped.create_views(conn, lake_root.parent)
    return conn


def _clear(cli_home, idx, conv=CONV_A):
    db = cli_home / "antigravity-cli" / "conversations" / f"{conv}.db"
    conn = sqlite3.connect(db)
    step_type, meta = conn.execute("SELECT step_type, metadata FROM steps WHERE idx = ?", (idx,)).fetchone()
    conn.execute("UPDATE steps SET status = 5, step_payload = ? WHERE idx = ?",
                 (step_payload(step_type, CLEARED, meta), idx))
    conn.commit()
    conn.close()


class TestStepTypeNames:
    """The name map is the contract's measured table, both ways. The first cut
    named only the six types the CLI store still writes, so on a store with
    older conversations every per-tool step decoded with no name."""

    def _observed(self):
        doc = (Path(__file__).parent.parent / "docs" / "ANTIGRAVITY_CONTRACT.md").read_text()
        start = doc.index("Observed step types, both stores:")
        block = doc[start:doc.index("\n\n", start)]
        return {int(n): name for n, name in re.findall(r"(\d+)\s+`([A-Z_]+)`", block)}

    def test_every_type_the_contract_observed_is_named_and_nothing_else(self):
        observed = self._observed()
        assert len(observed) > 20 and observed[8] == "VIEW_FILE" and observed[132] == "GENERIC"
        assert shaped.STEP_TYPE_NAMES == observed

    def test_a_per_tool_step_is_named_in_the_derived_table(self, cli_home, lake_root):
        db = cli_home / "antigravity-cli" / "conversations" / f"{CONV_A}.db"
        conn = sqlite3.connect(db)
        step_type, meta = conn.execute("SELECT step_type, metadata FROM steps WHERE idx = 2").fetchone()
        conn.execute("UPDATE steps SET step_type = 21, step_payload = ? WHERE idx = 2",
                     (step_payload(21, 3, meta),))
        conn.commit()
        conn.close()
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        steps = {r["idx"]: r for r in unit_table(lake_root, CONV_A, "step")}
        assert (steps[2]["step_type"], steps[2]["step_type_name"]) == (21, "RUN_COMMAND")

    def test_a_type_nobody_has_named_keeps_its_number_and_a_null_name(self):
        assert 9999 not in shaped.STEP_TYPE_NAMES
        assert shaped.read_step(step_payload(9999, 3, b""))["step_type"] == 9999


class TestDerivedFormatVersion:
    """A unit is skipped when its lake unit is unchanged, so a change to what
    is READ out of a step reaches an existing tree only through the version."""

    def test_the_readings_are_pinned_to_the_version(self):
        assert (shaped.SHAPED_FORMAT_VERSION, shaped.format_fingerprint()) == (3, PINNED_SHAPED_FINGERPRINT), (
            "what the shaped layer reads or writes changed: bump SHAPED_FORMAT_VERSION, then update this pin")

    def test_a_new_version_rederives_a_tree_that_already_holds_the_rows(self, cli_home, lake_root, monkeypatch):
        lake(cli_home, lake_root)
        first = shaped.shape_lake(lake_root)
        assert first.written and shaped.shape_lake(lake_root).written == 0
        monkeypatch.setattr(shaped, "SHAPED_FORMAT_VERSION", shaped.SHAPED_FORMAT_VERSION + 1)
        monkeypatch.setitem(shaped.STEP_TYPE_NAMES, shaped.USER_INPUT, "RENAMED")
        again = shaped.shape_lake(lake_root)
        assert (again.written, again.unchanged) == (first.written, 0)
        assert unit_table(lake_root, CONV_A, "step")[0]["step_type_name"] == "RENAMED"


PINNED_SHAPED_FINGERPRINT = "666f86ff1750f691"


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


    def test_progress_reports_each_conversation_and_how_it_ended(self, cli_home, lake_root):
        lake(cli_home, lake_root)
        events = []
        result = shaped.shape_lake(lake_root, on_unit=events.append)
        assert {e.phase for e in events} == {"decode"}
        total = result.written
        assert total > 1 and {e.total for e in events} == {total}
        ends = [e for e in events if e.status is not None]
        assert [e.index for e in ends] == list(range(1, total + 1))
        assert {e.status for e in ends} == {"written"}
        assert sum(e.rows["steps"] for e in ends) == result.steps
        assert [e.index for e in events if e.status is None] == list(range(1, total + 1))
        events = []
        shaped.shape_lake(lake_root, on_unit=events.append)
        assert [e.status for e in events] == ["unchanged"] * total


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

    def _add_older_shapes(self, cli_home):
        """A per-tool step as the app wrote tool calls before GENERIC (claim
        11), a tool step in a status nobody has named, and a checkpoint that
        states the usage of its own model call."""
        old_call = ("call_2", "run_command", '{"CommandLine": "pwd"}')
        meta = cli_metadata(T + 4, idx=4, owner=CONV_A, tool=old_call)
        rows = [
            (4, 21, 3, 0, meta, None, None, None, None, step_payload(21, 3, meta), 0),
            cli_step_row(5, GENERIC, T + 5, owner=CONV_A, status=1, tool=("call_3", "view_file", "{}")),
            cli_step_row(6, shaped.CHECKPOINT, T + 6, owner=CONV_A, text="summary so far",
                         usage={2: 9000, 3: 300}, generation=1),
            # A call whose result is an error message step, in status DONE.
            cli_step_row(7, shaped.ERROR_MESSAGE, T + 7, owner=CONV_A, text="path is outside the workspace",
                         tool=("call_4", "view_file", '{"AbsolutePath": "/etc/x"}')),
        ]
        _insert(cli_home / "antigravity-cli" / "conversations" / f"{CONV_A}.db", rows)
        log = cli_home / "antigravity-cli" / "brain" / CONV_A / ".system_generated" / "logs" / "transcript_full.jsonl"
        with log.open("a") as f:
            f.write(cli_transcript_line(4, "RUN_COMMAND", T + 4, content="Output:\n/work\n") + "\n")

    def test_a_call_from_before_generic_steps_is_a_tool_call(self, cli_home, lake_root):
        self._add_older_shapes(cli_home)
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        conn = views(lake_root)
        assert conn.execute(
            "SELECT idx, step_type_name, tool_call_id, tool_name, tool_output, tool_output_source "
            "FROM antigravity_tool_calls WHERE conversation_id = ? ORDER BY idx", [CONV_A]).fetchall() == [
                (2, "GENERIC", "call_1", "run_command", "a.py\n", "database"),
                (4, "RUN_COMMAND", "call_2", "run_command", "Output:\n/work\n", "transcript"),
                (5, "GENERIC", "call_3", "view_file", None, None),
                (7, "ERROR_MESSAGE", "call_4", "view_file", None, None)]
        # What the transcript holds for a tool call is its output, never the step's text.
        assert conn.execute("SELECT text, text_source FROM antigravity_steps WHERE conversation_id = ? AND idx = 4",
                            [CONV_A]).fetchone() == (None, None)

    def test_is_error_is_stated_for_a_status_nobody_has_named(self, cli_home, lake_root):
        self._add_older_shapes(cli_home)
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        conn = views(lake_root)
        assert conn.execute("SELECT status, status_name, is_error FROM antigravity_tool_calls "
                            "WHERE conversation_id = ? AND idx = 5", [CONV_A]).fetchone() == (1, None, False)

    def test_a_call_answered_by_an_error_message_is_an_error_with_its_text(self, cli_home, lake_root):
        self._add_older_shapes(cli_home)
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        conn = views(lake_root)
        assert conn.execute(
            "SELECT idx, status_name, is_error, error_text FROM antigravity_tool_calls "
            "WHERE conversation_id = ? ORDER BY idx", [CONV_A]).fetchall() == [
                (2, "DONE", False, None), (4, "DONE", False, None), (5, None, False, None),
                (7, "DONE", True, "path is outside the workspace")]

    def test_usage_says_which_kind_of_step_stated_it(self, cli_home, lake_root):
        self._add_older_shapes(cli_home)
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        conn = views(lake_root)
        # A checkpoint states its own call's usage under the generation of the
        # response that follows it; the type tells the two apart.
        assert conn.execute("SELECT idx, step_type_name, generation_idx, input_tokens FROM antigravity_usage "
                            "WHERE conversation_id = ? ORDER BY idx", [CONV_A]).fetchall() == [
            (1, "PLANNER_RESPONSE", 0, 100), (3, "PLANNER_RESPONSE", 1, 130), (6, "CHECKPOINT", 1, 9000)]

    def test_inherited_steps_are_in_the_view_and_marked(self, cli_home, lake_root):
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        assert views(lake_root).execute(
            "SELECT idx, is_inherited FROM antigravity_steps WHERE conversation_id = ? ORDER BY idx",
            [CONV_B]).fetchall() == [(0, True), (1, True), (2, False), (3, False)]

    def test_views_over_an_empty_tree_fail_loudly(self, lake_root):
        with pytest.raises(FileNotFoundError):
            shaped.create_views(duckdb.connect(), lake_root)


class TestTasks:
    """A background task has no id of its own: the tool step that started it,
    its log file and the system message that reports it share a step index
    (contract claim 19)."""

    def _add_tasks(self, cli_home):
        def report(n, verb, body):
            return (f'[Message] timestamp=2026-09-21T14:13:29Z sender={CONV_A}/task-{n} priority=MESSAGE_PRIORITY_HIGH '
                    f'content=Task id "{CONV_A}/task-{n}" {verb} with result:\n{body}')
        finished = report(8, "finished", "Command exited with code 2.\n\nCommand output:\nmake: child exited with code 0\n")
        rows = [
            cli_step_row(8, GENERIC, T + 8, owner=CONV_A, tool=("call_8", "run_command", '{"CommandLine": "make"}')),
            cli_step_row(9, SYSTEM_MESSAGE, T + 20, owner=CONV_A, source=5, text=finished),
            cli_step_row(10, GENERIC, T + 21, owner=CONV_A, tool=("call_10", "schedule", '{"DurationSeconds": 60}')),
            cli_step_row(11, GENERIC, T + 22, owner=CONV_A, tool=("call_11", "run_command", '{"CommandLine": "sleep 9"}')),
            cli_step_row(12, SYSTEM_MESSAGE, T + 30, owner=CONV_A, source=5,
                         text=report(11, "was canceled", "The command was stopped.")),
        ]
        _insert(cli_home / "antigravity-cli" / "conversations" / f"{CONV_A}.db", rows)
        # The fork replays the report; the task is still its source's.
        _insert(cli_home / "antigravity-cli" / "conversations" / f"{CONV_B}.db",
                [cli_step_row(40, SYSTEM_MESSAGE, T + 20, owner=CONV_A, source=5, text=finished)])
        tasks = cli_home / "antigravity-cli" / "brain" / CONV_A / ".system_generated" / "tasks"
        tasks.mkdir(parents=True)
        (tasks / "task-8.log").write_text("make: *** [all] Error 2\n")
        (tasks / "task-10.log").write_text("tick\n")
        # A log left by a branch the user reverted: its index now holds a user input.
        (tasks / "task-0.log").write_text("from a reverted branch\n")
        # Not a task log, though its path says task-<n>: a task's message inbox.
        inbox = cli_home / "antigravity-cli" / "brain" / CONV_A / "task-8" / ".system_generated" / "messages"
        inbox.mkdir(parents=True)
        (inbox / f"{CONV_B}.json").write_text("{}")

    def _tasks(self, cli_home, lake_root, columns, where="TRUE"):
        self._add_tasks(cli_home)
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        return views(lake_root).execute(
            f"SELECT {columns} FROM antigravity_tasks WHERE {where} ORDER BY conversation_id, idx").fetchall()

    def test_one_row_per_task_from_its_log_or_its_report(self, cli_home, lake_root):
        assert self._tasks(cli_home, lake_root, "conversation_id, idx, has_log, outcome") == [
            (CONV_A, 0, True, None), (CONV_A, 8, True, "finished"),
            (CONV_A, 10, True, None), (CONV_A, 11, False, "canceled")]

    def test_a_finished_task_ties_its_command_its_exit_code_and_its_log(self, cli_home, lake_root):
        assert self._tasks(cli_home, lake_root,
                           "tool_name, tool_args, exit_code, reported_idx, log_relpath, log_size_bytes, log_text",
                           "idx = 8") == [
            ("run_command", '{"CommandLine": "make"}', 2, 9,
             f"brain/{CONV_A}/.system_generated/tasks/task-8.log", 24, "make: *** [all] Error 2\n")]

    def test_the_exit_code_is_the_commands_not_one_its_output_mentions(self, cli_home, lake_root):
        assert self._tasks(cli_home, lake_root, "exit_code", "idx = 8") == [(2,)]

    def test_times_are_the_launch_steps_and_the_reports(self, cli_home, lake_root):
        assert self._tasks(cli_home, lake_root,
                           "epoch(reported_at) - epoch(started_at)", "idx IN (8, 11)") == [(12.0,), (8.0,)]

    def test_a_log_whose_index_no_longer_holds_its_command_keeps_its_row_and_states_no_launch(self, cli_home, lake_root):
        assert self._tasks(cli_home, lake_root, "tool_name, tool_args, started_at, has_log", "idx = 0") == [
            (None, None, None, True)]

    def test_a_task_nothing_reported_states_no_outcome(self, cli_home, lake_root):
        assert self._tasks(cli_home, lake_root, "tool_name, outcome, exit_code, reported_at", "idx = 10") == [
            ("schedule", None, None, None)]

    def test_a_fork_replaying_the_report_has_no_task_and_is_not_the_report(self, cli_home, lake_root):
        rows = self._tasks(cli_home, lake_root, "conversation_id, reported_idx", "idx = 8")
        assert rows == [(CONV_A, 9)]


# ---------------------------------------------------------------------------
# `ccutils lake antigravity --decode`: the effect, not the exit code
# ---------------------------------------------------------------------------

from click.testing import CliRunner

from ccutils.cli import cli
from ccutils.cli.lake_cmd import Reporter
from ccutils.parsers.lake import UnitEvent


def decode_cmd(cli_home, out, *flags):
    flags = flags if "--machine" in flags else (*flags, "--machine", MACHINE)
    return CliRunner().invoke(cli, ["lake", "antigravity", "-o", str(out), "--source", str(cli_home), *flags])


class TestDecodeCommand:
    def test_decode_writes_the_derived_tree_and_a_database_of_views(self, cli_home, tmp_path):
        out = tmp_path / "out"
        result = decode_cmd(cli_home, out, "--decode")
        assert result.exit_code == 0, result.output
        assert len(unit_table(out / f"machine={MACHINE}", CONV_A, "step")) == 4
        db = out / "antigravity.duckdb"
        conn = duckdb.connect(str(db), read_only=True)
        assert conn.execute("SELECT count(*) FROM antigravity_steps").fetchone()[0] == 8
        assert conn.execute("SELECT count(*) FROM antigravity_conversations").fetchone()[0] == 2
        conn.close()
        assert str(db) in result.output and "steps 8" in result.output

    def test_without_the_flag_nothing_is_derived(self, cli_home, tmp_path):
        out = tmp_path / "out"
        assert decode_cmd(cli_home, out).exit_code == 0
        assert not (out / f"machine={MACHINE}" / shaped.SHAPED_DIR).exists()

    def test_a_rerun_refreshes_the_views_database(self, cli_home, tmp_path):
        out = tmp_path / "out"
        decode_cmd(cli_home, out, "--decode")
        _clear(cli_home, 3)
        result = decode_cmd(cli_home, out, "--decode")
        assert result.exit_code == 0, result.output
        conn = duckdb.connect(str(out / "antigravity.duckdb"), read_only=True)
        assert conn.execute("SELECT is_cleared, text, text_source FROM antigravity_steps "
                            "WHERE conversation_id = ? AND idx = 3", [CONV_A]).fetchone() == (True, "One file.", "archive")

    def test_no_thinking_keeps_thinking_text_out_of_everything_derived(self, cli_home, tmp_path):
        out = tmp_path / "out"
        result = decode_cmd(cli_home, out, "--decode", "--no-thinking")
        assert result.exit_code == 0, result.output
        conn = duckdb.connect(str(out / "antigravity.duckdb"), read_only=True)
        assert conn.execute("SELECT count(*) FROM antigravity_steps WHERE thinking IS NOT NULL").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM antigravity_steps WHERE thinking_tokens IS NOT NULL").fetchone()[0] > 0
        conn.close()
        derived = out / f"machine={MACHINE}" / shaped.SHAPED_DIR
        assert not any(b"list first" in p.read_bytes() or b"worker plan" in p.read_bytes()
                       for p in derived.rglob("*") if p.is_file())
        # The lake is the archive and keeps everything; the output says so.
        assert "the lake itself keeps thinking text" in result.output

    def test_no_thinking_without_decode_is_refused(self, cli_home, tmp_path):
        result = decode_cmd(cli_home, tmp_path / "out", "--no-thinking")
        assert result.exit_code == 2
        assert not (tmp_path / "out").exists()


class TestMachines:
    """One lake base holds a directory per machine, `machine=<name>`. A run
    writes only its own; another machine's directory copied in beside it is
    read by the same views, and every view says which machine a row is from."""

    def test_a_run_writes_under_its_machines_directory_and_remembers_the_name(self, cli_home, tmp_path):
        out = tmp_path / "out"
        assert decode_cmd(cli_home, out, "--machine", "desk").exit_code == 0
        assert (out / "machine=desk" / "antigravity" / "_manifest.parquet").exists()
        assert not (out / "antigravity").exists()
        assert (out / "_this_machine").read_text().strip() == "desk"
        # The name is remembered, so later runs need not say it.
        result = CliRunner().invoke(cli, ["lake", "antigravity", "-o", str(out), "--source", str(cli_home)])
        assert result.exit_code == 0, result.output
        assert sorted(p.name for p in out.iterdir() if p.name.startswith("machine=")) == ["machine=desk"]

    def test_a_different_name_for_a_lake_that_has_one_is_refused(self, cli_home, tmp_path):
        out = tmp_path / "out"
        decode_cmd(cli_home, out, "--machine", "desk")
        result = decode_cmd(cli_home, out, "--machine", "laptop")
        assert result.exit_code == 2 and "desk" in result.output
        assert not (out / "machine=laptop").exists()

    def test_without_a_name_the_hostname_is_used(self, cli_home, tmp_path, monkeypatch):
        import socket
        monkeypatch.setattr(socket, "gethostname", lambda: "Some Host.local")
        out = tmp_path / "out"
        result = CliRunner().invoke(cli, ["lake", "antigravity", "-o", str(out), "--source", str(cli_home)])
        assert result.exit_code == 0, result.output
        assert (out / "_this_machine").read_text().strip() == "some-host"
        assert (out / "machine=some-host" / "antigravity").is_dir() and "some-host" in result.output

    def test_a_name_that_is_not_a_plain_word_is_refused(self, cli_home, tmp_path):
        for bad in ("a=b", "../up", "Has Space", ""):
            result = decode_cmd(cli_home, tmp_path / "out", "--machine", bad)
            assert result.exit_code == 2, bad
        assert not (tmp_path / "out").exists()

    def _two_machines(self, cli_home, tmp_path, between=None):
        """`laptop` ran elsewhere and its directory was copied in; `desk` ran
        here, after `between` changed what the store holds."""
        out, elsewhere = tmp_path / "out", tmp_path / "elsewhere"
        assert decode_cmd(cli_home, elsewhere, "--decode", "--machine", "laptop").exit_code == 0
        if between:
            between()
        assert decode_cmd(cli_home, out, "--decode", "--machine", "desk").exit_code == 0
        shutil.copytree(elsewhere / "machine=laptop", out / "machine=laptop")
        assert decode_cmd(cli_home, out, "--decode", "--machine", "desk").exit_code == 0   # the views see both
        return out

    def test_the_views_read_every_machine_and_say_which(self, cli_home, tmp_path):
        out = self._two_machines(cli_home, tmp_path)
        conn = duckdb.connect()
        shaped.create_views(conn, out)
        for view in shaped.VIEW_NAMES:
            assert conn.execute(f"DESCRIBE {view}").fetchall()[0][0] == "machine", view
        assert conn.execute("SELECT machine, count(*) FROM antigravity_conversations GROUP BY 1 ORDER BY 1"
                            ).fetchall() == [("desk", 2), ("laptop", 2)]

    def test_the_same_conversation_on_two_machines_is_two_and_nothing_joins_across(self, cli_home, tmp_path):
        out = self._two_machines(cli_home, tmp_path)
        conn = duckdb.connect()
        shaped.create_views(conn, out)
        # Each machine reads as it would alone: one row per step, per call, per response.
        assert conn.execute("SELECT machine, count(*) FROM antigravity_steps GROUP BY 1 ORDER BY 1"
                            ).fetchall() == [("desk", 8), ("laptop", 8)]
        assert conn.execute("SELECT machine, count(*) FROM antigravity_tool_calls GROUP BY 1 ORDER BY 1"
                            ).fetchall() == [("desk", 1), ("laptop", 1)]
        assert conn.execute("SELECT machine, sum(input_tokens) FROM antigravity_usage GROUP BY 1 ORDER BY 1"
                            ).fetchall() == [("desk", 280), ("laptop", 280)]
        assert conn.execute("SELECT count(*), count(DISTINCT (machine, store, conversation_id, idx)) "
                            "FROM antigravity_steps").fetchone() == (16, 16)

    def test_a_step_emptied_on_one_machine_does_not_read_the_other_machines_copy(self, cli_home, tmp_path):
        # The laptop archived the reply whole; the app then emptied it, and the
        # desk only ever read it empty. The desk has the transcript, not the row.
        out = self._two_machines(cli_home, tmp_path, between=lambda: _clear(cli_home, 3))
        conn = duckdb.connect(str(out / "antigravity.duckdb"), read_only=True)
        assert conn.execute("SELECT machine, is_cleared, text, text_source FROM antigravity_steps "
                            "WHERE conversation_id = ? AND idx = 3 ORDER BY 1", [CONV_A]).fetchall() == [
            ("desk", True, "One file.", "transcript"), ("laptop", False, "One file.", "database")]

    def test_tasks_are_each_machines_own(self, cli_home, tmp_path):
        TestTasks()._add_tasks(cli_home)
        out = self._two_machines(cli_home, tmp_path)
        conn = duckdb.connect(str(out / "antigravity.duckdb"), read_only=True)
        assert conn.execute("SELECT machine, count(*), count(tool_name), count(exit_code), count(log_text) "
                            "FROM antigravity_tasks GROUP BY 1 ORDER BY 1").fetchall() == [
            ("desk", 4, 3, 1, 3), ("laptop", 4, 3, 1, 3)]

    def test_a_later_run_here_leaves_the_other_machines_directory_alone(self, cli_home, tmp_path):
        out = self._two_machines(cli_home, tmp_path)
        before = {p.relative_to(out): p.stat().st_mtime_ns for p in (out / "machine=laptop").rglob("*") if p.is_file()}
        (cli_home / "antigravity-cli" / "conversations" / f"{CONV_B}.db").unlink()
        assert decode_cmd(cli_home, out, "--decode", "--force", "--machine", "desk").exit_code == 0
        after = {p.relative_to(out): p.stat().st_mtime_ns for p in (out / "machine=laptop").rglob("*") if p.is_file()}
        assert after == before
        conn = duckdb.connect(str(out / "antigravity.duckdb"), read_only=True)
        assert conn.execute("SELECT machine, is_pruned FROM antigravity_conversations WHERE conversation_id = ? "
                            "ORDER BY 1", [CONV_B]).fetchall() == [("desk", True), ("laptop", False)]

    def test_a_lake_from_before_machines_is_moved_whole_and_nothing_is_rewritten(self, cli_home, tmp_path):
        out = tmp_path / "out"
        write_lake(AntigravitySource(app_bundle=None), lake_root=out, root=cli_home)   # the old layout
        shaped.shape_lake(out)
        steps = out / "antigravity" / "antigravity-cli" / "conversations" / CONV_B / "steps.parquet"
        held = pq.read_table(steps).to_pylist()
        # The app deletes B; the lake is now its only copy.
        (cli_home / "antigravity-cli" / "conversations" / f"{CONV_B}.db").unlink()
        result = decode_cmd(cli_home, out, "--decode")
        assert result.exit_code == 0, result.output
        assert not (out / "antigravity").exists() and not (out / "antigravity-decoded").exists()
        moved = out / f"machine={MACHINE}" / "antigravity" / "antigravity-cli" / "conversations"
        kept = pq.read_table(moved / CONV_B / "steps.parquet").to_pylist()
        assert [r["step_payload"] for r in kept] == [r["step_payload"] for r in held]
        assert [r["lake_run_id"] for r in kept] == [r["lake_run_id"] for r in held]
        assert not any(r["source_present"] for r in kept)
        statuses = {r["unit_id"]: r["status"] for r in read_manifest(out / f"machine={MACHINE}" / "antigravity")}
        assert statuses[CONV_A] == "unchanged" and statuses[CONV_B] == "source_missing"
        assert "moved" in result.output


def workspace_entry(uri, root=None, slug=None, remote=None, branch=None):
    """One workspace of a conversation's metadata blob (claim 20): 1 uri,
    2 git root, 3 {1 owner/repo, 2 remote}, 4 branch."""
    parts = [f_str(1, uri)]
    if root:
        parts.append(f_str(2, root))
    if slug or remote:
        parts.append(f_msg(3, *([f_str(1, slug)] if slug else []), *([f_str(2, remote)] if remote else [])))
    if branch:
        parts.append(f_str(4, branch))
    return f_msg(1, *parts)


def _set_blob(cli_home, conv, blob):
    conn = sqlite3.connect(cli_home / "antigravity-cli" / "conversations" / f"{conv}.db")
    conn.execute("UPDATE trajectory_metadata_blob SET data = ?", (blob,))
    conn.commit()
    conn.close()


class TestProjectIdentity:
    """Which project a conversation belongs to. The measured case it has to
    get right: one repository checked out at different paths on two machines,
    from two remotes under two owner names. Only the repository's name agrees."""

    A = workspace_entry("file:///home/someone/ComfyUI/custom_nodes/My-Repo", "file:///home/someone/ComfyUI/custom_nodes/My-Repo",
                        "handle/My-Repo", "git@github.com:handle/My-Repo.git", "main")
    B = workspace_entry("file:///Users/another/workspace/my-repo", "file:///Users/another/workspace/my-repo",
                        "mirror-user/my-repo", "ssh://git@10.0.0.5:2222/mirror-user/my-repo", "feature/x")

    def test_the_stated_workspaces_are_read_in_order(self):
        blob = self.A + f_str(18, "project-1") + self.B
        assert shaped.read_workspaces(blob) == [
            {"ordinal": 0, "workspace_uri": "file:///home/someone/ComfyUI/custom_nodes/My-Repo",
             "git_root_uri": "file:///home/someone/ComfyUI/custom_nodes/My-Repo", "repo_slug": "handle/My-Repo",
             "git_remote": "git@github.com:handle/My-Repo.git", "git_branch": "main"},
            {"ordinal": 1, "workspace_uri": "file:///Users/another/workspace/my-repo",
             "git_root_uri": "file:///Users/another/workspace/my-repo", "repo_slug": "mirror-user/my-repo",
             "git_remote": "ssh://git@10.0.0.5:2222/mirror-user/my-repo", "git_branch": "feature/x"}]
        assert shaped.read_workspaces(f_str(18, "project-1")) == []
        assert shaped.read_workspaces(workspace_entry("file:///x/y"))[0] == {
            "ordinal": 0, "workspace_uri": "file:///x/y", "git_root_uri": None, "repo_slug": None,
            "git_remote": None, "git_branch": None}
        with pytest.raises(ValueError):
            shaped.read_workspaces(b"\xff\xff\xff")

    def _conversations(self, cli_home, lake_root, blobs, uris=None):
        for conv, blob in blobs.items():
            _set_blob(cli_home, conv, blob)
        if uris is not None:
            (cli_home / "antigravity-cli" / "conversation_summaries.db").unlink()
            write_summaries_db(cli_home / "antigravity-cli" / "conversation_summaries.db", [
                {"conversation_id": conv, "step_count": 1, "app_data_dir": "antigravity-cli", "workspace_uris": uri}
                for conv, uri in uris.items()])
        lake(cli_home, lake_root)
        shaped.shape_lake(lake_root)
        return dict((r[0], r[1:]) for r in views(lake_root).execute(
            "SELECT conversation_id, project, project_source, repo_slug, git_remote, git_branch, project_path "
            "FROM antigravity_conversations").fetchall())

    def test_one_repository_at_two_paths_from_two_remotes_is_one_project(self, cli_home, lake_root):
        rows = self._conversations(cli_home, lake_root, {CONV_A: self.A, CONV_B: self.B}, {
            CONV_A: '["file:///home/someone/ComfyUI/custom_nodes/My-Repo"]',
            CONV_B: '["file:///Users/another/workspace/my-repo"]'})
        assert rows[CONV_A] == ("my-repo", "git_remote", "handle/My-Repo", "git@github.com:handle/My-Repo.git",
                                "main", "ComfyUI/custom_nodes/My-Repo")
        assert rows[CONV_B] == ("my-repo", "git_remote", "mirror-user/my-repo",
                                "ssh://git@10.0.0.5:2222/mirror-user/my-repo", "feature/x", "workspace/my-repo")

    def test_the_first_workspace_is_the_conversations(self, cli_home, lake_root):
        rows = self._conversations(cli_home, lake_root, {CONV_A: self.B + self.A})
        assert rows[CONV_A][:3] == ("my-repo", "git_remote", "mirror-user/my-repo")

    def test_with_no_remote_stated_the_project_is_the_workspaces_directory(self, cli_home, lake_root):
        rows = self._conversations(cli_home, lake_root, {CONV_A: workspace_entry("file:///srv/build/My%20Repo/")}, {
            CONV_A: '["file:///srv/build/My%20Repo/"]', CONV_B: "[]"})
        assert rows[CONV_A] == ("my repo", "workspace_path", None, None, None, "/srv/build/My Repo")
        # No workspace stated anywhere: no project, and no source for one.
        assert rows[CONV_B] == (None, None, None, None, None, None)

    def test_a_blob_that_does_not_parse_is_a_row_with_its_error_and_the_path_still_names_the_project(self, cli_home, lake_root):
        rows = self._conversations(cli_home, lake_root, {CONV_A: b"\xff\xff\xff"}, {
            CONV_A: '["file:///home/someone/workspace/proj-a"]', CONV_B: "[]"})
        assert rows[CONV_A] == ("proj-a", "workspace_path", None, None, None, "workspace/proj-a")
        table = unit_table(lake_root, CONV_A, "workspace")
        assert len(table) == 1 and table[0]["decode_error"] and table[0]["ordinal"] is None

    def test_a_machine_decoded_before_workspaces_still_reads_with_the_path_as_its_project(self, cli_home, tmp_path):
        _set_blob(cli_home, CONV_A, self.A)
        out, elsewhere = tmp_path / "out", tmp_path / "elsewhere"
        assert decode_cmd(cli_home, elsewhere, "--decode", "--machine", "laptop").exit_code == 0
        assert decode_cmd(cli_home, out, "--decode", "--machine", "desk").exit_code == 0
        shutil.copytree(elsewhere / "machine=laptop", out / "machine=laptop")
        for old in (out / "machine=laptop").rglob("workspace.parquet"):
            old.unlink()
        conn = duckdb.connect()
        shaped.create_views(conn, out)
        assert conn.execute("SELECT machine, project, project_source FROM antigravity_conversations "
                            "WHERE conversation_id = ? ORDER BY 1", [CONV_A]).fetchall() == [
            ("desk", "my-repo", "git_remote"), ("laptop", "project", "workspace_path")]


class TestProgressOutput:
    """What a person watching the run sees. Progress goes to stderr, so the
    summary on stdout is the same with or without it."""

    def test_verbose_names_every_unit_it_wrote_and_decoded(self, cli_home, tmp_path):
        out = tmp_path / "out"
        result = decode_cmd(cli_home, out, "--decode", "--verbose")
        assert result.exit_code == 0, result.output
        lines = result.stderr.splitlines()
        named = [l for l in lines if f"antigravity-cli/conversation/{CONV_A}" in l]
        mirrored = [l for l in named if " mirror " in l]
        assert len(mirrored) == 1 and "written" in mirrored[0] and "steps 4" in mirrored[0]
        assert re.search(r"\[\s*\d+/\d+\]", mirrored[0])
        decoded = [l for l in named if " decode " in l]
        assert len(decoded) == 1 and "steps 4" in decoded[0]
        assert len(named) == 2
        assert "\r" not in result.stderr
        # The summary is still on stdout, and none of the progress is.
        assert "Lake:" in result.stdout and CONV_A not in result.stdout

    def test_verbose_says_nothing_per_unit_when_nothing_changed(self, cli_home, tmp_path):
        out = tmp_path / "out"
        decode_cmd(cli_home, out, "--decode")
        result = decode_cmd(cli_home, out, "--decode", "--verbose")
        assert result.exit_code == 0, result.output
        assert CONV_A not in result.stderr

    def test_without_a_terminal_or_the_flag_nothing_is_added(self, cli_home, tmp_path):
        result = decode_cmd(cli_home, tmp_path / "out", "--decode")
        assert result.exit_code == 0, result.output
        assert result.stderr == ""

    def test_verbose_shows_a_failed_unit_as_it_fails(self, cli_home, tmp_path):
        db = cli_home / "antigravity-cli" / "conversations" / f"{CONV_A}.db"
        db.write_bytes(b"not a database")
        result = decode_cmd(cli_home, tmp_path / "out", "--verbose")
        assert result.exit_code == 1
        failed = [l for l in result.stderr.splitlines() if CONV_A in l and "[" in l]
        assert failed and "error" in failed[0]


class TestLiveLine:
    """On a terminal, one line is redrawn in place and removed at the end."""

    def _events(self, n=3, status="written"):
        for i in range(1, n + 1):
            yield UnitEvent("mirror", i, n, "antigravity-cli", "conversation", f"{i:08d}-aaaa-bbbb-cccc-dddddddddddd")
            yield UnitEvent("mirror", i, n, "antigravity-cli", "conversation",
                            f"{i:08d}-aaaa-bbbb-cccc-dddddddddddd", status=status, rows={"steps": i})

    def test_it_redraws_one_line_and_leaves_none_behind(self):
        stream = io.StringIO()
        reporter = Reporter(stream, live=True, verbose=False, width=lambda: 60)
        reporter.phase("Scanning")
        for event in self._events():
            reporter.unit(event)
        reporter.clear()
        text = stream.getvalue()
        assert "\n" not in text
        assert "Scanning" in text and "[3/3]" in text and "00000003" in text
        assert text.endswith("\r\x1b[K")
        drawn = [d for d in text.split("\r") if d]
        assert all(len(d.replace("\x1b[K", "")) < 60 for d in drawn)

    def test_a_unit_that_starts_work_is_always_drawn_and_unchanged_ones_are_throttled(self):
        clock = iter(x * 0.001 for x in range(10_000))
        stream = io.StringIO()
        reporter = Reporter(stream, live=True, verbose=False, width=lambda: 80, clock=lambda: next(clock))
        for i in range(1, 501):
            reporter.unit(UnitEvent("mirror", i, 501, "s", "conversation", f"unit-{i}", status="unchanged"))
        assert stream.getvalue().count("\r") < 50
        reporter.unit(UnitEvent("mirror", 501, 501, "s", "conversation", "the-slow-one"))
        assert "the-slow-one" in stream.getvalue()

    def test_neither_live_nor_verbose_writes_nothing(self):
        stream = io.StringIO()
        reporter = Reporter(stream, live=False, verbose=False)
        reporter.phase("Scanning")
        for event in self._events():
            reporter.unit(event)
        reporter.clear()
        assert stream.getvalue() == ""


def test_the_readmes_antigravity_recipes_bind_against_the_views(cli_home, lake_root):
    """The README shows SQL over these views. tests/test_query_recipes.py binds
    every ```sql block against the warehouse, which does not have them, so
    the Antigravity blocks are fenced ```sql antigravity and held here."""
    import re
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    blocks = re.findall(r"^```sql antigravity\n(.*?)^```", readme, re.M | re.S)
    assert blocks, "no ```sql antigravity block left in README.md"
    TestTasks()._add_tasks(cli_home)
    lake(cli_home, lake_root)
    shaped.shape_lake(lake_root)
    conn = views(lake_root)
    for block in blocks:
        code = "\n".join(line.split("--", 1)[0] for line in block.splitlines())
        for statement in (s.strip() for s in code.split(";")):
            if statement:
                # Counted inside the engine: fetching a TIMESTAMPTZ into Python
                # needs pytz, which is not a dependency and not the recipe's concern.
                rows = conn.execute(f"SELECT count(*) FROM ({statement})").fetchone()[0]
                assert rows > 0, f"a README recipe returns nothing on the fixture store:\n{statement}"
