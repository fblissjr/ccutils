"""A fork's replayed prefix is not the fork's work, in the facts computed per session.

A fork's transcript begins with a copy of its spawner's conversation, up to
and including the call that created it, and is the fork's own from there
(docs/JSONL_CONTRACT.md claim 12). The views leave replayed tool calls out
through `semantic_tool_call_replay`. The facts computed at load time cannot
use that view's cross-session rule, but they do not need it for a fork: the
cut is the timestamp of the fork's own spawn call, which its sidecar names
and its own transcript holds. It is stored once, as
`dim_session.fork_cut_timestamp`, and everything that describes a session
reads only what came after it.

The fixture is the shape measured on the corpus on 2026-10-13: in every fork
transcript the first entry after the spawn call is one user entry holding
both the "Fork started" tool result and the fork's own prompt as text.

What each test guards: delete it and a fork is described by the conversation
it inherited (its spawner's first prompt, tools, files, plans, tokens and
delegations) instead of by what it did.
"""

from __future__ import annotations

import json

import pytest
from test_delegation_attribution import (
    FORK_ACK,
    _call,
    _launch_payload,
    _load,
    _transcript,
)

from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl

SPAWNER_PROMPT = "Refactor the billing module and write a migration plan"
FORK_PROMPT = "Fix the failing date parser test in the fork"


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _inherited(*, in_fork):
    """The spawner's history up to the fork: a read, a plan, a spawn, the fork.

    The same four calls sit in both transcripts. In the fork's copy the last
    one's result is the acknowledgment plus the fork's own prompt.
    """
    return [
        _call("tu_read", "10:02:00", name="Read", tool_input={"file_path": "/work/billing.py"},
              text="BODY", usage={"input_tokens": 2000, "output_tokens": 1000},
              payload={"type": "text", "file": {"filePath": "/work/billing.py",
                                                "content": "BODY", "numLines": 3,
                                                "startLine": 1, "totalLines": 3}}),
        _call("tu_plan", "10:03:00", name="ExitPlanMode", tool_input={"plan": "1. migrate"},
              text="User has approved your plan.", payload={"plan": "1. migrate"},
              usage={"input_tokens": 300, "output_tokens": 200}),
        _call("tu_child", "10:04:00", payload=_launch_payload("c1"),
              usage={"input_tokens": 400, "output_tokens": 100}),
        _call("tu_fork", "10:05:00", subagent_type="fork",
              usage={"input_tokens": 5000, "output_tokens": 3000},
              **({"text": FORK_ACK, "prompt": FORK_PROMPT} if in_fork
                 else {"payload": _launch_payload("f1")})),
    ]


def _spawner(tmp_path):
    return _transcript(
        tmp_path, "agent-p1", _inherited(in_fork=False), first_prompt=SPAWNER_PROMPT,
        end="10:50:00", out_tokens=900,
        sidecar={"agentType": "general-purpose", "toolUseId": "tu_root_p", "spawnDepth": 1})


def _fork(tmp_path):
    own = _call("tu_edit", "10:07:00", name="Edit",
                tool_input={"file_path": "/work/dates.ts", "old_string": "a", "new_string": "b"},
                text="edited", usage={"input_tokens": 40, "output_tokens": 60},
                payload={"filePath": "/work/dates.ts", "structuredPatch": []})
    return _transcript(
        tmp_path, "agent-f1", [*_inherited(in_fork=True), own],
        first_prompt=SPAWNER_PROMPT, end="10:20:00", out_tokens=700,
        sidecar={"agentType": "fork", "isFork": True, "toolUseId": "tu_fork",
                 "parentAgentId": "p1", "spawnDepth": 2})


@pytest.fixture(params=["fork_alone", "spawner_then_fork", "fork_then_spawner"])
def loaded(request, conn, tmp_path):
    """The fork's own rows must not depend on whether its spawner is loaded, or when."""
    spawner, fork = _spawner(tmp_path), _fork(tmp_path)
    order = {"fork_alone": (fork,), "spawner_then_fork": (spawner, fork),
             "fork_then_spawner": (fork, spawner)}[request.param]
    _load(conn, tmp_path, *order)
    return conn


def _facet(conn, session_id, facet_id):
    row = conn.execute(
        "SELECT f.value_text, f.value_json, f.value_numeric, f.value_bool "
        "FROM fact_session_facets f JOIN dim_facet_type t USING (facet_type_key) "
        "WHERE f.session_id = ? AND t.facet_id = ? AND NOT f.is_deleted",
        [session_id, facet_id],
    ).fetchone()
    assert row is not None, f"no {facet_id} row for {session_id}"
    text, as_json, numeric, as_bool = row
    if as_json is not None:
        return json.loads(as_json)
    return next((v for v in (text, numeric, as_bool) if v is not None), None)


class TestTheCutIsStoredOnce:
    def test_a_fork_states_where_its_own_history_starts(self, loaded):
        assert loaded.execute(
            "SELECT strftime(fork_cut_timestamp, '%H:%M:%S') FROM dim_session "
            "WHERE session_id = 'agent-f1'"
        ).fetchone() == ("10:05:00",)

    def test_only_a_fork_has_one(self, conn, tmp_path):
        """`toolUseId` alone is not a cut: an ordinary agent does not hold its spawn."""
        _load(conn, tmp_path, _spawner(tmp_path), _fork(tmp_path))
        assert conn.execute(
            "SELECT session_id, fork_cut_timestamp IS NOT NULL FROM dim_session ORDER BY 1"
        ).fetchall() == [("agent-f1", True), ("agent-p1", False)]

    def test_a_fork_whose_spawn_is_not_in_its_transcript_has_none(self, conn, tmp_path):
        fork = _transcript(tmp_path, "agent-f1", [
            _call("tu_other", "10:02:00", name="Read", tool_input={"file_path": "/work/a.py"}),
        ], sidecar={"agentType": "fork", "isFork": True, "toolUseId": "tu_not_here",
                    "parentAgentId": "p1", "spawnDepth": 2})
        _load(conn, tmp_path, fork)
        assert conn.execute(
            "SELECT fork_cut_timestamp FROM dim_session WHERE session_id = 'agent-f1'"
        ).fetchone() == (None,)

    def test_the_replay_view_reads_the_same_cut(self, loaded):
        """One definition: the view marks exactly the calls at or before it."""
        assert loaded.execute(
            "SELECT tool_use_id, is_replay FROM semantic_tool_call_replay "
            "WHERE session_id = 'agent-f1' ORDER BY 1"
        ).fetchall() == [("tu_child", True), ("tu_edit", False), ("tu_fork", True),
                         ("tu_plan", True), ("tu_read", True)]


class TestAForkIsDescribedByItsOwnWork:
    def test_its_first_prompt_is_its_own(self, loaded):
        """Not the spawner's first prompt, which opens the fork's transcript."""
        first, intent = loaded.execute(
            "SELECT first_user_message, intent FROM dim_session WHERE session_id = 'agent-f1'"
        ).fetchone()
        assert FORK_PROMPT in first
        assert SPAWNER_PROMPT not in first
        # 'Refactor ...' classifies as refactoring; the fork's prompt as a bug fix.
        assert intent != "refactoring"

    def test_its_complexity_counts_its_own_calls_and_messages(self, conn, tmp_path):
        """Complexity is scored from tool and message counts, and both were inherited.

        The fork's transcript holds five tool calls and twelve messages; one
        call and four messages are its own. Nested one level, that is the
        difference between 'moderate' (either count over its threshold) and
        'simple' (neither). The spawner, which did make five calls, stays
        where its own counts put it.
        """
        _load(conn, tmp_path, _spawner(tmp_path), _fork(tmp_path))
        rows = dict(conn.execute(
            "SELECT session_id, complexity FROM dim_session").fetchall())
        counts = conn.execute(
            "SELECT COUNT(*) FILTER (WHERE timestamp > TIMESTAMP '2026-07-17 10:05:00'), COUNT(*) "
            "FROM fact_messages WHERE session_id = 'agent-f1' AND NOT is_deleted"
        ).fetchone()
        assert counts == (4, 12), "the fixture no longer straddles the thresholds"
        assert rows["agent-f1"] == "simple"

    def test_its_tool_histogram_is_its_own(self, loaded):
        assert _facet(loaded, "agent-f1", "F06") == [{"tool": "Edit", "count": 1}]

    def test_its_files_are_its_own(self, loaded):
        assert _facet(loaded, "agent-f1", "F08") == 1.0
        assert _facet(loaded, "agent-f1", "F09") == ["ts"]

    def test_its_tokens_are_its_own(self, loaded):
        """The Edit's response and its final one: 40 + 10 in, 60 + 700 out."""
        assert _facet(loaded, "agent-f1", "F15") == 50.0
        assert _facet(loaded, "agent-f1", "F30") == 760.0
        assert sum(m["tokens"] for m in _facet(loaded, "agent-f1", "F11")) == 810

    def test_its_duration_starts_after_its_spawn(self, loaded):
        """10:05:59, the entry carrying its prompt, to 10:20:00. Not from 10:00:00."""
        assert _facet(loaded, "agent-f1", "F12") == 841.0

    def test_it_delegated_nothing_and_planned_nothing(self, loaded):
        """The spawn and the plan in its transcript are its spawner's."""
        assert _facet(loaded, "agent-f1", "F17") is False
        assert _facet(loaded, "agent-f1", "F19") is False
        assert loaded.execute(
            "SELECT COUNT(*) FROM fact_plan_revisions "
            "WHERE session_id = 'agent-f1' AND NOT is_deleted"
        ).fetchone()[0] == 0

    def test_the_spawner_keeps_everything_it_did(self, conn, tmp_path):
        """A control: the cut removes nothing from a session that is not a fork."""
        _load(conn, tmp_path, _spawner(tmp_path), _fork(tmp_path))
        assert sorted((t["tool"], t["count"]) for t in _facet(conn, "agent-p1", "F06")) == [
            ("Agent", 2), ("ExitPlanMode", 1), ("Read", 1)]
        assert _facet(conn, "agent-p1", "F17") is True
        assert _facet(conn, "agent-p1", "F19") is True
        assert _facet(conn, "agent-p1", "F15") == 2000 + 300 + 400 + 5000 + 10
        assert conn.execute(
            "SELECT COUNT(*) FROM fact_plan_revisions "
            "WHERE session_id = 'agent-p1' AND NOT is_deleted"
        ).fetchone()[0] == 1
        first = conn.execute(
            "SELECT first_user_message FROM dim_session WHERE session_id = 'agent-p1'"
        ).fetchone()[0]
        assert SPAWNER_PROMPT in first


class TestEveryFacetGetsTheCut:
    """The restriction is applied in one place, so a new facet cannot forget it."""

    def test_a_fact_read_is_limited_to_the_sessions_own_work(self):
        from ccutils.etl.fact_session_facets import _own_work_only

        sql = _own_work_only(
            "F90",
            "(SELECT COUNT(*) FROM fact_messages fm WHERE fm.is_deleted = FALSE "
            "AND fm.session_id = scope.session_id)",
        )
        assert "fm.timestamp > scope.fork_cut_timestamp" in sql
        assert "scope.fork_cut_timestamp IS NULL" in sql

    def test_a_facet_that_reads_no_fact_is_left_alone(self):
        from ccutils.etl.fact_session_facets import _own_work_only

        assert _own_work_only("F90", "scope.intent") == "scope.intent"

    def test_a_facet_that_cannot_be_scoped_is_refused(self):
        """Loud, at build time: an unscoped read would describe a fork by its prefix."""
        from ccutils.etl.fact_session_facets import _own_work_only

        with pytest.raises(ValueError, match="F90"):
            _own_work_only(
                "F90",
                "(SELECT COUNT(*) FROM fact_messages fm "
                "WHERE fm.session_id IN (SELECT session_id FROM dim_session))",
            )


class TestWhatLeavesTheMachineIsTheForksOwn:
    def test_tier_2_inputs_hold_no_inherited_conversation(self, conn, tmp_path):
        """The text sent to the API for a fork is the fork's, not its spawner's."""
        from ccutils.etl.facets import FacetOutput

        captured = []

        class _Spy:
            def extract(self, session_inputs, enabled_facets):
                captured.append(session_inputs)
                return {spec.facet_id: FacetOutput(
                    facet_id=spec.facet_id, prompt_version=spec.prompt_version,
                    value="canned") for spec in enabled_facets}

        run_v15_etl(conn, _fork(tmp_path), project_name="demo",
                    parquet_lake_root=tmp_path / "lake", facet_extractor=_Spy())
        assert len(captured) == 1
        sent = repr(captured[0])
        assert FORK_PROMPT in sent
        assert SPAWNER_PROMPT not in sent
        assert "Edit" in sent and "ExitPlanMode" not in sent and "Read" not in sent
        # From its cut at 10:05:00 to 10:20:00, not from the spawner's 10:00:00.
        assert captured[0].duration_seconds == 900


class TestADelegationToAForkCountsTheForksOwnWork:
    """The other reader of the cut: what a parent's delegation derives from its child."""

    def test_derived_figures_start_at_the_cut(self, conn, tmp_path):
        """Measured on two real builds: about three quarters of forks'
        `derived_io_tokens` was the spawner's replayed history, and every
        other child's figure was already its own."""
        _load(conn, tmp_path, _spawner(tmp_path), _fork(tmp_path))
        tokens, duration_ms, tool_uses = conn.execute(
            "SELECT derived_io_tokens, derived_duration_ms, derived_tool_use_count "
            "FROM semantic_agent_delegations WHERE tool_use_id = 'tu_fork'"
        ).fetchone()
        assert tokens == 40 + 60 + 10 + 700
        assert duration_ms == 841000.0
        assert tool_uses == 1

    def test_the_parents_summary_counts_the_forks_own_output(self, conn, tmp_path):
        """`delegated_output_tokens`: the fork's 60 + 700, not the replayed 4,300."""
        child = _transcript(
            tmp_path, "agent-c1", start="10:04:30", end="10:09:00", out_tokens=250,
            sidecar={"agentType": "general-purpose", "toolUseId": "tu_child",
                     "parentAgentId": "p1", "spawnDepth": 2})
        _load(conn, tmp_path, _spawner(tmp_path), _fork(tmp_path), child)
        assert conn.execute(
            "SELECT total_delegations, delegated_output_tokens FROM semantic_session_summary "
            "WHERE session_id = 'agent-p1'"
        ).fetchone() == (2, 760 + 250)

    def test_a_child_that_is_not_a_fork_is_counted_whole(self, conn, tmp_path):
        """A control: no cut, nothing withheld."""
        child = _transcript(
            tmp_path, "agent-c1", start="10:04:30", end="10:09:00", out_tokens=250,
            sidecar={"agentType": "general-purpose", "toolUseId": "tu_child",
                     "parentAgentId": "p1", "spawnDepth": 2})
        _load(conn, tmp_path, _spawner(tmp_path), child)
        assert conn.execute(
            "SELECT derived_io_tokens, derived_duration_ms FROM semantic_agent_delegations "
            "WHERE tool_use_id = 'tu_child'"
        ).fetchone() == (260, 270000.0)



class TestAForksTotalsAreItsOwn:
    """The session summary and the cost views: a fork's own messages and tokens.

    The fork's transcript holds twelve messages and six API responses; four
    messages and two responses came after its spawn. Its facets already said
    so, and the summary did not, so a fork's `F15` and its summary token
    total disagreed and its per-session cost read several times too high.

    Delete these and a fork is billed for the conversation it inherited.
    """

    def test_message_and_token_totals_start_at_the_cut(self, loaded):
        assert loaded.execute(
            "SELECT total_messages, user_messages, assistant_messages, "
            "       total_input_tokens, total_output_tokens, api_response_count "
            "FROM semantic_session_summary WHERE session_id = 'agent-f1'"
        ).fetchone() == (4, 2, 2, 50, 760, 2)

    def test_the_facets_and_the_summary_agree(self, loaded):
        tokens_in, tokens_out = loaded.execute(
            "SELECT total_input_tokens, total_output_tokens FROM semantic_session_summary "
            "WHERE session_id = 'agent-f1'"
        ).fetchone()
        assert _facet(loaded, "agent-f1", "F15") == tokens_in
        assert _facet(loaded, "agent-f1", "F30") == tokens_out

    def test_summed_over_sessions_a_response_is_counted_once(self, conn, tmp_path):
        """The spawner's 7,710 input tokens and the fork's own 50, not 7,710 twice."""
        _load(conn, tmp_path, _spawner(tmp_path), _fork(tmp_path))
        assert conn.execute(
            "SELECT SUM(total_input_tokens), "
            "       (SELECT SUM(input_tokens) FROM fact_token_usage WHERE NOT is_deleted) "
            "FROM semantic_session_summary"
        ).fetchone() == (7760, 15460)

    def test_a_session_that_is_not_a_fork_is_counted_whole(self, conn, tmp_path):
        """A control: no cut, nothing withheld. Ten messages, five responses."""
        _load(conn, tmp_path, _spawner(tmp_path), _fork(tmp_path))
        assert conn.execute(
            "SELECT total_messages, total_input_tokens, total_output_tokens, api_response_count "
            "FROM semantic_session_summary WHERE session_id = 'agent-p1'"
        ).fetchone() == (10, 7710, 5200, 5)

    @pytest.mark.parametrize("view", ["semantic_token_usage", "semantic_context_growth"])
    def test_a_per_response_view_marks_the_inherited_rows(self, loaded, view):
        """One row per response stays one row; a column says which were inherited.

        Dropping them would hide where a fork's context came from, and a
        consumer summing the view would then get the fork's own total without
        knowing rows were missing.
        """
        rows = loaded.execute(
            f"SELECT in_fork_prefix, COUNT(*) FROM {view} "
            "WHERE session_id = 'agent-f1' GROUP BY 1 ORDER BY 1"
        ).fetchall()
        assert rows == [(False, 2), (True, 4)]

    def test_no_other_session_has_inherited_rows(self, conn, tmp_path):
        """FALSE, never NULL, so `NOT in_fork_prefix` keeps every ordinary row."""
        _load(conn, tmp_path, _spawner(tmp_path), _fork(tmp_path))
        assert conn.execute(
            "SELECT DISTINCT in_fork_prefix FROM semantic_token_usage "
            "WHERE session_id = 'agent-p1'"
        ).fetchall() == [(False,)]
