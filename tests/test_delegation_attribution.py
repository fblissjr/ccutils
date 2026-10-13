"""Who made a delegation, and which agent it spawned, when the transcript alone cannot say.

`semantic_agent_delegations` and the `delegation_rollup` of
`semantic_session_summary` read the Agent / Task rows of `fact_tool_calls`.
Two things made those rows say the wrong thing about agents that delegate:

- The child was found only through `agent_id`, which comes from the spawning
  call's `toolUseResult`. An agent transcript often states no such payload
  (docs/JSONL_CONTRACT.md claim 11), and a background launch made from inside
  an agent states none at all, so the delegation had no child, no completion
  state and no derived figures while the child's transcript sat in the
  warehouse.
- A fork's transcript replays its spawner's history, including the call that
  created the fork. `fact_tool_calls` holds one row per `tool_use_id`, so the
  copy does not add a row: it takes the row over when it loads later, and the
  delegation then belongs to the fork, which appears to have spawned itself.

The spawned agent's sidecar states both facts from the other side: the call
that spawned it (`toolUseId`) and who made that call (`parentAgentId`, or the
root session when `spawnDepth` is 1). These tests hold the views to that.

Fixtures model the shape measured on the real corpus on 2026-10-13: a copied
call keeps its id and timestamp, gets new entry uuids, and its result is the
text "Fork started — processing in background" with no payload.

Delete this file and a delegation made inside an agent goes back to having no
child, and whichever transcript loads last decides who made it.
"""

from __future__ import annotations

import json

import pytest

from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl

ROOT = "root-0001"
ASYNC_ACK = "Async agent launched successfully."
FORK_ACK = "Fork started — processing in background"


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _call(tool_id, at, *, text=ASYNC_ACK, payload=None, subagent_type="general-purpose"):
    """One Agent call and its result. `at` is HH:MM:SS; a copy shares it."""
    return {"id": tool_id, "at": at, "text": text, "payload": payload,
            "subagent_type": subagent_type}


def _launch_payload(agent_id):
    """What a parent states for a background launch: an id, never a rollup."""
    return {"isAsync": True, "status": "async_launched", "agentId": agent_id,
            "description": "task", "resolvedModel": "claude-sonnet-5"}


def _transcript(tmp_path, stem, calls=(), *, sidecar=None, start="10:00:00",
                end="10:30:00", out_tokens=100):
    """Write a root session (stem == ROOT) or an agent file below it.

    Every line carries the ROOT's sessionId, as real agent files do; identity
    is the file stem. Entry uuids are prefixed with the stem, so a call copied
    into a second file has the same tool_use id under different entries.
    """
    project = tmp_path / "projects" / "-work-demo"
    if stem == ROOT:
        path = project / f"{ROOT}.jsonl"
    else:
        path = project / ROOT / "subagents" / f"{stem}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    def entry(kind, name, clock, content, **extra):
        message = {"role": kind, "content": content}
        if kind == "assistant":
            message["model"] = "claude-sonnet-5"
        return {"type": kind, "uuid": f"{stem}-{name}", "sessionId": ROOT,
                "timestamp": f"2026-07-17T{clock}.000Z", "cwd": "/work",
                "version": "2.1.212", "message": message, **extra}

    lines = [entry("user", "u0", start, "go")]
    for i, call in enumerate(calls):
        lines.append(entry("assistant", f"a{i}", call["at"], [
            {"type": "tool_use", "id": call["id"], "name": "Agent",
             "input": {"description": f"task {call['id']}", "prompt": "do it",
                       "subagent_type": call["subagent_type"]}}],
            requestId=f"{stem}-req{i}"))
        result = entry("user", f"r{i}", call["at"][:-2] + "59", [
            {"type": "tool_result", "tool_use_id": call["id"], "content": call["text"]}])
        if call["payload"] is not None:
            result["toolUseResult"] = call["payload"]
        lines.append(result)
    final = entry("assistant", "done", end, [{"type": "text", "text": f"{stem} done"}],
                  requestId=f"{stem}-final")
    final["message"]["stop_reason"] = "end_turn"
    final["message"]["usage"] = {"input_tokens": 10, "output_tokens": out_tokens}
    lines.append(final)

    path.write_text("\n".join(json.dumps(line) for line in lines))
    if sidecar is not None:
        path.with_suffix(".meta.json").write_text(json.dumps(sidecar))
    return path


def _load(conn, tmp_path, *paths):
    for path in paths:
        run_v15_etl(conn, path, project_name="demo", parquet_lake_root=tmp_path / "lake")


def _edges(conn):
    """Every delegation as (call, parent, child), which is what must not move."""
    return conn.execute(
        "SELECT tool_use_id, parent_session_id, agent_session_id "
        "FROM semantic_agent_delegations ORDER BY tool_use_id"
    ).fetchall()


def _summary(conn, session_id):
    return conn.execute(
        "SELECT total_delegations, total_spawn_failures, max_child_depth, "
        "delegated_output_tokens FROM semantic_session_summary WHERE session_id = ?",
        [session_id],
    ).fetchone()


class TestANestedDelegationFindsItsChildThroughTheSidecar:
    """A background launch made inside an agent states no payload at all."""

    def _build(self, conn, tmp_path):
        spawner = _transcript(
            tmp_path, "agent-p1", [_call("tu_nested", "10:05:00")],
            sidecar={"agentType": "general-purpose", "toolUseId": "tu_root_p", "spawnDepth": 1})
        child = _transcript(
            tmp_path, "agent-c1", start="10:06:00", end="10:09:00", out_tokens=500,
            sidecar={"agentType": "general-purpose", "toolUseId": "tu_nested",
                     "parentAgentId": "p1", "spawnDepth": 2})
        # Spawner first, child second: the order a batch run uses.
        _load(conn, tmp_path, spawner, child)

    def test_the_child_is_resolved_and_its_outcome_derived(self, conn, tmp_path):
        self._build(conn, tmp_path)
        row = conn.execute(
            "SELECT parent_session_id, agent_id, agent_session_id, agent_link_source, "
            "       completion_state, derived_io_tokens, derived_output_text "
            "FROM semantic_agent_delegations WHERE tool_use_id = 'tu_nested'"
        ).fetchone()
        assert row == ("agent-p1", None, "agent-c1", "child_sidecar",
                       "completed", 510, "agent-c1 done")

    def test_the_parent_stated_nothing_and_nothing_is_put_in_its_columns(self, conn, tmp_path):
        """`agent_id` stays what the PARENT stated. The link has its own column."""
        self._build(conn, tmp_path)
        row = conn.execute(
            "SELECT agent_id, agent_status, agent_total_tokens, agent_total_duration_ms "
            "FROM semantic_agent_delegations WHERE tool_use_id = 'tu_nested'"
        ).fetchone()
        assert row == (None, None, None, None)

    def test_the_summary_counts_the_child_on_the_agent_that_spawned_it(self, conn, tmp_path):
        self._build(conn, tmp_path)
        assert _summary(conn, "agent-p1") == (1, 0, 2, 500)
        assert _summary(conn, "agent-c1")[0] == 0

    def test_a_payload_that_names_the_agent_is_still_the_first_source(self, conn, tmp_path):
        """Where the parent states an id, that is the link, sidecar or not."""
        root = _transcript(tmp_path, ROOT, [
            _call("tu_stated", "10:01:00", payload=_launch_payload("s1")),
            _call("tu_absent", "10:02:00", payload=_launch_payload("gone"))])
        agent = _transcript(
            tmp_path, "agent-s1", start="10:01:30", end="10:04:00",
            sidecar={"agentType": "general-purpose", "toolUseId": "tu_stated", "spawnDepth": 1})
        _load(conn, tmp_path, root, agent)
        rows = conn.execute(
            "SELECT tool_use_id, agent_id, agent_session_id, agent_link_source "
            "FROM semantic_agent_delegations ORDER BY tool_use_id"
        ).fetchall()
        assert rows == [
            # Stated by the parent, transcript never loaded: the link is still stated.
            ("tu_absent", "gone", None, "parent_payload"),
            ("tu_stated", "s1", "agent-s1", "parent_payload"),
        ]

    def test_a_call_no_side_links_has_no_source(self, conn, tmp_path):
        spawner = _transcript(
            tmp_path, "agent-p1", [_call("tu_orphan", "10:05:00")],
            sidecar={"agentType": "general-purpose", "spawnDepth": 1})
        _load(conn, tmp_path, spawner)
        assert conn.execute(
            "SELECT agent_session_key, agent_link_source, completion_state "
            "FROM semantic_agent_delegations WHERE tool_use_id = 'tu_orphan'"
        ).fetchone() == (None, None, None)


def _spawner_and_fork(tmp_path):
    """An agent forks. The fork's file replays the call that created it."""
    spawner = _transcript(
        tmp_path, "agent-p1",
        [_call("tu_fork", "10:05:00", payload=_launch_payload("f1"), subagent_type="fork")],
        sidecar={"agentType": "general-purpose", "toolUseId": "tu_root_p", "spawnDepth": 1})
    fork = _transcript(
        tmp_path, "agent-f1",
        [_call("tu_fork", "10:05:00", text=FORK_ACK, subagent_type="fork")],
        end="10:20:00", out_tokens=700,
        sidecar={"agentType": "fork", "isFork": True, "toolUseId": "tu_fork",
                 "parentAgentId": "p1", "spawnDepth": 2})
    return spawner, fork


class TestAForkDoesNotOwnTheCallThatCreatedIt:
    """One call, two transcripts, one row: load order must not decide the parent."""

    @pytest.fixture(params=["spawner_first", "fork_first"])
    def built(self, request, conn, tmp_path):
        spawner, fork = _spawner_and_fork(tmp_path)
        order = (spawner, fork) if request.param == "spawner_first" else (fork, spawner)
        _load(conn, tmp_path, *order)
        return conn

    def test_the_delegation_belongs_to_the_spawner_in_either_order(self, built):
        assert _edges(built) == [("tu_fork", "agent-p1", "agent-f1")]

    def test_no_delegation_has_itself_as_its_child(self, built):
        assert built.execute(
            "SELECT COUNT(*) FROM semantic_agent_delegations "
            "WHERE parent_session_key = agent_session_key"
        ).fetchone()[0] == 0

    def test_the_parent_is_marked_as_stated_by_the_sidecar(self, built):
        assert built.execute(
            "SELECT parent_link_source, completion_state, derived_io_tokens "
            "FROM semantic_agent_delegations WHERE tool_use_id = 'tu_fork'"
        ).fetchone() == ("child_sidecar", "completed", 710)

    def test_the_summary_counts_it_once_on_the_spawner(self, built):
        assert _summary(built, "agent-p1") == (1, 0, 2, 700)
        assert _summary(built, "agent-f1") == (0, 0, None, 0)

    def test_the_fixture_reproduces_the_takeover(self, conn, tmp_path):
        """A control on the fixture: the copy really does take the row.

        If this stops holding, the tests above pass without exercising the
        re-attribution, and say nothing.
        """
        spawner, fork = _spawner_and_fork(tmp_path)
        _load(conn, tmp_path, spawner, fork)
        assert conn.execute(
            "SELECT session_id, result_payload_json IS NULL FROM fact_tool_calls "
            "WHERE tool_use_id = 'tu_fork' AND NOT is_deleted"
        ).fetchall() == [("agent-f1", True)]


class TestACallCopiedFromAnAncestorStaysWithTheAncestor:
    """A fork replays every earlier spawn of its spawner, not only its own."""

    def _build(self, conn, tmp_path, *, root_spawner):
        """The spawner makes two calls: a child, then a fork that copies both."""
        if root_spawner:
            spawner_stem, depth, parent = ROOT, 1, {}
            spawner_sidecar = None
        else:
            spawner_stem, depth, parent = "agent-p1", 2, {"parentAgentId": "p1"}
            spawner_sidecar = {"agentType": "general-purpose", "toolUseId": "tu_root_p",
                               "spawnDepth": 1}
        spawner = _transcript(tmp_path, spawner_stem, [
            _call("tu_child", "10:03:00", payload=_launch_payload("c1")),
            _call("tu_fork", "10:05:00", payload=_launch_payload("f1"), subagent_type="fork"),
        ], sidecar=spawner_sidecar)
        child = _transcript(
            tmp_path, "agent-c1", start="10:03:30", end="10:08:00", out_tokens=300,
            sidecar={"agentType": "general-purpose", "toolUseId": "tu_child",
                     "spawnDepth": depth, **parent})
        fork = _transcript(tmp_path, "agent-f1", [
            _call("tu_child", "10:03:00", text=ASYNC_ACK),
            _call("tu_fork", "10:05:00", text=FORK_ACK, subagent_type="fork"),
        ], end="10:20:00", out_tokens=700,
            sidecar={"agentType": "fork", "isFork": True, "toolUseId": "tu_fork",
                     "spawnDepth": depth, **parent})
        # The fork last, so both copies take their rows over.
        _load(conn, tmp_path, spawner, child, fork)
        return spawner_stem

    @pytest.mark.parametrize("root_spawner", [False, True], ids=["agent", "root"])
    def test_both_delegations_stay_with_the_session_that_made_them(
        self, conn, tmp_path, root_spawner
    ):
        spawner = self._build(conn, tmp_path, root_spawner=root_spawner)
        assert _edges(conn) == [
            ("tu_child", spawner, "agent-c1"),
            ("tu_fork", spawner, "agent-f1"),
        ]

    @pytest.mark.parametrize("root_spawner", [False, True], ids=["agent", "root"])
    def test_the_fork_delegated_nothing(self, conn, tmp_path, root_spawner):
        spawner = self._build(conn, tmp_path, root_spawner=root_spawner)
        assert _summary(conn, "agent-f1")[0] == 0
        assert _summary(conn, spawner)[::3] == (2, 1000)


class TestAnUnstatedSpawnerIsNotInvented:
    """A sidecar that names the call but not who made it settles less."""

    def test_the_transcript_holding_the_call_stays_the_parent(self, conn, tmp_path):
        """Nested by `spawnDepth`, no `parentAgentId`: the root is NOT assumed."""
        spawner = _transcript(
            tmp_path, "agent-p1", [_call("tu_nested", "10:05:00")],
            sidecar={"agentType": "general-purpose", "toolUseId": "tu_root_p", "spawnDepth": 1})
        child = _transcript(
            tmp_path, "agent-c1", start="10:06:00", end="10:09:00",
            sidecar={"agentType": "general-purpose", "toolUseId": "tu_nested", "spawnDepth": 2})
        _load(conn, tmp_path, spawner, child)
        assert conn.execute(
            "SELECT parent_session_id, parent_link_source, agent_session_id, agent_link_source "
            "FROM semantic_agent_delegations WHERE tool_use_id = 'tu_nested'"
        ).fetchone() == ("agent-p1", "transcript", "agent-c1", "child_sidecar")

    def test_a_session_holding_only_its_own_spawn_delegated_nothing(self, conn, tmp_path):
        """The call created this session, and nothing says who made it.

        It is provably not this session's delegation and cannot be given to
        anyone else, so it is no row at all. This is the one case that is
        dropped; every other copied call is attributed.
        """
        fork = _transcript(
            tmp_path, "agent-f1", [_call("tu_self", "10:05:00", text=FORK_ACK)],
            sidecar={"agentType": "fork", "isFork": True, "toolUseId": "tu_self",
                     "spawnDepth": 2})
        _load(conn, tmp_path, fork)
        assert _edges(conn) == []
        assert _summary(conn, "agent-f1") == (0, 0, None, 0)


class TestOneCallIsOneDelegation:
    def test_two_sidecars_naming_one_call_do_not_double_it(self, conn, tmp_path):
        """Unseen in the corpus; a view must still not fan out if it happens."""
        spawner = _transcript(
            tmp_path, "agent-p1", [_call("tu_twice", "10:05:00")],
            sidecar={"agentType": "general-purpose", "toolUseId": "tu_root_p", "spawnDepth": 1})
        twins = [
            _transcript(tmp_path, stem, start="10:06:00", end="10:09:00",
                        sidecar={"agentType": "general-purpose", "toolUseId": "tu_twice",
                                 "parentAgentId": "p1", "spawnDepth": 2})
            for stem in ("agent-c1", "agent-c2")
        ]
        _load(conn, tmp_path, spawner, *twins)
        assert conn.execute(
            "SELECT COUNT(*) FROM semantic_agent_delegations WHERE tool_use_id = 'tu_twice'"
        ).fetchone()[0] == 1
        assert _summary(conn, "agent-p1")[0] == 1
