"""Canaries for the upstream JSONL contract (docs/JSONL_CONTRACT.md).

These differ from the rest of the suite in what they assert against. Every
other test in this repo runs on synthetic fixtures, which means it asserts
that the code agrees with whoever wrote the fixture -- a suite in that shape
cannot falsify a premise about the format, and in 2026-08 it did not: the
agent-layout fixtures encoded a directory shape that had not existed for
years, every assertion passed, and the function under test returned nothing
on real data.

So these read the real corpus, and skip when it is absent. A skip is honest
here: the claim is about what Claude Code writes, and a machine with no
transcripts has nothing to say about it.

Each claim gets two tests:

- a corpus canary, which goes red when Claude Code's format changes;
- an oracle test, which feeds the same check a deliberately violating entry
  and asserts it is rejected.

The second is not ceremony. A corpus canary that cannot fail looks exactly
like one with nothing to report, and telling those apart after the fact is
the problem this whole file exists to avoid.
"""

import json
import random

import pytest

SAMPLE_SIZE = 60
SEED = 23


def _corpus_files():
    from pathlib import Path

    root = Path.home() / ".claude" / "projects"
    if not root.is_dir():
        return []
    return [f for f in root.glob("**/*.jsonl")]


@pytest.fixture(scope="module")
def corpus_sample():
    files = _corpus_files()
    if not files:
        pytest.skip("no local Claude Code corpus to check the contract against")
    random.seed(SEED)
    return random.sample(files, min(SAMPLE_SIZE, len(files)))


def _entries(paths, entry_type=None):
    for f in paths:
        try:
            lines = f.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if entry_type is None or obj.get("type") == entry_type:
                yield obj


# ---------------------------------------------------------------------------
# Claim 2: an assistant entry carries exactly one content block.
# ---------------------------------------------------------------------------


def single_block_violations(entries):
    """Assistant entries breaking the one-block rule, as (uuid, n, kinds).

    Extracted so the oracle test can feed it a violation directly. Entries
    whose content is not a list are not covered by the claim and are skipped.
    """
    bad = []
    for obj in entries:
        content = obj.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        kinds = {b.get("type") for b in content if isinstance(b, dict)}
        if len(content) != 1 or len(kinds) > 1:
            bad.append((obj.get("uuid"), len(content), sorted(k or "" for k in kinds)))
    return bad


class TestAssistantBlockShape:
    """Assistant entries are one-block in the overwhelming majority, not always.

    This class originally asserted ZERO multi-block entries, on a 120-file
    sample. It passed when written and went red hours later on a different
    draw, because the sample is taken from a growing corpus. The canary was
    right to fire: corpus-wide there are **17 violations in 198,230**
    assistant entries (0.009%), including entries mixing `text` with
    `tool_use` and `thinking`.

    So the absolute claim was wrong. Two things replace it.

    The DURABLE requirement is that the projection captures every text block
    regardless of how many there are -- that is what would actually lose
    prose, and it is testable deterministically without the corpus. Verified
    directly: an entry with text/thinking/tool_use/text yields both prose
    blocks in `content_text`.

    The CORPUS claim becomes a rate, not an absolute: if multi-block entries
    stop being a rounding error, `fact_messages`' one-row-per-entry grain and
    the reading that a NULL `content_text` on a tool-carrying row is "the
    format" both need revisiting. An external audit read that NULL as an 80%
    data-loss bug; it is not, but the margin is 0.009% rather than zero.
    """

    # Measured 2026-08-28 corpus-wide: 17 / 198,230 = 0.0086%. The threshold
    # is ~10x headroom -- it should catch "Claude Code started emitting
    # multi-block routinely", not normal drift.
    MAX_VIOLATION_RATE = 0.001

    def test_all_text_blocks_are_captured(self, tmp_path):
        """The requirement that actually matters, through the real ETL.

        The first version of this test built a dict literal and
        list-comprehended the text blocks back out of it -- it asserted that
        a list comprehension works, and would have passed unchanged if the
        SQL projection dropped every text block. Its docstring pointed at an
        ETL assertion elsewhere that did not exist. A test audit run the same
        morning classifies precisely that shape as decorative.

        So this one runs the pipeline. An entry carrying
        [text, thinking, tool_use, text] -- the real multi-block shape, 17 of
        which exist in the corpus -- must yield BOTH prose blocks in
        content_text, or claim 2's consequence ("a NULL content_text on a
        tool-carrying row is the format, not a loss") is false.
        """
        import json as _json

        from ccutils import create_star_schema
        from ccutils.etl.orchestrator import run_v15_etl

        src = tmp_path / "proj" / "mb.jsonl"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("\n".join(_json.dumps(e) for e in [
            {"type": "user", "uuid": "u1", "sessionId": "mb",
             "timestamp": "2026-01-15T10:00:00.000Z", "cwd": "/w",
             "message": {"role": "user", "content": "go"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
             "sessionId": "mb", "timestamp": "2026-01-15T10:00:01.000Z",
             "message": {"role": "assistant", "model": "claude-opus-5",
                         "content": [
                             {"type": "text", "text": "FIRST PROSE"},
                             {"type": "thinking", "thinking": ""},
                             {"type": "tool_use", "id": "t1", "name": "Bash",
                              "input": {"command": "ls"}},
                             {"type": "text", "text": "SECOND PROSE"}]}},
        ]))

        conn = create_star_schema(tmp_path / "mb.duckdb")
        run_v15_etl(conn, src, project_name="x",
                    parquet_lake_root=tmp_path / "lake")
        text, blocks = conn.execute(
            "SELECT content_text, content_block_count FROM fact_messages "
            "WHERE message_type = 'assistant'"
        ).fetchone()
        conn.close()

        assert blocks == 4
        assert "FIRST PROSE" in text and "SECOND PROSE" in text, (
            f"multi-block prose was dropped: {text!r}"
        )

    def test_multi_block_entries_stay_a_rounding_error(self):
        """Corpus-wide, not sampled: 17 in 198,230 when this was written.

        Scanned in full rather than sampled because the violation rate is far
        below what a 60-file sample can see -- which is exactly how the
        original absolute assertion managed to pass at first.
        """
        files = _corpus_files()
        if not files:
            pytest.skip("no local Claude Code corpus to check the contract against")

        total = violations = 0
        for f in files:
            try:
                lines = f.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                if '"assistant"' not in line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict) or obj.get("type") != "assistant":
                    continue
                content = obj.get("message", {}).get("content")
                if not isinstance(content, list):
                    continue
                total += 1
                kinds = {b.get("type") for b in content if isinstance(b, dict)}
                if len(content) != 1 or len(kinds) > 1:
                    violations += 1

        assert total, "corpus contained no assistant entries with list content"
        rate = violations / total
        assert rate < self.MAX_VIOLATION_RATE, (
            f"{violations} of {total} assistant entries ({rate:.4%}) carry "
            "multiple or mixed content blocks, over the "
            f"{self.MAX_VIOLATION_RATE:.1%} threshold. docs/JSONL_CONTRACT.md "
            "claim 2 needs revisiting, and with it fact_messages' grain and "
            "the reading that a NULL content_text on a tool-carrying row is "
            "the format rather than a loss."
        )


# ---------------------------------------------------------------------------
# Claim 5: thinking text is persisted on a minority of blocks.
# ---------------------------------------------------------------------------

# The first Claude Code version in this corpus whose thinking blocks carry
# text again after the long empty stretch. An identifier, not a measurement:
# it scopes the canary to the format being written now, so a change in new
# sessions is not diluted by months of older, uniformly empty transcripts.
THINKING_TEXT_SINCE = (2, 1, 257)

# Reasoned, not measured: the share at which "most reasoning is on disk"
# becomes true, and the word "minority" in docs/JSONL_CONTRACT.md claim 5 and
# in the reader's guide stops holding.
THINKING_TEXT_MAJORITY = 0.5


def _version_tuple(value):
    try:
        return tuple(int(part) for part in str(value).split("."))
    except ValueError:
        return None


def thinking_text_counts(entries, since=None):
    """(thinking blocks carrying text, all thinking blocks) over `entries`.

    Whitespace-only text counts as absent. With `since`, only entries stamped
    with that Claude Code version or later are counted; an entry whose
    version is missing or unparseable is left out rather than guessed at.
    """
    with_text = total = 0
    for obj in entries:
        if since is not None:
            version = _version_tuple(obj.get("version"))
            if version is None or version < since:
                continue
        content = obj.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "thinking":
                continue
            total += 1
            if (block.get("thinking") or "").strip():
                with_text += 1
    return with_text, total


def thinking_text_is_a_minority(with_text, total):
    return with_text / total < THINKING_TEXT_MAJORITY


class TestThinkingTextIsAMinority:
    """Some thinking blocks carry their text on disk, and most do not.

    This class was `TestThinkingTextIsAbsent` and asserted that NO block in a
    60-file sample carried text. That was an absolute about a rare event,
    checked on a sample, and it flickered red and green as the corpus grew
    until the event stopped being rare: from Claude Code 2.1.257 a share of
    blocks carries short text again. The measurement and its conditions are
    in CHANGELOG.md; the test prints the current counts (run it with `-s`).

    What replaces it is the claim the docs now make, in the one direction a
    design decision hangs on. Text being present at all means every surface
    must treat thinking as real content. Text staying a minority means
    `has_thinking` still says nothing about whether reasoning can be read,
    and a populator for it would be mostly empty. If the share crosses
    `THINKING_TEXT_MAJORITY`, both readings need revisiting.

    Scanned in full, scoped to `THINKING_TEXT_SINCE`: a sample of a corpus
    that is mostly older transcripts would not see new sessions change.
    """

    def test_blocks_written_now_mostly_carry_no_text(self):
        files = _corpus_files()
        if not files:
            pytest.skip("no local Claude Code corpus to check the contract against")

        with_text = total = files_with_blocks = files_with_text = 0
        for f in files:
            entries = []
            try:
                with open(f, errors="replace") as fh:
                    for line in fh:
                        if '"thinking"' not in line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict) and obj.get("type") == "assistant":
                            entries.append(obj)
            except OSError:
                continue
            file_text, file_total = thinking_text_counts(
                entries, since=THINKING_TEXT_SINCE
            )
            with_text += file_text
            total += file_total
            files_with_blocks += bool(file_total)
            files_with_text += bool(file_text)

        since = ".".join(str(n) for n in THINKING_TEXT_SINCE)
        if not total:
            pytest.skip(f"no thinking blocks written by Claude Code {since} or later")
        print(
            f"\nclaim 5, Claude Code {since} or later: {with_text} of {total} "
            f"thinking blocks carry text, in {files_with_text} of "
            f"{files_with_blocks} transcripts that hold a thinking block"
        )
        assert thinking_text_is_a_minority(with_text, total), (
            f"{with_text} of {total} thinking blocks written by Claude Code "
            f"{since} or later carry text, at or over the "
            f"{THINKING_TEXT_MAJORITY:.0%} line. Most reasoning is on disk "
            "now: docs/JSONL_CONTRACT.md claim 5 and the reader's guide both "
            "say a minority, and a populator for it is no longer mostly empty."
        )

    def test_the_check_rejects_a_violation(self):
        """The oracle can fail."""
        entries = [
            {"uuid": f"u{i}", "version": "2.1.300",
             "message": {"content": [{"type": "thinking", "thinking": text}]}}
            for i, text in enumerate(["actual reasoning", "more of it", ""])
        ]
        with_text, total = thinking_text_counts(entries, since=THINKING_TEXT_SINCE)
        assert (with_text, total) == (2, 3)
        assert not thinking_text_is_a_minority(with_text, total)

    def test_empty_and_whitespace_thinking_are_both_absent(self):
        for value in ("", "   ", "\n"):
            entry = {
                "uuid": "u4",
                "message": {"content": [{"type": "thinking", "thinking": value}]},
            }
            assert thinking_text_counts([entry]) == (0, 1), repr(value)

    def test_older_and_unversioned_entries_are_outside_the_scope(self):
        """Versions compare as numbers: 2.1.72 is older than 2.1.257."""
        block = {"content": [{"type": "thinking", "thinking": "reasoning"}]}
        entries = [
            {"uuid": "old", "version": "2.1.72", "message": block},
            {"uuid": "none", "message": block},
            {"uuid": "odd", "version": "dev", "message": block},
            {"uuid": "new", "version": "2.1.257", "message": block},
        ]
        assert thinking_text_counts(entries, since=THINKING_TEXT_SINCE) == (1, 1)
        assert thinking_text_counts(entries) == (4, 4)


# ---------------------------------------------------------------------------
# Claim 11: toolUseResult is stated beside every tool result in a parent
# transcript, and is often missing in an agent transcript.
# ---------------------------------------------------------------------------

# The first Claude Code version in this corpus after the long stretch in which
# agent transcripts almost never carried the payload (the version seen before
# it is 2.1.153). An identifier, not a measurement: it scopes the agent canary
# to the format being written now, so months of older transcripts that mostly
# lack the payload cannot hide a change in new ones.
TOOL_RESULT_PAYLOAD_SINCE = (2, 1, 195)

# Reasoned, not measured. A parent result without the payload is unseen in the
# corpus; the line is where "every parent result states it" stops being a
# usable approximation and the typed columns on fact_tool_calls go empty for
# ordinary sessions.
PARENT_PAYLOAD_MAX_ABSENT_RATE = 0.001

# Reasoned, not measured: the share at which the typed columns stop covering
# most of the agent work being written now.
AGENT_PAYLOAD_ABSENT_MAJORITY = 0.5


def tool_result_payload_counts(entries, since=None):
    """(tool results whose entry states no toolUseResult, all tool results).

    Counted per `tool_result` block, since that is the grain of
    fact_tool_calls. An explicit null counts as absent, because that is what
    the parser does with it. With `since`, only entries stamped with that
    Claude Code version or later are counted; a missing or unparseable version
    is left out rather than guessed at.
    """
    absent = total = 0
    for obj in entries:
        if since is not None:
            version = _version_tuple(obj.get("version"))
            if version is None or version < since:
                continue
        content = obj.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        results = sum(
            1 for block in content
            if isinstance(block, dict) and block.get("type") == "tool_result"
        )
        total += results
        if obj.get("toolUseResult") is None:
            absent += results
    return absent, total


def parent_payload_absence_is_a_rounding_error(absent, total):
    return absent / total < PARENT_PAYLOAD_MAX_ABSENT_RATE


def agent_results_mostly_state_the_payload(absent, total):
    return absent / total < AGENT_PAYLOAD_ABSENT_MAJORITY


def _is_agent_transcript(path):
    """Claim 1: what a transcript is comes from the file, never from a field."""
    return path.stem.startswith("agent-")


@pytest.fixture(scope="module")
def tool_result_payload_corpus():
    """One pass over the whole corpus, shared by both canaries.

    Scanned in full rather than sampled: a parent result without the payload
    is rarer than any sample could see, and the agent share swings between
    Claude Code versions, so a draw says more about which versions it landed
    on than about the format.
    """
    files = _corpus_files()
    if not files:
        pytest.skip("no local Claude Code corpus to check the contract against")

    counts = {"parent": [0, 0], "agent": [0, 0], "agent_since": [0, 0]}
    by_tool = {}
    for f in files:
        kind = "agent" if _is_agent_transcript(f) else "parent"
        names, results = {}, []
        try:
            with open(f, errors="replace") as fh:
                for line in fh:
                    if '"tool_use"' not in line and '"tool_result"' not in line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    message = obj.get("message")
                    content = message.get("content") if isinstance(message, dict) else None
                    if not isinstance(content, list):
                        continue
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            names[block.get("id")] = block.get("name")
                    if any(isinstance(b, dict) and b.get("type") == "tool_result"
                           for b in content):
                        results.append(obj)
        except OSError:
            continue

        absent, total = tool_result_payload_counts(results)
        counts[kind][0] += absent
        counts[kind][1] += total
        if kind == "agent":
            absent, total = tool_result_payload_counts(
                results, since=TOOL_RESULT_PAYLOAD_SINCE
            )
            counts["agent_since"][0] += absent
            counts["agent_since"][1] += total
            for obj in results:
                stated = obj.get("toolUseResult") is not None
                for block in obj["message"]["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool = names.get(block.get("tool_use_id")) or "?"
                        tool = "mcp__*" if tool.startswith("mcp__") else tool
                        row = by_tool.setdefault(tool, [0, 0])
                        row[0] += not stated
                        row[1] += 1
    return {"counts": counts, "agent_by_tool": by_tool}


class TestToolResultPayloadPresence:
    """A tool result's structured payload: always there for a parent, often not for an agent.

    `fact_tool_calls.result_payload_json` and every column typed from it
    (`read_num_lines`, `glob_num_files`, the Agent rollups) read the
    entry-level `toolUseResult`, and nothing falls back when it is missing.
    Found 2026-10-13 from an audit of a one-project build that showed
    `glob_num_files` NULL on every row: every Glob call in that project sat in
    an agent transcript.

    Two directions a decision hangs on, one canary each. In parent
    transcripts the payload must keep being stated, or the typed columns
    empty for ordinary sessions. In agent transcripts written now it must
    stay the common case, or a query that sums a typed column silently
    leaves out most current agent work. That it is often missing over the
    whole corpus is the measurement, not an assertion: the canaries print
    the counts (run with `-s`).
    """

    def test_a_parent_result_states_the_payload(self, tool_result_payload_corpus):
        absent, total = tool_result_payload_corpus["counts"]["parent"]
        if not total:
            pytest.skip("no tool results in any parent transcript")
        print(
            f"\nclaim 11, parent transcripts: {absent} of {total} tool results "
            "state no toolUseResult"
        )
        assert parent_payload_absence_is_a_rounding_error(absent, total), (
            f"{absent} of {total} tool results in parent transcripts state no "
            f"toolUseResult, at or over {PARENT_PAYLOAD_MAX_ABSENT_RATE:.1%}. "
            "The typed result columns on fact_tool_calls are emptying for "
            "ordinary sessions; see docs/JSONL_CONTRACT.md claim 11."
        )

    def test_agent_results_written_now_mostly_state_the_payload(
        self, tool_result_payload_corpus
    ):
        counts = tool_result_payload_corpus["counts"]
        all_absent, all_total = counts["agent"]
        absent, total = counts["agent_since"]
        since = ".".join(str(n) for n in TOOL_RESULT_PAYLOAD_SINCE)
        if not total:
            pytest.skip(f"no agent tool results written by Claude Code {since} or later")
        by_tool = sorted(
            tool_result_payload_corpus["agent_by_tool"].items(),
            key=lambda item: -item[1][1],
        )[:10]
        print(
            f"\nclaim 11, agent transcripts, every version: {all_absent} of "
            f"{all_total} tool results state no toolUseResult"
            f"\nclaim 11, agent transcripts, Claude Code {since} or later: "
            f"{absent} of {total}"
            "\nclaim 11, agent transcripts by tool, every version: "
            + ", ".join(f"{tool} {a} of {n}" for tool, (a, n) in by_tool)
        )
        assert agent_results_mostly_state_the_payload(absent, total), (
            f"{absent} of {total} tool results in agent transcripts written by "
            f"Claude Code {since} or later state no toolUseResult, at or over "
            f"the {AGENT_PAYLOAD_ABSENT_MAJORITY:.0%} line. The typed result "
            "columns no longer cover most current agent work; see "
            "docs/JSONL_CONTRACT.md claim 11."
        )

    def test_the_parent_check_rejects_a_violation(self):
        """The oracle can fail."""
        stated = {"uuid": "u1", "toolUseResult": {"numFiles": 2},
                  "message": {"content": [{"type": "tool_result", "tool_use_id": "t1"}]}}
        missing = {"uuid": "u2",
                   "message": {"content": [{"type": "tool_result", "tool_use_id": "t2"}]}}
        absent, total = tool_result_payload_counts([stated, missing])
        assert (absent, total) == (1, 2)
        assert not parent_payload_absence_is_a_rounding_error(absent, total)
        assert parent_payload_absence_is_a_rounding_error(
            *tool_result_payload_counts([stated])
        )

    def test_the_agent_check_rejects_a_violation(self):
        """The oracle can fail."""
        def result(uuid, **extra):
            return {"uuid": uuid, "version": "2.1.300", **extra,
                    "message": {"content": [{"type": "tool_result", "tool_use_id": uuid}]}}

        entries = [result("a"), result("b"), result("c", toolUseResult={"stdout": ""})]
        absent, total = tool_result_payload_counts(
            entries, since=TOOL_RESULT_PAYLOAD_SINCE
        )
        assert (absent, total) == (2, 3)
        assert not agent_results_mostly_state_the_payload(absent, total)

    def test_an_error_string_is_a_stated_payload_and_null_is_not(self):
        """A failed call states its payload as a bare string; that is stated."""
        block = {"content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True}]}
        assert tool_result_payload_counts(
            [{"uuid": "e", "toolUseResult": "Error: no such file", "message": block}]
        ) == (0, 1)
        assert tool_result_payload_counts(
            [{"uuid": "n", "toolUseResult": None, "message": block}]
        ) == (1, 1)

    def test_only_tool_result_blocks_are_counted(self):
        """A prompt, or an entry carrying no result, is outside the claim."""
        entries = [
            {"uuid": "p", "message": {"content": "run the tests"}},
            {"uuid": "t", "message": {"content": [{"type": "text", "text": "hi"}]}},
            {"uuid": "a", "message": {"content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]}},
        ]
        assert tool_result_payload_counts(entries) == (0, 0)

    def test_older_and_unversioned_entries_are_outside_the_agent_scope(self):
        """Versions compare as numbers: 2.1.153 is older than 2.1.195."""
        block = {"content": [{"type": "tool_result", "tool_use_id": "t1"}]}
        entries = [
            {"uuid": "old", "version": "2.1.153", "message": block},
            {"uuid": "none", "message": block},
            {"uuid": "odd", "version": "dev", "message": block},
            {"uuid": "new", "version": "2.1.195", "message": block},
        ]
        assert tool_result_payload_counts(
            entries, since=TOOL_RESULT_PAYLOAD_SINCE
        ) == (1, 1)
        assert tool_result_payload_counts(entries) == (4, 4)

    def test_a_result_without_the_payload_keeps_its_row_and_its_text(self, tmp_path):
        """What ccutils does with the gap, through the real ETL.

        Two Read calls in an agent transcript, the shape the corpus holds:
        one result whose entry states toolUseResult and one whose entry does
        not. Both must land in fact_tool_calls with their result text. Only
        the first has a payload and the column typed from it, and the second
        is found by the predicate docs/JSONL_CONTRACT.md claim 11 gives. If
        this fails, either a result is being dropped for lacking a payload or
        a typed column is being filled from somewhere the claim does not say.
        """
        from ccutils import create_star_schema
        from ccutils.etl.orchestrator import run_v15_etl

        agent_dir = tmp_path / "projects" / "-work-demo" / "parent-1" / "subagents"
        agent_dir.mkdir(parents=True)
        src = agent_dir / "agent-abc.jsonl"

        def entry(kind, uuid, second, content, **extra):
            return {"type": kind, "uuid": uuid, "sessionId": "parent-1",
                    "isSidechain": True, "agentId": "abc", "version": "2.1.153",
                    "timestamp": f"2026-05-20T10:00:{second:02d}.000Z", "cwd": "/work",
                    "message": {"role": kind, "content": content,
                                **({"model": "claude-opus-4-7"} if kind == "assistant" else {})},
                    **extra}

        def read_call(tool_id, path):
            return [{"type": "tool_use", "id": tool_id, "name": "Read",
                     "input": {"file_path": path}}]

        def read_result(tool_id, text):
            return [{"type": "tool_result", "tool_use_id": tool_id, "content": text}]

        src.write_text("\n".join(json.dumps(e) for e in [
            entry("user", "u1", 0, "read both files"),
            entry("assistant", "a1", 1, read_call("t-stated", "/work/a.py")),
            entry("user", "r1", 2, read_result("t-stated", "FIRST BODY"),
                  toolUseResult={"type": "text", "file": {
                      "filePath": "/work/a.py", "content": "FIRST BODY",
                      "numLines": 7, "startLine": 1, "totalLines": 7}}),
            entry("assistant", "a2", 3, read_call("t-missing", "/work/b.py")),
            entry("user", "r2", 4, read_result("t-missing", "SECOND BODY")),
        ]))

        conn = create_star_schema(tmp_path / "payload.duckdb")
        run_v15_etl(conn, src, project_name="demo", parquet_lake_root=tmp_path / "lake")
        rows = conn.execute(
            "SELECT tool_use_id, result_entry_id IS NOT NULL, result_content_text, "
            "       result_payload_json IS NOT NULL, read_num_lines "
            "FROM fact_tool_calls WHERE NOT is_deleted ORDER BY tool_use_id"
        ).fetchall()
        unstated = conn.execute(
            "SELECT tool_use_id FROM fact_tool_calls WHERE NOT is_deleted "
            "AND result_entry_id IS NOT NULL AND result_payload_json IS NULL"
        ).fetchall()
        conn.close()

        assert rows == [
            ("t-missing", True, "SECOND BODY", False, None),
            ("t-stated", True, "FIRST BODY", True, 7),
        ]
        assert unstated == [("t-missing",)]
