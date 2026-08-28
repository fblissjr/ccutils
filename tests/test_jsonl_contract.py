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
# Claim 5: thinking text is not persisted.
# ---------------------------------------------------------------------------


def thinking_blocks_with_text(entries):
    """Thinking blocks whose text survived to disk."""
    found = []
    for obj in entries:
        content = obj.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "thinking":
                continue
            if (block.get("thinking") or "").strip():
                found.append(obj.get("uuid"))
    return found


class TestThinkingTextIsAbsent:
    """Reasoning is not recoverable from these transcripts.

    Corpus-wide at time of writing: 50,214 thinking blocks carrying an empty
    string against 54 with content. `has_thinking` counts something the
    source never wrote, which is why no populator stores it.

    This canary is the one most likely to go red for a good reason: if a
    future Claude Code starts persisting reasoning, nothing else in the
    pipeline would notice, and the coverage layer would keep telling readers
    the data does not exist.
    """

    def test_corpus_holds(self, corpus_sample):
        entries = list(_entries(corpus_sample, "assistant"))
        assert entries, "sample contained no assistant entries"
        with_text = thinking_blocks_with_text(entries)
        assert not with_text, (
            f"{len(with_text)} thinking blocks now carry text, e.g. "
            f"{with_text[:3]}. Reasoning is being persisted after all -- "
            "docs/JSONL_CONTRACT.md claim 5 and the coverage layer's entry "
            "for it are both wrong, and storing it becomes possible."
        )

    def test_the_check_rejects_a_violation(self):
        """The oracle can fail."""
        entry = {
            "uuid": "u3",
            "message": {
                "content": [{"type": "thinking", "thinking": "actual reasoning"}]
            },
        }
        assert thinking_blocks_with_text([entry])

    def test_empty_and_whitespace_thinking_are_both_absent(self):
        for value in ("", "   ", "\n"):
            entry = {
                "uuid": "u4",
                "message": {"content": [{"type": "thinking", "thinking": value}]},
            }
            assert not thinking_blocks_with_text([entry]), repr(value)
