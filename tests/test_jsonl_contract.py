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


class TestOneContentBlockPerAssistantEntry:
    """Text, thinking and tool calls arrive as separate entries.

    This is the claim an external audit misread as an 80% data-loss bug:
    `content_text` being NULL on a tool-carrying row is the format, not a
    loss. If Claude Code ever starts emitting prose alongside a tool call in
    one entry, that reading becomes true and the message grain breaks.
    """

    def test_corpus_holds(self, corpus_sample):
        entries = list(_entries(corpus_sample, "assistant"))
        assert entries, "sample contained no assistant entries"
        violations = single_block_violations(entries)
        assert not violations, (
            f"{len(violations)} assistant entries carry more than one block "
            f"or mix block kinds, e.g. {violations[:3]}. docs/JSONL_CONTRACT.md "
            "claim 2 no longer holds; fact_messages' grain needs revisiting."
        )

    def test_the_check_rejects_a_violation(self):
        """The oracle can fail."""
        mixed = {
            "uuid": "u1",
            "message": {
                "content": [
                    {"type": "text", "text": "here goes"},
                    {"type": "tool_use", "id": "t1"},
                ]
            },
        }
        assert single_block_violations([mixed])

    def test_the_check_passes_a_conforming_entry(self):
        ok = {"uuid": "u2", "message": {"content": [{"type": "tool_use", "id": "t1"}]}}
        assert not single_block_violations([ok])


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
