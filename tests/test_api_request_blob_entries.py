"""`api-request-blob` entries are copies, and no fact counts them.

From Claude Code 2.1.316 a transcript also holds `api-request-blob` entries
(claim 13 of `docs/JSONL_CONTRACT.md`): content-addressed copies of the
messages of an API request, keyed by `hash`, with no `uuid`, `timestamp` or
`version`, next to `api-request-shape` entries. Their `message.content`
repeats what the conversation's own entries hold: assistant turns with
their thinking and tool_use blocks, user turns with tool_result blocks.

Anything that reads `message.content` without asking what kind of entry
holds it counts each of those twice. That is how the claim-11 canary went
red on 2026-11-13: 238 copied tool results with no `toolUseResult`, and
nothing wrong upstream.

The oracle is the same session without the copies: adding them must leave
every fact table exactly as it was. Shapes are the measured ones (keys
`hash`, `message`, `sessionId`, `type`), not guessed.

What breaks without it: a populator that forgets its type predicate double
counts tool calls, messages or thinking in every session written by a
current Claude Code, and nothing else fails.
"""

import json

from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl

SID = "blob-session"


def _conversation():
    return [
        {"type": "user", "uuid": "u1", "sessionId": SID, "version": "2.1.323",
         "timestamp": "2026-11-13T10:00:00Z", "cwd": "/p",
         "message": {"role": "user", "content": "list the files"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1", "sessionId": SID,
         "version": "2.1.323", "timestamp": "2026-11-13T10:00:05Z", "cwd": "/p",
         "message": {"id": "msg_1", "role": "assistant", "model": "claude-opus-4-7",
                     "usage": {"input_tokens": 10, "output_tokens": 5},
                     "content": [
                         {"type": "thinking", "thinking": "plan", "signature": "s"},
                         {"type": "text", "text": "listing"},
                         {"type": "tool_use", "id": "toolu_b1", "name": "Bash",
                          "input": {"command": "ls"}}]}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1", "sessionId": SID,
         "version": "2.1.323", "timestamp": "2026-11-13T10:00:06Z", "cwd": "/p",
         "toolUseResult": {"stdout": "a b", "stderr": "", "interrupted": False},
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "toolu_b1", "content": "a b"}]}},
    ]


def _copies():
    """What a current Claude Code writes beside the conversation."""
    convo = _conversation()
    return [
        {"type": "api-request-blob", "hash": "h-user", "sessionId": SID,
         "message": convo[0]["message"]},
        {"type": "api-request-blob", "hash": "h-assistant", "sessionId": SID,
         "message": convo[1]["message"]},
        {"type": "api-request-blob", "hash": "h-result", "sessionId": SID,
         "message": convo[2]["message"]},
        {"type": "api-request-shape", "sessionId": SID,
         "hashes": ["h-user", "h-assistant", "h-result"]},
    ]


def _facts(tmp_path, name, lines):
    # The same source path for both builds: keys derive from it, so only
    # the file's contents may differ between the two.
    proj = tmp_path / "projects" / "-p"
    proj.mkdir(parents=True, exist_ok=True)
    f = proj / f"{SID}.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in lines))
    conn = create_star_schema(tmp_path / f"{name}.duckdb")
    run_v15_etl(conn, f, parquet_lake_root=tmp_path / f"lake-{name}")
    tables = [r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' "
        "AND table_name LIKE 'fact_%'").fetchall()]
    # Compare what each fact says, not the lineage or the source position:
    # entry_id hashes the line number, which the copies shift.
    # extracted_at is the wall clock.
    skip = {"entry_id", "etl_run_id", "created_at", "last_updated_at", "extracted_at",
            "hash_diff", "result_entry_id", "sequence_num", "date_key", "time_key"}
    out = {}
    for t in tables:
        names = [r[0] for r in conn.execute(f"DESCRIBE {t}").fetchall()]
        cols = []
        for c in names:
            if c in skip:
                continue
            # An entry with no message.id is its own response, keyed on its
            # entry_id (fact_messages R23); compare that as "own id", since
            # entry_id moves with the line number.
            if c == "api_message_id" and "entry_id" in names:
                c = "CASE WHEN api_message_id = entry_id THEN '<own entry>' ELSE api_message_id END"
            cols.append(c)
        out[t] = sorted(map(repr, conn.execute(
            f"SELECT {', '.join(cols)} FROM {t}").fetchall()))
    conn.close()
    return out


def test_the_copies_change_no_fact(tmp_path):
    plain = _facts(tmp_path, "plain", _conversation())
    convo = _conversation()
    # Interleaved the way they appear on disk: copies right after the turn.
    with_copies = _facts(tmp_path, "with", [convo[0], *_copies()[:1], convo[1],
                                            *_copies()[1:2], convo[2], *_copies()[2:]])
    assert plain["fact_messages"] and plain["fact_tool_calls"], "the oracle loaded nothing"
    for table in plain:
        assert with_copies[table] == plain[table], table
