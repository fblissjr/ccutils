"""Shared test helpers importable by name (unique basename so the
import can't collide with any other conftest/module on sys.path,
unlike a `from conftest import ...`)."""

import json


def make_minimal_session_lines(
    session_id,
    *,
    ts_base="2026-04-19T10:00",
    model="claude-opus-4-7",
    entry_session_id=None,
):
    """One minimal valid session: a user line + an assistant line.

    Single source of truth for the entry SHAPE synthetic tests need --
    when the parser grows a new required field, update here, not in
    per-file copies. entry_session_id overrides the sessionId embedded in
    the entries (the REAL subagent contract: agent-file entries carry the
    PARENT's sessionId while the file is named agent-<id>).
    """
    sid = entry_session_id if entry_session_id is not None else session_id
    return [
        {"type": "user", "uuid": f"{session_id}-u1", "sessionId": sid,
         "timestamp": f"{ts_base}:00Z", "cwd": "/p",
         "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "uuid": f"{session_id}-a1",
         "parentUuid": f"{session_id}-u1",
         "sessionId": sid, "timestamp": f"{ts_base}:05Z",
         "message": {"role": "assistant", "model": model,
                     "content": [{"type": "text", "text": "ok"}]}},
    ]


def write_minimal_session(path, session_id, **kwargs):
    """Write make_minimal_session_lines to path; returns path."""
    path.write_text(
        "\n".join(json.dumps(d) for d in make_minimal_session_lines(session_id, **kwargs))
    )
    return path
