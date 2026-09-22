"""Fixtures that model Antigravity's on-disk contract (docs/ANTIGRAVITY_CONTRACT.md).

Three things live here:

- A protobuf wire encoder and decoder. The lake writer decodes nothing (Tier 1
  mirrors bytes), so the only consumers are fixtures, which need to build
  realistic blobs, and contract canaries, which read a few fields of real
  blobs. The pair is checked against golden bytes from the protobuf encoding
  spec, not only against each other: an encoder and decoder written from the
  same misreading agree perfectly.
- The DDL of the real SQLite files, copied verbatim from `sqlite_master`. A
  corpus canary pins these strings to the files on disk, so a fixture built
  from them models what Antigravity writes rather than what a test author
  assumed (the agent-layout fixtures once modelled a layout that had not
  existed for years, and every test passed).
- `make_gemini_home`, which builds a whole `<HOME>/.gemini` tree: stores, a
  WAL-mode database, brain dirs with excluded subtrees, encrypted files, an
  unknown and a backup store, and decoy credentials that must never be opened.
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


def varint(n: int) -> bytes:
    if n < 0:
        n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def f_varint(field_no: int, value: int) -> bytes:
    return varint(field_no << 3) + varint(value)


def f_bytes(field_no: int, value: bytes) -> bytes:
    return varint((field_no << 3) | 2) + varint(len(value)) + value


def f_str(field_no: int, value: str) -> bytes:
    return f_bytes(field_no, value.encode("utf-8"))


def f_msg(field_no: int, *parts: bytes) -> bytes:
    return f_bytes(field_no, b"".join(parts))


def f_timestamp(field_no: int, seconds: int, nanos: int = 0) -> bytes:
    """google.protobuf.Timestamp {1: seconds, 2: nanos}; proto3 omits zeros."""
    parts = [f_varint(1, seconds)] if seconds else []
    if nanos:
        parts.append(f_varint(2, nanos))
    return f_msg(field_no, *parts)


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if i >= len(buf):
            raise ValueError("truncated varint")
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return result, i
        if shift > 63:
            raise ValueError("varint too long")


def parse(buf: bytes) -> list[tuple[int, int, object]]:
    """bytes -> [(field, wire_type, value)]; raises on invalid wire data.

    Length-delimited values stay bytes: without a schema a string, a nested
    message and packed scalars are indistinguishable, and guessing is how a
    decoder starts lying. Callers that know the schema at a path say so."""
    out = []
    i, n = 0, len(buf)
    while i < n:
        key, i = _read_varint(buf, i)
        field_no, wt = key >> 3, key & 7
        if field_no == 0:
            raise ValueError("field number 0")
        if wt == 0:
            value, i = _read_varint(buf, i)
        elif wt == 1:
            if i + 8 > n:
                raise ValueError("truncated fixed64")
            value, i = buf[i:i + 8], i + 8
        elif wt == 2:
            length, i = _read_varint(buf, i)
            if i + length > n:
                raise ValueError("truncated length-delimited field")
            value, i = buf[i:i + length], i + length
        elif wt == 5:
            if i + 4 > n:
                raise ValueError("truncated fixed32")
            value, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        out.append((field_no, wt, value))
    return out


def get(buf: bytes, *path: int):
    """The value at a field path, protobuf last-wins for repeated scalars.

    Every hop but the last must be a length-delimited sub-message. Returns None
    when any hop is absent (proto3 cannot tell absent from zero; the caller
    decides what absence means at that path)."""
    current: object = buf
    for field_no in path:
        if not isinstance(current, (bytes, bytearray)):
            return None
        found = None
        for f, _wt, value in parse(bytes(current)):
            if f == field_no:
                found = value
        if found is None:
            return None
        current = found
    return current


def timestamp_seconds(buf: bytes | None) -> float | None:
    if buf is None:
        return None
    fields: dict[int, int] = {f: v for f, wt, v in parse(buf) if wt == 0}  # type: ignore[misc]
    return fields.get(1, 0) + fields.get(2, 0) / 1e9


# ---------------------------------------------------------------------------
# The real DDL, verbatim from sqlite_master (ordered by type, name). A corpus
# canary in test_antigravity_contract.py compares these to every real file.
# ---------------------------------------------------------------------------

REAL_CONVERSATION_DDL: tuple[str, ...] = (
    "CREATE INDEX `idx_steps_status` ON `steps`(`status`)",
    "CREATE INDEX `idx_steps_step_type` ON `steps`(`step_type`)",
    "CREATE TABLE `battle_mode_infos` (`idx` integer,`data` blob,PRIMARY KEY (`idx`))",
    "CREATE TABLE `executor_metadata` (`idx` integer,`data` blob,PRIMARY KEY (`idx`))",
    "CREATE TABLE `gen_metadata` (`idx` integer,`data` blob,`size` integer NOT NULL DEFAULT 0,PRIMARY KEY (`idx`))",
    "CREATE TABLE `parent_references` (`idx` integer,`data` blob,PRIMARY KEY (`idx`))",
    "CREATE TABLE `steps` (`idx` integer,`step_type` integer NOT NULL DEFAULT 0,`status` integer NOT NULL DEFAULT 0,`has_subtrajectory` numeric NOT NULL DEFAULT false,`metadata` blob,`error_details` blob,`permissions` blob,`task_details` blob,`render_info` blob,`step_payload` blob,`step_format` integer NOT NULL DEFAULT 0,PRIMARY KEY (`idx`))",
    "CREATE TABLE `trajectory_meta` (`trajectory_id` text,`cascade_id` text,`trajectory_type` integer,`source` integer,PRIMARY KEY (`trajectory_id`))",
    "CREATE TABLE `trajectory_metadata_blob` (`id` text DEFAULT \"main\",`data` blob,PRIMARY KEY (`id`))",
)

REAL_SUMMARIES_DDL: tuple[str, ...] = (
    "CREATE INDEX `idx_conversation_summaries_last_modified_time` ON `conversation_summaries`(`last_modified_time`)",
    "CREATE INDEX `idx_conversation_summaries_last_user_input_time` ON `conversation_summaries`(`last_user_input_time`)",
    "CREATE TABLE `conversation_summaries` (`conversation_id` text,`title` text NOT NULL DEFAULT \"\",`preview` text NOT NULL DEFAULT \"\",`step_count` integer NOT NULL DEFAULT 0,`last_modified_time` datetime NOT NULL,`workspace_uris` text NOT NULL,`status` text NOT NULL DEFAULT \"\",`source` text NOT NULL DEFAULT \"\",`project_id` text NOT NULL DEFAULT \"\",`agent_name` text NOT NULL DEFAULT \"\",`parent_conversation_id` text NOT NULL DEFAULT \"\",`nesting_depth` integer NOT NULL DEFAULT 0,`battle_id` text NOT NULL DEFAULT \"\",`winning_conversation_id` text NOT NULL DEFAULT \"\",`not_fully_idle` numeric NOT NULL DEFAULT false,`killed` numeric NOT NULL DEFAULT false,`last_user_input_time` datetime NOT NULL,`last_user_input_step_index` integer NOT NULL DEFAULT -1,`app_data_dir` text NOT NULL DEFAULT \"\",`raw_summary` blob, group_id TEXT NOT NULL DEFAULT '',PRIMARY KEY (`conversation_id`))",
)


def _apply_ddl(conn: sqlite3.Connection, ddl: tuple[str, ...]) -> None:
    # Tables before indexes; the tuples are ordered by (type, name) to match
    # the canary's query, which puts indexes first.
    for stmt in sorted(ddl, key=lambda s: not s.startswith("CREATE TABLE")):
        conn.execute(stmt)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

BASE_TS = 1_790_000_000  # 2026-09 in epoch seconds


def step_metadata(created_s: int, execution_id: str = "exec-1", tool: tuple[str, str] | None = None) -> bytes:
    """A CortexStepMetadata: 1 created_at, 12 execution_id, 4 tool_call{1 id, 2 name}."""
    parts = [f_timestamp(1, created_s), f_str(12, execution_id)]
    if tool:
        parts.append(f_msg(4, f_str(1, tool[0]), f_str(2, tool[1])))
    return b"".join(parts)


def step_payload(step_type: int, status: int, metadata: bytes, body: bytes = b"") -> bytes:
    """A gemini_coder.Step: 1 type, 4 status, 5 metadata, plus a oneof body.

    The `metadata` column of the real table is byte-identical to field 5 of
    the payload (contract claim); fixtures keep that true."""
    return f_varint(1, step_type) + f_varint(4, status) + f_bytes(5, metadata) + body


@dataclass
class Step:
    idx: int
    step_type: int = 15            # PLANNER_RESPONSE
    status: int = 3                # DONE
    text: str = "hello"
    tool: tuple[str, str] | None = None
    blob_size: int = 0             # extra payload bytes, to exercise large rows

    def row(self, created_s: int):
        meta = step_metadata(created_s, tool=self.tool)
        body = f_msg(20, f_str(1, self.text))
        if self.blob_size:
            body += f_bytes(99, os.urandom(self.blob_size))
        return (
            self.idx, self.step_type, self.status, 0,
            meta, None, None, None, None,
            step_payload(self.step_type, self.status, meta, body), 0,
        )


_STEP_COLUMNS = (
    "idx", "step_type", "status", "has_subtrajectory", "metadata",
    "error_details", "permissions", "task_details", "render_info",
    "step_payload", "step_format",
)


def write_conversation_db(
    path: Path,
    cascade_id: str,
    steps: list[Step],
    *,
    trajectory_id: str | None = None,
    generations: int = 1,
    extra_steps_column: str | None = None,
    drop_steps_column: str | None = None,
    extra_table: str | None = None,
    keep_open_wal: bool = False,
) -> sqlite3.Connection | None:
    """Write one conversation file with the real schema.

    keep_open_wal=True then switches the file to WAL mode, disables
    autocheckpoint, and returns an open connection: rows appended through it
    exist only in the -wal. Closing it would checkpoint them into the main
    file and the fixture would silently stop testing WAL. The caller closes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    ddl = REAL_CONVERSATION_DDL
    if drop_steps_column:
        ddl = tuple(
            s.replace(f"`{drop_steps_column}` blob,", "") if s.startswith("CREATE TABLE `steps`") else s
            for s in ddl
        )
    _apply_ddl(conn, ddl)
    if extra_steps_column:
        conn.execute(f"ALTER TABLE steps ADD COLUMN `{extra_steps_column}` text")
    if extra_table:
        conn.execute(f"CREATE TABLE `{extra_table}` (`idx` integer, `data` blob)")
        conn.execute(f"INSERT INTO `{extra_table}` VALUES (0, x'0102')")
    traj = trajectory_id or f"traj-{cascade_id}"
    conn.execute("INSERT INTO trajectory_meta VALUES (?, ?, 4, 1)", (traj, cascade_id))
    conn.execute(
        "INSERT INTO trajectory_metadata_blob VALUES ('main', ?)",
        (f_timestamp(2, BASE_TS) + f_str(18, "project-1"),),
    )
    cols = [r[1] for r in conn.execute("PRAGMA table_info(steps)")]
    for s in steps:
        row: dict[str, object] = dict(zip(_STEP_COLUMNS, s.row(BASE_TS + s.idx)))
        if extra_steps_column:
            row[extra_steps_column] = f"extra-{s.idx}"
        present = [c for c in cols if c in row]
        conn.execute(
            f"INSERT INTO steps ({', '.join(present)}) VALUES ({', '.join('?' for _ in present)})",
            [row[c] for c in present],
        )
    for g in range(generations if steps else 0):
        data = f_msg(1, f_str(19, "gemini-test-model")) + f_str(4, "exec-1")
        conn.execute("INSERT INTO gen_metadata VALUES (?, ?, ?)", (g, data, len(data)))
    if steps:
        conn.execute("INSERT INTO executor_metadata VALUES (0, ?)", (f_str(9, "exec-1"),))
    conn.commit()
    conn.close()
    if keep_open_wal:
        # The rows above are in the main file; everything appended through the
        # returned connection lives only in the -wal until a checkpoint.
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        return conn
    return None


def append_steps(conn: sqlite3.Connection, steps: list[Step]) -> None:
    for s in steps:
        conn.execute(
            f"INSERT INTO steps ({', '.join(_STEP_COLUMNS)}) VALUES ({', '.join('?' for _ in _STEP_COLUMNS)})",
            s.row(BASE_TS + s.idx),
        )
    conn.commit()


def write_summaries_db(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    _apply_ddl(conn, REAL_SUMMARIES_DDL)
    for r in rows:
        full = {
            "conversation_id": r["conversation_id"],
            "title": r.get("title", ""),
            "preview": r.get("preview", "preview text"),
            "step_count": r.get("step_count", 0),
            "last_modified_time": r.get("last_modified_time", "2026-09-19 21:31:07.000000+00:00"),
            "workspace_uris": r.get("workspace_uris", '["file:///workspace/project"]'),
            "status": "CASCADE_RUN_STATUS_IDLE",
            "source": r.get("source", ""),
            "project_id": r.get("project_id", "project-1"),
            "agent_name": r.get("agent_name", ""),
            "parent_conversation_id": r.get("parent_conversation_id", ""),
            "nesting_depth": r.get("nesting_depth", 0),
            "battle_id": "",
            "winning_conversation_id": "",
            "not_fully_idle": 0,
            "killed": 0,
            "last_user_input_time": "2026-09-19 21:30:00.000000+00:00",
            "last_user_input_step_index": -1,
            "app_data_dir": r.get("app_data_dir", "antigravity"),
            "raw_summary": r.get("raw_summary", f_varint(2, r.get("step_count", 0))),
            "group_id": "",
        }
        conn.execute(
            f"INSERT INTO conversation_summaries ({', '.join(full)}) VALUES ({', '.join('?' for _ in full)})",
            list(full.values()),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# A whole <HOME>/.gemini tree
# ---------------------------------------------------------------------------

CONV_A = "11111111-1111-4111-8111-111111111111"   # main store, .db + brain
CONV_B = "22222222-2222-4222-8222-222222222222"   # main store, .db, subagent of A
CONV_EMPTY = "33333333-3333-4333-8333-333333333333"  # main store, .db with 0 steps
CONV_PB = "44444444-4444-4444-8444-444444444444"  # main store, legacy encrypted .pb + brain
CONV_CLI = "55555555-5555-4555-8555-555555555555"  # cli store, .db
IMPLICIT = "66666666-6666-4666-8666-666666666666"

DECOY_RELPATHS = (
    "oauth_creds.json",
    "google_accounts.json",
    "jetski-standalone-oauth-token",
    "antigravity-browser-profile/Cookies",
    "config/mcp_config.json",
    "antigravity/antigravity_state.pbtxt",
    "antigravity/brain/" + CONV_A + "/.git/HEAD",
    "antigravity/brain/" + CONV_A + "/.git/objects/ab/cdef",
    "antigravity/brain/" + CONV_A + "/.system_generated/logs/chunks/transcript_full/00000000.jsonl",
    "antigravity-backup/conversations/" + CONV_PB + ".pb",
    "weird-store/conversations/" + CONV_A + ".db",
)


@dataclass
class GeminiHome:
    root: Path
    decoys: list[Path] = field(default_factory=list)
    wal_conn: sqlite3.Connection | None = None

    def store(self, name: str = "antigravity") -> Path:
        return self.root / name

    def close(self) -> None:
        if self.wal_conn is not None:
            self.wal_conn.close()
            self.wal_conn = None


def _write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    path.write_bytes(data)
    return path


def transcript_line(step_index: int, step_type: str, created_s: int) -> str:
    from datetime import datetime, timezone
    ts = datetime.fromtimestamp(created_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return json.dumps({
        "step_index": step_index, "type": step_type, "source": "MODEL",
        "status": "DONE", "created_at": ts, "content": "text",
    })


def make_gemini_home(root: Path, *, wal: bool = False, lock_decoys: bool = True) -> GeminiHome:
    """Build a realistic <HOME>/.gemini under `root`.

    wal=True leaves CONV_CLI in WAL mode with committed-but-uncheckpointed
    rows held by an open connection (GeminiHome.wal_conn); call close()."""
    home = GeminiHome(root=root)
    main = root / "antigravity"
    cli = root / "antigravity-cli"

    write_conversation_db(main / "conversations" / f"{CONV_A}.db", CONV_A,
                          [Step(0, 14, text="user asks"), Step(1, 15, tool=("call_1", "view_file")),
                           Step(2, 8, tool=("call_1", "view_file"))], generations=2)
    write_conversation_db(main / "conversations" / f"{CONV_B}.db", CONV_B, [Step(0, 14), Step(1, 15)])
    write_conversation_db(main / "conversations" / f"{CONV_EMPTY}.db", CONV_EMPTY, [])
    _write(main / "conversations" / f"{CONV_PB}.pb", random.Random(1).randbytes(4096))
    _write(main / "implicit" / f"{IMPLICIT}.pb", random.Random(2).randbytes(2048))
    write_summaries_db(main / "conversation_summaries.db", [
        {"conversation_id": CONV_A, "step_count": 3},
        {"conversation_id": CONV_B, "step_count": 2, "parent_conversation_id": CONV_A,
         "nesting_depth": 1, "agent_name": "explorer"},
        {"conversation_id": CONV_EMPTY},
        {"conversation_id": CONV_PB, "step_count": 7},
        {"conversation_id": CONV_CLI, "source": "CORTEX_TRAJECTORY_SOURCE_CLI",
         "app_data_dir": "antigravity"},
    ])
    _write(main / "annotations" / f"{CONV_A}.pbtxt", 'title:"A title" last_user_view_time:{seconds:1790000000}\n')
    _write(main / "agyhub_summaries_proto.pb", f_msg(1, f_str(1, CONV_A)))

    brain = main / "brain" / CONV_A
    logs = brain / ".system_generated" / "logs"
    _write(logs / "transcript_full.jsonl", "\n".join(
        transcript_line(i, t, BASE_TS + i) for i, t in enumerate(["USER_INPUT", "PLANNER_RESPONSE", "VIEW_FILE"])) + "\n")
    _write(logs / "transcript.jsonl", transcript_line(0, "USER_INPUT", BASE_TS) + "\n")
    _write(brain / ".system_generated" / "messages" / "77777777-7777-4777-8777-777777777777.json",
           json.dumps({"id": "m1", "sender": "user", "content": "hi"}))
    _write(brain / ".system_generated" / "steps" / "2" / "output.txt", "file contents\n")
    _write(brain / ".system_generated" / "tasks" / "task-1.log", "task log\n")
    _write(brain / "task.md", "# Task\n")
    _write(brain / "task.md.metadata.json", json.dumps({"artifactType": "ARTIFACT_TYPE_TASK", "version": 1}))
    _write(brain / "task.md.resolved.0", "# Task v0\n")
    _write(brain / ".agents" / "agents" / "explorer" / "agent.md", "agent definition\n")
    _write(brain / ".user_uploaded" / "photo.png", b"\x89PNG user upload")
    _write(brain / "shot.png", b"\x89PNG screenshot")
    _write(brain / "notes.txt", "an unclassified small file\n")
    _write(brain / ".system_generated" / "steps" / "5" / "content.md", "a step's full output\n")
    _write(brain / ".system_generated" / "messages" / "undelivered" / "88888888-8888-4888-8888-888888888888",
           json.dumps({"id": "m2"}))
    _write(brain / "artifacts" / "plan.md", "# Older-layout artifact\n")
    _write(brain / ".tempmediaStorage" / "media_1789395530474.img", b"\x89PNG temp media")
    _write(brain / "big.bin", b"\0" * (1024 * 1024 + 1))
    (brain / "leak").symlink_to(root / "oauth_creds.json")
    _write(main / "brain" / "tempmediaStorage" / "x.png", b"\x89PNG temp")
    _write(main / "browser_recordings" / CONV_A / "metadata.json", json.dumps({"highlights": []}))
    _write(main / "browser_recordings" / CONV_A / "1790000000000000000.jpg", b"\xff\xd8 frame")
    _write(main / "brain" / CONV_PB / "implementation_plan.md", "# Plan\n")
    _write(main / "scratch" / "project" / "package.json", "{}")

    conn = write_conversation_db(cli / "conversations" / f"{CONV_CLI}.db", CONV_CLI,
                                 [Step(0, 14), Step(1, 15)], keep_open_wal=wal)
    if wal:
        assert conn is not None
        append_steps(conn, [Step(2, 21, tool=("call_9", "run_command"))])
        home.wal_conn = conn
    write_summaries_db(cli / "conversation_summaries.db", [
        {"conversation_id": CONV_CLI, "app_data_dir": "antigravity-cli"},
    ])
    _write(cli / "history.jsonl", json.dumps({
        "display": "a prompt", "timestamp": 1790000000000, "workspace": "/workspace/project",
        "conversationId": CONV_CLI}) + "\n")
    _write(root / "config" / "projects" / "project-1.json", json.dumps({"id": "project-1", "name": "Project One"}))

    # Decoys: credentials beside the data, plus paths the allowlist excludes.
    for rel in DECOY_RELPATHS:
        p = root / rel
        if not p.exists():
            _write(p, b"DECOY must never be read")
        home.decoys.append(p)
    if lock_decoys:
        for p in home.decoys:
            p.chmod(0)
    return home


def unlock(home: GeminiHome) -> None:
    """Restore permissions so pytest can clean the tree up."""
    for p in home.decoys:
        if p.exists():
            p.chmod(0o644)
