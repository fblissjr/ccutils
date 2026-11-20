"""What the shaped layer reads out of the Antigravity lake.

The lake is a byte-faithful archive and decodes nothing. Everything here is
derived from it and can be rebuilt from it: it lives in its own tree, never
inside the archive's unit directories (docs/ROADMAP.md, phase 1).

Fields are read by number. Each reading is a claim in
docs/ANTIGRAVITY_CONTRACT.md (17 and 18), measured against the brain
transcript, which states the same values in plain JSON, and held by a canary
in tests/test_antigravity_contract.py. A field this module does not name is
not read: nothing is guessed from the shape of a value.

Absent is not a value. proto3 writes no zero and the CLI empties old steps
in place (claim 16), so a field that is not there reads as None, never as 0
or "". What a cleared step lost is read from the transcript by whoever joins
the two, and they say that they did.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ccutils.parsers.antigravity import decode
from ccutils.parsers.lake import OnUnit, UnitEvent, read_manifest

# Step field 1. The names are the ones the brain transcript states for each
# number (claim 6, "Observed step types"); a test holds this map against that
# table both ways. Six types remain in what the app writes today (claim 17),
# and those are the only ones whose bodies are read below. The per-tool types
# are what tool calls were before they moved into GENERIC (claim 11): they
# are named here, and their bodies are not decoded yet.
USER_INPUT, PLANNER_RESPONSE, ERROR_MESSAGE, CHECKPOINT, SYSTEM_MESSAGE, GENERIC = 14, 15, 17, 23, 101, 132
STEP_TYPE_NAMES = {
    USER_INPUT: "USER_INPUT", PLANNER_RESPONSE: "PLANNER_RESPONSE", ERROR_MESSAGE: "ERROR_MESSAGE",
    CHECKPOINT: "CHECKPOINT", SYSTEM_MESSAGE: "SYSTEM_MESSAGE", GENERIC: "GENERIC",
    5: "CODE_ACTION", 7: "GREP_SEARCH", 8: "VIEW_FILE", 9: "LIST_DIRECTORY", 21: "RUN_COMMAND",
    25: "FIND", 28: "COMMAND_STATUS", 31: "READ_URL_CONTENT", 33: "SEARCH_WEB", 38: "MCP_TOOL",
    85: "BROWSER_SUBAGENT", 90: "EPHEMERAL_MESSAGE", 91: "GENERATE_IMAGE", 98: "CONVERSATION_HISTORY",
    127: "INVOKE_SUBAGENT", 138: "ASK_QUESTION",
}
# Step field 4, named from the transcript's `status` on time-matched lines.
STATUS_NAMES = {2: "RUNNING", 3: "DONE", 4: "INVALID", 5: "CLEARED", 6: "CANCELED", 7: "ERROR"}
STATUS_CLEARED = 5
STATUS_ERROR = 7
# Metadata field 3, named from the transcript's `source`.
SOURCE_NAMES = {2: "MODEL", 4: "USER_EXPLICIT", 5: "SYSTEM", 6: "SYSTEM_SDK"}

# Where each step type keeps its text: body field, then the path inside it.
_TEXT_PATH = {
    USER_INPUT: (19, 2),
    PLANNER_RESPONSE: (20, 1),
    SYSTEM_MESSAGE: (114, 1),
    CHECKPOINT: (30, 5),
    ERROR_MESSAGE: (24, 3, 2),
}


def _int(value: decode.Value | None) -> int | None:
    return value if isinstance(value, int) else None


def read_step(payload: bytes) -> dict[str, object]:
    """The fields of one `step_payload` (a gemini_coder.Step) the shaped layer reads.

    Raises ValueError when the payload is not valid wire format: a decode
    failure is recorded by the caller, never turned into a row of NULLs."""
    top: dict[int, decode.Value] = {}
    for number, _, value in decode.fields(payload):
        top[number] = value
    step_type = _int(top.get(1))
    meta = top.get(5)
    usage = decode.get(meta, 9)
    call = decode.get(meta, 4)
    lineage = decode.get(meta, 20)

    text = thinking = tool_output = None
    path = _TEXT_PATH.get(step_type) if step_type is not None else None
    if path:
        text = decode.text(decode.get(top.get(path[0]), *path[1:]))
    if step_type == USER_INPUT and text is None:
        # A cleared user input loses field 2 and keeps its repeated parts.
        parts = [decode.text(decode.get(part, 1)) for part in decode.get_all(top.get(19), 3)]
        if parts and all(p is not None for p in parts):
            text = "".join(p for p in parts if p is not None)
    if step_type == PLANNER_RESPONSE:
        thinking = decode.text(decode.get(top.get(20), 3))
    if step_type == GENERIC:
        tool_output = decode.text(decode.get(top.get(140), 2, 1))

    return {
        "step_type": step_type,
        "status": _int(top.get(4)),
        "created_at": decode.timestamp(decode.get(meta, 1)),
        "source": _int(decode.get(meta, 3)),
        "execution_id": decode.text(decode.get(meta, 12)),
        "generation_idx": _int(decode.get(lineage, 3)),
        "owner_conversation_id": decode.text(decode.get(lineage, 4)),
        "model_enum": _int(decode.get(meta, 11)),
        "input_tokens": _int(decode.get(usage, 2)),
        "output_tokens": _int(decode.get(usage, 3)),
        "cache_read_tokens": _int(decode.get(usage, 5)),
        "thinking_tokens": _int(decode.get(usage, 9)),
        "tool_call_id": decode.text(decode.get(call, 1)),
        "tool_name": decode.text(decode.get(call, 2)),
        "tool_args": decode.text(decode.get(call, 3)),
        "text": text,
        "thinking": thinking,
        "tool_output": tool_output,
    }


def created_second(value: datetime | None) -> int | None:
    """A step's creation time to the second, the grain the transcript states it at."""
    return int(value.timestamp()) if value is not None else None


# ---------------------------------------------------------------------------
# Shaping a lake into the derived tree
# ---------------------------------------------------------------------------

# Bump when a table's columns or a reading changes: every unit is re-derived.
# A unit whose lake unit is unchanged is skipped, so without the bump a new
# reading never reaches a tree that already exists. `format_fingerprint` is
# pinned to this number by a test.
SHAPED_FORMAT_VERSION = 2
HARNESS_DIR = "antigravity"
SHAPED_DIR = "antigravity-decoded"       # beside the lake's own tree, never inside it
SHAPED_MANIFEST = "_shaped.parquet"

_TS = pa.timestamp("us", tz="UTC")
STEP_SCHEMA = pa.schema([
    ("store", pa.string()), ("conversation_id", pa.string()), ("idx", pa.int64()),
    # False on a row the lake kept after the source dropped or emptied it.
    ("source_present", pa.bool_()), ("lake_ingested_at", _TS),
    ("step_type", pa.int64()), ("step_type_name", pa.string()),
    ("status", pa.int64()), ("status_name", pa.string()), ("is_cleared", pa.bool_()),
    ("created_at", _TS), ("created_second", pa.int64()),
    ("source", pa.int64()), ("source_name", pa.string()),
    ("execution_id", pa.string()), ("generation_idx", pa.int64()),
    ("owner_conversation_id", pa.string()), ("is_inherited", pa.bool_()),
    ("model_enum", pa.int64()),
    ("input_tokens", pa.int64()), ("output_tokens", pa.int64()),
    ("cache_read_tokens", pa.int64()), ("thinking_tokens", pa.int64()),
    ("tool_call_id", pa.string()), ("tool_name", pa.string()), ("tool_args", pa.large_string()),
    ("text", pa.large_string()), ("thinking", pa.large_string()), ("tool_output", pa.large_string()),
    ("payload_bytes", pa.int64()), ("decode_error", pa.string()),
])
LINE_SCHEMA = pa.schema([
    ("store", pa.string()), ("conversation_id", pa.string()), ("line_no", pa.int64()),
    ("step_index", pa.int64()), ("type", pa.string()), ("source", pa.string()), ("status", pa.string()),
    ("created_at", _TS), ("created_second", pa.int64()),
    ("content", pa.large_string()), ("thinking", pa.large_string()), ("tool_calls", pa.large_string()),
    ("input_tokens", pa.int64()), ("output_tokens", pa.int64()), ("cache_read_tokens", pa.int64()),
    ("model", pa.string()), ("media", pa.large_string()), ("error", pa.large_string()),
    # The transcript elides long values in place and says how much it cut.
    ("is_truncated", pa.bool_()),
    # The line written for the step the lake's current row holds at this
    # index: same creation second, last such line (claim 6). Not the other
    # lines a reverted branch left under the index.
    ("is_current", pa.bool_()),
])
_MANIFEST = pa.schema([
    ("store", pa.string()), ("conversation_id", pa.string()), ("out_relpath", pa.string()),
    ("status", pa.string()), ("error", pa.string()),
    ("derived_from", pa.string()), ("shaped_format_version", pa.int32()), ("include_thinking", pa.bool_()),
    ("steps", pa.int64()), ("undecoded_steps", pa.int64()),
    ("transcript_lines", pa.int64()), ("unparsed_lines", pa.int64()),
])
_TRUNCATED = re.compile(r"<truncated \d+ bytes>")
_ROWS_PER_GROUP = 2000


def format_fingerprint() -> str:
    """A hash of the name maps, the text paths and the two table schemas.

    It covers what is declared as data. A reading written as code in
    `read_step` (the tool call's path, the usage fields) is not in it, and a
    change there needs the version bumped by hand."""
    parts = [sorted(STEP_TYPE_NAMES.items()), sorted(STATUS_NAMES.items()), sorted(SOURCE_NAMES.items()),
             sorted(_TEXT_PATH.items()), str(STEP_SCHEMA), str(LINE_SCHEMA)]
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


@dataclass
class ShapeResult:
    tree: Path
    written: int = 0
    unchanged: int = 0
    steps: int = 0
    undecoded_steps: int = 0
    transcript_lines: int = 0
    unparsed_lines: int = 0
    errors: list[str] = field(default_factory=list)


def _second(iso: object) -> int | None:
    if not isinstance(iso, str):
        return None
    try:
        return int(datetime.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return None


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


class _Rows:
    """Buffer rows and write them as row groups of a bounded number of rows."""

    def __init__(self, path: Path, schema: pa.Schema):
        self.schema, self.rows, self.count = schema, [], 0
        self.writer = pq.ParquetWriter(str(path), schema, compression="zstd")

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= _ROWS_PER_GROUP:
            self._flush()

    def _flush(self) -> None:
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.count += len(self.rows)
            self.rows = []

    def close(self) -> int:
        self._flush()
        self.writer.close()
        return self.count


def _shape_steps(unit_dir: Path, out: Path, store: str, conv: str, include_thinking: bool):
    """steps.parquet -> step.parquet. Returns (rows, undecoded, {idx: creation second of the current row})."""
    current: dict[int, int | None] = {}
    undecoded = 0
    rows = _Rows(out / "step.parquet", STEP_SCHEMA)
    source = unit_dir / "steps.parquet"
    if source.exists():
        steps = pq.ParquetFile(source)
        columns = ["idx", "step_type", "status", "source_present", "ingested_at", "step_payload"]
        for g in range(steps.num_row_groups):
            for r in steps.read_row_group(g, columns=columns).to_pylist():
                payload = r["step_payload"] or b""
                error = None
                try:
                    got = read_step(payload)
                except ValueError as e:
                    got, error = {}, f"{type(e).__name__}: {e}"
                    undecoded += 1
                created = got.get("created_at")
                second = created_second(created) if isinstance(created, datetime) else None
                present = r["source_present"] is not False
                if present:
                    current[r["idx"]] = second
                owner = got.get("owner_conversation_id")
                rows.add({
                    "store": store, "conversation_id": conv, "idx": r["idx"],
                    "source_present": present, "lake_ingested_at": r["ingested_at"],
                    "step_type": r["step_type"], "step_type_name": STEP_TYPE_NAMES.get(r["step_type"]),
                    "status": r["status"], "status_name": STATUS_NAMES.get(r["status"]),
                    "is_cleared": r["status"] == STATUS_CLEARED,
                    "created_at": created, "created_second": second,
                    "source": got.get("source"), "source_name": SOURCE_NAMES.get(got.get("source")),  # type: ignore[arg-type]
                    "execution_id": got.get("execution_id"), "generation_idx": got.get("generation_idx"),
                    "owner_conversation_id": owner,
                    "is_inherited": (owner != conv) if owner is not None else None,
                    "model_enum": got.get("model_enum"),
                    "input_tokens": got.get("input_tokens"), "output_tokens": got.get("output_tokens"),
                    "cache_read_tokens": got.get("cache_read_tokens"),
                    "thinking_tokens": got.get("thinking_tokens"),
                    "tool_call_id": got.get("tool_call_id"), "tool_name": got.get("tool_name"),
                    "tool_args": got.get("tool_args"),
                    "text": got.get("text"),
                    "thinking": got.get("thinking") if include_thinking else None,
                    "tool_output": got.get("tool_output"),
                    "payload_bytes": len(payload), "decode_error": error,
                })
    return rows.close(), undecoded, current


def _shape_transcript(unit_dir: Path, out: Path, store: str, conv: str, include_thinking: bool,
                      current: dict[int, int | None]):
    """The full transcript's bytes in files.parquet -> transcript_line.parquet. Returns (rows, unparsed)."""
    rows = _Rows(out / "transcript_line.parquet", LINE_SCHEMA)
    unparsed = 0
    files = unit_dir / "files.parquet"
    content = None
    if files.exists():
        table = pq.read_table(files, columns=["kind", "content", "source_present"],
                              filters=[("kind", "=", "transcript_full")])
        candidates = table.to_pylist()
        # A file the source still has wins over one the lake kept after it went.
        candidates.sort(key=lambda r: r["source_present"] is not True)
        content = next((r["content"] for r in candidates if r["content"] is not None), None)
    parsed: list[dict] = []
    if content is not None:
        for raw in content.decode("utf-8", errors="replace").splitlines():
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                obj = None
            if not isinstance(obj, dict):
                unparsed += 1
                continue
            parsed.append(obj)
    # The last line at a step's own creation second is the one written for it.
    last_at: dict[int, int] = {}
    for line_no, obj in enumerate(parsed):
        index = _as_int(obj.get("step_index"))
        if index is not None and index in current and current[index] is not None \
                and _second(obj.get("created_at")) == current[index]:
            last_at[index] = line_no
    for line_no, obj in enumerate(parsed):
        index = _as_int(obj.get("step_index"))
        second = _second(obj.get("created_at"))
        text, thinking = _as_str(obj.get("content")), _as_str(obj.get("thinking"))
        rows.add({
            "store": store, "conversation_id": conv, "line_no": line_no, "step_index": index,
            "type": _as_str(obj.get("type")), "source": _as_str(obj.get("source")),
            "status": _as_str(obj.get("status")),
            "created_at": datetime.fromtimestamp(second, tz=timezone.utc) if second is not None else None,
            "created_second": second,
            "content": text, "thinking": thinking if include_thinking else None,
            "tool_calls": json.dumps(obj["tool_calls"]) if obj.get("tool_calls") is not None else None,
            "input_tokens": _as_int(obj.get("input_tokens")), "output_tokens": _as_int(obj.get("output_tokens")),
            "cache_read_tokens": _as_int(obj.get("cache_read_tokens")),
            "model": _as_str(obj.get("model")),
            "media": json.dumps(obj["media"]) if obj.get("media") is not None else None,
            "error": _as_str(obj.get("error")),
            "is_truncated": bool(_TRUNCATED.search(text or "") or _TRUNCATED.search(thinking or "")),
            "is_current": index is not None and last_at.get(index) == line_no,
        })
    return rows.close(), unparsed


def shape_lake(lake_root: Path | str, *, include_thinking: bool = True, force: bool = False,
               on_unit: OnUnit | None = None) -> ShapeResult:
    """Derive `<lake_root>/antigravity-decoded/` from `<lake_root>/antigravity/`.

    One directory per conversation unit, `step.parquet` and
    `transcript_line.parquet`, replaced whole when the lake unit it came from
    was rewritten, when this module's format version changed, or when
    `include_thinking` differs from what the unit was built with. Nothing in
    the lake is touched; deleting the derived tree loses nothing.

    `include_thinking=False` writes no thinking text into either table. The
    lake keeps it regardless (it is the archive).

    `on_unit` is called with a `UnitEvent` (phase `decode`) as each
    conversation starts and ends; one that needs no work reports only its end."""
    lake_root = Path(lake_root)
    harness_dir = lake_root / HARNESS_DIR
    tree = lake_root / SHAPED_DIR
    tree.mkdir(parents=True, exist_ok=True)
    result = ShapeResult(tree=tree)
    manifest_path = tree / SHAPED_MANIFEST
    previous = {r["out_relpath"]: r for r in pq.read_table(manifest_path).to_pylist()} if manifest_path.exists() else {}
    rows: dict[str, dict] = {}
    units = [u for u in read_manifest(harness_dir)
             if u["unit_kind"] == "conversation" and (harness_dir / u["out_relpath"]).is_dir()]

    def report(index: int, unit: dict, status: str | None = None, rows: dict[str, int] | None = None,
               error: str | None = None) -> None:
        if on_unit is not None:
            on_unit(UnitEvent("decode", index, len(units), unit["store"], unit["unit_kind"], unit["unit_id"],
                              status=status, rows=rows or {}, error=error))

    for index, unit in enumerate(units, 1):
        rel = unit["out_relpath"]
        # Changes when the unit is rewritten, and when its source goes away.
        derived_from = f"{unit['last_ingested_at']}|{unit['status'] == 'source_missing'}"
        prev = previous.get(rel)
        final = tree / rel
        if (not force and prev and prev["status"] == "written" and prev["derived_from"] == derived_from
                and prev["shaped_format_version"] == SHAPED_FORMAT_VERSION
                and prev["include_thinking"] == include_thinking and final.is_dir()):
            rows[rel] = prev
            result.unchanged += 1
            report(index, unit, "unchanged")
            continue
        report(index, unit)
        store, conv = unit["store"], unit["unit_id"]
        tmp = final.parent / f".{final.name}.tmp-{uuid.uuid4().hex}"
        base = {"store": store, "conversation_id": conv, "out_relpath": rel, "derived_from": derived_from,
                "shaped_format_version": SHAPED_FORMAT_VERSION, "include_thinking": include_thinking}
        try:
            tmp.mkdir(parents=True)
            steps, undecoded, current = _shape_steps(harness_dir / rel, tmp, store, conv, include_thinking)
            lines, unparsed = _shape_transcript(harness_dir / rel, tmp, store, conv, include_thinking, current)
            if final.exists():
                shutil.rmtree(final)
            os.replace(tmp, final)
        except Exception as e:  # one unit failing must not stop the others
            shutil.rmtree(tmp, ignore_errors=True)
            error = f"{type(e).__name__}: {e}"
            rows[rel] = {**base, "status": "error", "error": error, "steps": None, "undecoded_steps": None,
                         "transcript_lines": None, "unparsed_lines": None}
            result.errors.append(f"{rel}: {error}")
            report(index, unit, "error", error=error)
            continue
        rows[rel] = {**base, "status": "written", "error": None, "steps": steps, "undecoded_steps": undecoded,
                     "transcript_lines": lines, "unparsed_lines": unparsed}
        result.written += 1
        result.steps += steps
        result.undecoded_steps += undecoded
        result.transcript_lines += lines
        result.unparsed_lines += unparsed
        report(index, unit, "written", rows={"steps": steps, "transcript_lines": lines})
    tmp_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp-{uuid.uuid4().hex}")
    pq.write_table(pa.Table.from_pylist([rows[k] for k in sorted(rows)], schema=_MANIFEST), str(tmp_manifest))
    os.replace(tmp_manifest, manifest_path)
    return result


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
#
# Each view exists because it encodes something a query over the tables would
# get wrong (docs/ETL_ARCHITECTURE.md rule 3): which of several archived rows
# is a step, where a cleared step's content can still be read and that it was
# read from there, which index holds a conversation's row, what a pruned
# conversation is. None carries an ORDER BY.

_VIEWS = """
CREATE OR REPLACE VIEW antigravity_conversations AS
WITH archived AS (
    SELECT unit_id AS conversation_id, store, status
    FROM read_parquet('{manifest}')
    WHERE unit_kind = 'conversation'
)
SELECT
    s.conversation_id,
    COALESCE(a.store, s.store) AS store,
    NULLIF(s.title, '') AS title,
    s.preview,
    s.step_count,
    s.last_modified_time,
    s.last_user_input_time,
    s.workspace_uris,
    s.project_id,
    NULLIF(s.agent_name, '') AS agent_name,
    NULLIF(s.parent_conversation_id, '') AS parent_conversation_id,
    s.nesting_depth,
    s.status,
    s.killed <> 0 AS killed,
    -- The app has deleted the conversation and the lake holds what it read
    -- before that. NULL when the lake never read it: only this row is left.
    a.status = 'source_missing' AS is_pruned,
    a.conversation_id IS NOT NULL AS is_archived,
    s.source_present AS in_index
FROM read_parquet('{summaries}', union_by_name = true) s
LEFT JOIN archived a USING (conversation_id)
-- Two stores can index one conversation; its own store's row is the one kept.
QUALIFY row_number() OVER (
    PARTITION BY s.conversation_id
    ORDER BY (s.store = a.store) DESC NULLS LAST, s.source_present DESC, s.last_modified_time DESC
) = 1;

CREATE OR REPLACE VIEW antigravity_steps AS
WITH step AS (SELECT * FROM read_parquet('{steps}', union_by_name = true)),
head AS (
    -- One row per step: what the source holds now, or for a conversation
    -- the app has deleted, the last row the lake read.
    SELECT * FROM step
    QUALIFY row_number() OVER (
        PARTITION BY store, conversation_id, idx
        ORDER BY source_present DESC, lake_ingested_at DESC, payload_bytes DESC
    ) = 1
),
fuller AS (
    -- The same step as the lake read it before the app emptied it.
    SELECT s.* FROM step s
    JOIN head h ON h.store = s.store AND h.conversation_id = s.conversation_id AND h.idx = s.idx
        AND h.created_second = s.created_second AND s.payload_bytes > h.payload_bytes
    QUALIFY row_number() OVER (
        PARTITION BY s.store, s.conversation_id, s.idx ORDER BY s.payload_bytes DESC, s.lake_ingested_at DESC
    ) = 1
),
line AS (SELECT * FROM read_parquet('{lines}', union_by_name = true) WHERE is_current)
SELECT
    h.store, h.conversation_id, h.idx, h.created_at,
    h.step_type, h.step_type_name, h.status, h.status_name, h.is_cleared,
    h.source_name, h.execution_id, h.generation_idx,
    h.owner_conversation_id, h.is_inherited, h.model_enum,
    h.input_tokens, h.output_tokens, h.cache_read_tokens, h.thinking_tokens,
    COALESCE(h.tool_call_id, f.tool_call_id) AS tool_call_id,
    COALESCE(h.tool_name, f.tool_name) AS tool_name,
    COALESCE(h.tool_args, f.tool_args) AS tool_args,
    -- A tool call is a step that states one, in either encoding: a GENERIC
    -- step, or one of the per-tool types the app wrote before (claim 11).
    {is_call} AS is_tool_call,
    -- Each value names where it was read: the database as it is now, an
    -- earlier row the lake kept, or the brain transcript, whose form of a
    -- user, system or tool text is wrapped and can be truncated. What the
    -- transcript holds for a tool call is its output, for any other step
    -- its text.
    COALESCE(h.text, f.text, CASE WHEN NOT {is_call} THEN l.content END) AS text,
    CASE WHEN h.text IS NOT NULL THEN 'database' WHEN f.text IS NOT NULL THEN 'archive'
         WHEN NOT {is_call} AND l.content IS NOT NULL THEN 'transcript' END AS text_source,
    COALESCE(h.thinking, f.thinking, l.thinking) AS thinking,
    CASE WHEN h.thinking IS NOT NULL THEN 'database' WHEN f.thinking IS NOT NULL THEN 'archive'
         WHEN l.thinking IS NOT NULL THEN 'transcript' END AS thinking_source,
    COALESCE(h.tool_output, f.tool_output, CASE WHEN {is_call} THEN l.content END) AS tool_output,
    CASE WHEN h.tool_output IS NOT NULL THEN 'database' WHEN f.tool_output IS NOT NULL THEN 'archive'
         WHEN {is_call} AND l.content IS NOT NULL THEN 'transcript' END AS tool_output_source,
    l.is_truncated AS transcript_is_truncated,
    h.source_present AS in_source
FROM head h
LEFT JOIN fuller f ON f.store = h.store AND f.conversation_id = h.conversation_id AND f.idx = h.idx
LEFT JOIN line l ON l.store = h.store AND l.conversation_id = h.conversation_id AND l.step_index = h.idx;

CREATE OR REPLACE VIEW antigravity_tool_calls AS
-- A tool call and its result are one step (claim 18). Keyed on the step: a
-- call id repeats inside a conversation and across forks. Both encodings
-- are here, and `step_type_name` says which. `is_error` is read from the
-- status number, so a status nobody has named is still not an error
-- instead of NULL; and a call answered by an ERROR_MESSAGE step is an
-- error whatever its status, with the message in `error_text`.
SELECT store, conversation_id, idx, created_at, step_type_name, tool_call_id, tool_name, tool_args,
       tool_output, tool_output_source, status, status_name,
       COALESCE(status = {error}, FALSE) OR step_type = {error_message} AS is_error,
       CASE WHEN step_type = {error_message} THEN text END AS error_text,
       is_cleared, is_inherited, execution_id
FROM antigravity_steps WHERE is_tool_call;

CREATE OR REPLACE VIEW antigravity_usage AS
-- One row per step that states usage. A model response states its
-- generation's; a checkpoint states the usage of its own call, under the
-- generation index of the response after it and with no model, so
-- `step_type_name` is what tells the two apart. Not a fork's inherited
-- steps, which are the source conversation's usage.
SELECT store, conversation_id, idx, created_at, step_type_name, generation_idx, model_enum,
       input_tokens, output_tokens, cache_read_tokens, thinking_tokens, is_cleared
FROM antigravity_steps
WHERE input_tokens IS NOT NULL AND is_inherited IS NOT TRUE;

CREATE OR REPLACE VIEW antigravity_tasks AS
-- A background task has no id of its own (claim 19). The tool step that
-- started it, its log file and the system message that reports it share a
-- step index, and this is the one place that tie is written down.
WITH log AS (
    SELECT store, conversation_id,
           CAST(regexp_extract(relpath, '/tasks/task-([0-9]+)[.]log$', 1) AS BIGINT) AS idx,
           relpath, size_bytes, source_present, content
    FROM read_parquet('{files}', union_by_name = true)
    WHERE kind = 'task_log'
),
named AS (
    -- Not a fork's replayed messages: the task they name is the source's.
    SELECT store, idx AS reported_idx, created_at AS reported_at, text,
           regexp_extract(text, 'Task id "([0-9a-f-]+)/task-([0-9]+)" (finished|was canceled) with result',
                          ['conversation_id', 'idx', 'verb']) AS task
    FROM antigravity_steps
    WHERE step_type = {system_message} AND is_inherited IS NOT TRUE AND text LIKE '%Task id "%with result%'
),
report AS (
    SELECT store, task.conversation_id AS conversation_id, CAST(task.idx AS BIGINT) AS idx,
           reported_idx, reported_at,
           CASE task.verb WHEN 'finished' THEN 'finished' ELSE 'canceled' END AS outcome,
           -- The first mention is the task's own: it comes before the output.
           TRY_CAST(NULLIF(regexp_extract(text, 'exited with code (-?[0-9]+)', 1), '') AS BIGINT) AS exit_code
    FROM named
    WHERE task.idx <> ''
    QUALIFY row_number() OVER (PARTITION BY store, task.conversation_id, task.idx ORDER BY reported_idx DESC) = 1
),
task AS (
    SELECT COALESCE(l.store, r.store) AS store,
           COALESCE(l.conversation_id, r.conversation_id) AS conversation_id,
           COALESCE(l.idx, r.idx) AS idx,
           r.outcome, r.exit_code, r.reported_at, r.reported_idx,
           l.relpath, l.size_bytes, l.source_present, l.content
    FROM log l
    FULL JOIN report r ON r.store = l.store AND r.conversation_id = l.conversation_id AND r.idx = l.idx
)
SELECT
    t.store, t.conversation_id, t.idx,
    -- NULL when the step at that index is no longer the one that started the
    -- task: a revert re-uses indices (claim 14) and leaves the old log behind.
    s.tool_name, s.tool_args, s.created_at AS started_at, s.status_name AS launch_status_name,
    -- NULL when nothing reported it: still running, or the conversation ended first.
    t.outcome, t.exit_code, t.reported_at, t.reported_idx,
    t.relpath IS NOT NULL AS has_log, t.relpath AS log_relpath, t.size_bytes AS log_size_bytes,
    t.source_present AS log_in_source, try(decode(t.content)) AS log_text
FROM task t
LEFT JOIN antigravity_steps s ON s.store = t.store AND s.conversation_id = t.conversation_id AND s.idx = t.idx
    AND s.tool_name IN ({task_tools});
"""

# The tools whose step starts a background task (claim 19).
TASK_TOOLS = ("run_command", "schedule")
VIEW_NAMES = ("antigravity_conversations", "antigravity_steps", "antigravity_tool_calls", "antigravity_usage",
              "antigravity_tasks")


def create_views(conn, lake_root: Path | str) -> list[str]:
    """Create the `antigravity_*` views in `conn` over the lake and its derived tree.

    Raises FileNotFoundError when there is nothing to read, instead of
    creating views that fail at query time."""
    lake_root = Path(lake_root)
    harness_dir, tree = lake_root / HARNESS_DIR, lake_root / SHAPED_DIR
    globs = {
        "manifest": harness_dir / "_manifest.parquet",
        "summaries": harness_dir / "*" / "summaries" / "conversation_summaries.parquet",
        "steps": tree / "*" / "conversations" / "*" / "step.parquet",
        "lines": tree / "*" / "conversations" / "*" / "transcript_line.parquet",
        "files": harness_dir / "*" / "conversations" / "*" / "files.parquet",
    }
    for name, pattern in globs.items():
        if not glob.glob(str(pattern)):
            raise FileNotFoundError(f"nothing to read for {name}: {pattern} (run the lake, then shape it)")
    is_call = (f"(h.step_type = {GENERIC} OR COALESCE(h.tool_call_id, f.tool_call_id, "
               "h.tool_name, f.tool_name) IS NOT NULL)")
    sql = _VIEWS.format(is_call=is_call, error=STATUS_ERROR, error_message=ERROR_MESSAGE,
                        system_message=SYSTEM_MESSAGE, task_tools=", ".join(f"'{t}'" for t in TASK_TOOLS),
                        **{k: str(v).replace("'", "''") for k, v in globs.items()})
    for statement in sql.split(";\n"):
        if statement.strip():
            conn.execute(statement)
    return list(VIEW_NAMES)


VIEWS_DATABASE = "antigravity.duckdb"


def write_views_database(lake_root: Path | str) -> Path:
    """`<derived tree>/antigravity.duckdb`: a DuckDB file holding only the views.

    Rebuilt whole each time. It stores no rows, so it is as disposable as the
    tree it sits in; the views name this lake's files by absolute path."""
    import duckdb

    lake_root = Path(lake_root)
    path = lake_root / SHAPED_DIR / VIEWS_DATABASE
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    conn = duckdb.connect(str(tmp))
    try:
        create_views(conn, lake_root)
    except BaseException:
        conn.close()
        tmp.unlink(missing_ok=True)
        raise
    conn.close()
    os.replace(tmp, path)
    return path
