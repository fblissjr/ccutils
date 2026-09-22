"""Canaries for the Antigravity on-disk contract (docs/ANTIGRAVITY_CONTRACT.md).

Same shape as test_jsonl_contract.py. Antigravity is unversioned upstream and
its format has already changed once inside the corpus, so every assumption the
lake makes is written down as a claim and checked against the real store on
this machine. Each claim gets:

- a corpus canary, which reads the real `<HOME>/.gemini` stores (read-only,
  `immutable=1`) and goes red when Antigravity changes what it writes. It skips
  loudly when there is no store; a skip is honest because the claim is about
  what the app writes.
- an oracle test, which feeds the same checker a deliberate violation. A
  canary that cannot fail looks exactly like one with nothing to report.

Claims are scanned over the whole corpus, not sampled: the corpus is small
(hundreds of files) and a rare-event claim cannot be established by a sample
smaller than the event's inverse rate.
"""

from __future__ import annotations

import collections
import json
import math
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from helpers_antigravity import (
    REAL_CONVERSATION_DDL,
    REAL_SUMMARIES_DDL,
    f_bytes,
    f_timestamp,
    f_varint,
    get,
    parse,
    timestamp_seconds,
)

ROOT = Path.home() / ".gemini"
DB_STORES = ("antigravity", "antigravity-cli")


def _dbs():
    return [db for s in DB_STORES if (ROOT / s / "conversations").is_dir()
            for db in sorted((ROOT / s / "conversations").glob("*.db"))]


@pytest.fixture(scope="module")
def dbs():
    found = _dbs()
    if not found:
        pytest.skip("no local Antigravity store to check the contract against")
    return found


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?immutable=1", uri=True)


def _ddl(conn) -> tuple[str, ...]:
    return tuple(r[0] for r in conn.execute(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"))


# ---------------------------------------------------------------------------
# Claim 1: every conversation file has the fixtures' DDL; every summaries table
# has the fixtures' columns.
# ---------------------------------------------------------------------------


def _columns_of(ddl: tuple[str, ...]) -> list[tuple[str, str]]:
    conn = sqlite3.connect(":memory:")
    for stmt in ddl:
        if stmt.startswith("CREATE TABLE"):
            conn.execute(stmt)
    table = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchone()[0]
    cols = [(r[1], r[2].lower()) for r in conn.execute(f"PRAGMA table_info(`{table}`)")]
    conn.close()
    return cols


def ddl_violations(ddl_by_file: dict[str, tuple[str, ...]], expected: tuple[str, ...]) -> list[str]:
    return sorted(name for name, ddl in ddl_by_file.items() if ddl != expected)


class TestSchemaIsUniform:
    def test_corpus_holds(self, dbs):
        by_file = {}
        for db in dbs:
            conn = _ro(db)
            by_file[f"{db.parent.parent.name}/{db.name}"] = _ddl(conn)
            conn.close()
        assert ddl_violations(by_file, REAL_CONVERSATION_DDL) == [], (
            "a conversation DB's schema differs from REAL_CONVERSATION_DDL: the fixtures no "
            "longer model the real files and schema.CONVERSATION_TABLES may be stale")
        # The summaries DDL TEXT differs between stores: the CLI's table gained
        # `raw_summary` through a migration and declares it unquoted. The
        # columns (names, order, declared types) are what the writer relies on.
        expected = _columns_of(REAL_SUMMARIES_DDL)
        for store in DB_STORES:
            path = ROOT / store / "conversation_summaries.db"
            if path.exists():
                conn = _ro(path)
                actual = [(r[1], r[2].lower()) for r in conn.execute("PRAGMA table_info(conversation_summaries)")]
                conn.close()
                assert actual == expected, store

    def test_oracle(self):
        drifted = REAL_CONVERSATION_DDL[:-1] + ("CREATE TABLE `trajectory_metadata_blob` (`id` text)",)
        assert ddl_violations({"a.db": REAL_CONVERSATION_DDL, "b.db": drifted}, REAL_CONVERSATION_DDL) == ["b.db"]


# ---------------------------------------------------------------------------
# Claim 2: the filename is the conversation id.
# ---------------------------------------------------------------------------


def identity_violations(rows: list[tuple[str, list[str], int]]) -> list[str]:
    """(stem, trajectory_meta cascade_ids, step count) -> stems breaking the claim.

    Empty files are exempt: measured, one of two carries a cascade_id found
    nowhere else on disk."""
    return [stem for stem, ids, steps in rows if steps and ids != [stem]]


class TestFilenameIsTheConversationId:
    def test_corpus_holds(self, dbs):
        rows = []
        for db in dbs:
            conn = _ro(db)
            ids = [r[0] for r in conn.execute("SELECT cascade_id FROM trajectory_meta")]
            steps = conn.execute("SELECT count(*) FROM steps").fetchone()[0]
            conn.close()
            rows.append((db.stem, ids, steps))
        assert identity_violations(rows) == []

    def test_oracle(self):
        assert identity_violations([("a", ["a"], 3), ("b", ["x"], 2), ("c", ["y"], 0), ("d", ["d", "d2"], 1)]) == ["b", "d"]


# ---------------------------------------------------------------------------
# Claim 3: the step blob columns are copies of fields of step_payload.
# ---------------------------------------------------------------------------

# column -> field number of gemini_coder.Step
_COPIED_FIELDS = {"metadata": 5, "error_details": 31, "permissions": 133, "task_details": 148}


def payload_copy_violations(rows) -> list[tuple[int, str]]:
    """rows of (idx, step_type, status, has_sub, metadata, error_details,
    permissions, task_details, step_payload)."""
    bad = []
    for idx, step_type, status, has_sub, *copies, payload in rows:
        fields: dict[int, object] = {}
        for f, _wt, v in parse(payload):
            fields[f] = v
        if fields.get(1, 0) != step_type:
            bad.append((idx, "step_type"))
        if fields.get(4, 0) != status:
            bad.append((idx, "status"))
        if bool(has_sub) != (6 in fields):
            bad.append((idx, "has_subtrajectory"))
        for (col, f), value in zip(_COPIED_FIELDS.items(), copies):
            if value != fields.get(f):
                bad.append((idx, col))
    return bad


class TestBlobColumnsAreCopies:
    def test_corpus_holds(self, dbs):
        bad = []
        n = 0
        for db in dbs:
            conn = _ro(db)
            rows = conn.execute(
                "SELECT idx, step_type, status, has_subtrajectory, metadata, error_details, "
                "permissions, task_details, step_payload FROM steps").fetchall()
            conn.close()
            n += len(rows)
            bad += [(db.stem, *v) for v in payload_copy_violations(rows)]
        assert n > 0
        assert bad == [], f"{len(bad)} of {n} steps: a column no longer mirrors its Step field, e.g. {bad[:3]}"

    def test_oracle(self):
        meta = f_timestamp(1, 1_790_000_000)
        good = f_varint(1, 15) + f_varint(4, 3) + f_bytes(5, meta)
        rows = [
            (0, 15, 3, 0, meta, None, None, None, good),
            (1, 14, 3, 0, meta, None, None, None, good),           # step_type column disagrees
            (2, 15, 3, 0, meta + b"\x00", None, None, None, good),  # metadata column not a copy
            (3, 15, 3, 1, meta, None, None, None, good),           # has_subtrajectory without field 6
        ]
        assert payload_copy_violations(rows) == [(1, "step_type"), (2, "metadata"), (3, "has_subtrajectory")]


# ---------------------------------------------------------------------------
# Claim 4: every stored value has its column's declared storage type.
# ---------------------------------------------------------------------------


def typeof_violations(counts: dict[str, collections.Counter], expected: dict[str, str]) -> list[str]:
    allowed = {"text": {"text"}, "integer": {"integer"}, "blob": {"blob"}, "real": {"real"}}
    return sorted(col for col, c in counts.items()
                  if set(c) - {"null"} - allowed[expected[col]])


class TestStorageTypes:
    def test_corpus_holds(self, dbs):
        from ccutils.parsers.antigravity.schema import CONVERSATION_TABLES, SUMMARIES_TABLES
        counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        expected = {}
        sources = [(db, CONVERSATION_TABLES) for db in dbs] + [
            (ROOT / s / "conversation_summaries.db", SUMMARIES_TABLES)
            for s in DB_STORES if (ROOT / s / "conversation_summaries.db").exists()]
        for path, tables in sources:
            conn = _ro(path)
            for table, cols in tables.items():
                for col, kind in cols:
                    expected[f"{table}.{col}"] = kind
                    for t, n in conn.execute(f"SELECT typeof(`{col}`), count(*) FROM `{table}` GROUP BY 1"):
                        counts[f"{table}.{col}"][t] += n
            conn.close()
        assert typeof_violations(counts, expected) == []

    def test_oracle(self):
        counts = {"steps.idx": collections.Counter({"integer": 5}),
                  "steps.step_payload": collections.Counter({"blob": 4, "text": 1}),
                  "steps.render_info": collections.Counter({"null": 9})}
        expected = {"steps.idx": "integer", "steps.step_payload": "blob", "steps.render_info": "blob"}
        assert typeof_violations(counts, expected) == ["steps.step_payload"]


# ---------------------------------------------------------------------------
# Claim 5: the files are in WAL mode (so recent rows can live only in -wal).
# ---------------------------------------------------------------------------


def not_wal(headers: dict[str, bytes]) -> list[str]:
    return sorted(name for name, h in headers.items() if len(h) < 20 or (h[18], h[19]) != (2, 2))


class TestWalMode:
    def test_corpus_holds(self, dbs):
        paths = list(dbs) + [ROOT / s / "conversation_summaries.db" for s in DB_STORES]
        headers = {}
        for p in paths:
            if p.exists():
                with open(p, "rb") as f:
                    headers[str(p.relative_to(ROOT))] = f.read(100)
        assert not_wal(headers) == [], (
            "a file left WAL mode; the snapshot still works, but docs/ANTIGRAVITY_CONTRACT.md claim 5 is stale")

    def test_oracle(self):
        wal = bytes(18) + b"\x02\x02" + bytes(80)
        rollback = bytes(18) + b"\x01\x01" + bytes(80)
        assert not_wal({"a": wal, "b": rollback}) == ["b"]


# ---------------------------------------------------------------------------
# Claim 6: step-type names are stated in the brain transcript, and its
# created_at is the step metadata's created_at (field 5.1) to the second.
# ---------------------------------------------------------------------------

# Measured: 5 of ~50,000 joined lines name a type other than their step_type's
# dominant name. The threshold is ~10x headroom.
MAX_NAME_DISAGREEMENT = 0.001


def name_disagreement(pairs) -> tuple[dict[int, str], float]:
    """(step_type, stated name) pairs -> (dominant name per type, disagreement rate)."""
    by_type: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for step_type, name in pairs:
        by_type[step_type][name] += 1
    dominant = {t: c.most_common(1)[0][0] for t, c in by_type.items()}
    total = sum(sum(c.values()) for c in by_type.values())
    off = sum(sum(c.values()) - c[dominant[t]] for t, c in by_type.items())
    return dominant, (off / total if total else 0.0)


def created_at_mismatches(pairs) -> int:
    """(metadata created_at seconds, transcript created_at ISO string) pairs."""
    bad = 0
    for seconds, iso in pairs:
        stated = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
        if seconds is None or int(seconds) != int(stated):
            bad += 1
    return bad


def _transcript(db: Path) -> dict[int, list[dict]]:
    path = db.parent.parent / "brain" / db.stem / ".system_generated" / "logs" / "transcript_full.jsonl"
    out: dict[int, list[dict]] = collections.defaultdict(list)
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # measured: 1 unparseable line corpus-wide
        if isinstance(obj, dict) and "step_index" in obj:
            out[obj["step_index"]].append(obj)
    return out


class TestStepNamesAreStated:
    def test_corpus_holds(self, dbs):
        names, times = [], []
        for db in dbs:
            lines = _transcript(db)
            if not lines:
                continue
            conn = _ro(db)
            for idx, step_type, metadata in conn.execute("SELECT idx, step_type, metadata FROM steps"):
                joined = lines.get(idx) or []
                for line in joined:
                    names.append((step_type, line.get("type")))
                # A step_index written twice can carry two different times
                # (measured: all 32 mismatches corpus-wide are on duplicated
                # indices); the claim is about indices written once.
                if len(joined) == 1 and joined[0].get("created_at"):
                    times.append((timestamp_seconds(get(metadata, 1)), joined[0]["created_at"]))
            conn.close()
        assert len(names) > 1000, "too few joined lines to say anything"
        dominant, rate = name_disagreement(names)
        assert rate <= MAX_NAME_DISAGREEMENT, f"{rate:.4%} of lines disagree with their type's name"
        assert len(set(dominant.values())) == len(dominant), f"two step types share a name: {dominant}"
        assert created_at_mismatches(times) == 0

    def test_oracle(self):
        pairs = [(15, "PLANNER_RESPONSE")] * 99 + [(15, "GENERIC")]
        dominant, rate = name_disagreement(pairs)
        assert dominant == {15: "PLANNER_RESPONSE"} and rate == pytest.approx(0.01)
        assert created_at_mismatches([(1_790_000_000.9, "2026-09-21T14:13:20Z"), (5.0, "2026-09-21T14:13:20Z")]) == 1


# ---------------------------------------------------------------------------
# Claim 7: legacy conversations/*.pb are encrypted (random-looking, no parse).
# ---------------------------------------------------------------------------


def entropy(data: bytes) -> float:
    counts = collections.Counter(data)
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def readable_legacy(files: dict[str, bytes]) -> list[str]:
    """Legacy files that do NOT look encrypted: low entropy, or a valid parse."""
    out = []
    for name, data in files.items():
        head = data[: 1 << 20]
        try:
            parse(head)
            parses = True
        except ValueError:
            parses = False
        if entropy(head) < 7.9 or parses:
            out.append(name)
    return sorted(out)


class TestLegacyPbIsEncrypted:
    def test_corpus_holds(self):
        files = {}
        for store in DB_STORES + ("antigravity-ide",):
            for p in sorted((ROOT / store / "conversations").glob("*.pb")):
                with open(p, "rb") as f:
                    files[f"{store}/{p.name}"] = f.read(1 << 20)
        if not files:
            pytest.skip("no legacy .pb files")
        assert readable_legacy(files) == [], (
            "a legacy .pb is readable: the lake archives it as ciphertext and would miss its content")

    def test_oracle(self):
        rng = random.Random(3)
        cipher = rng.randbytes(1 << 16)
        plain = (f_varint(1, 15) + f_bytes(5, b"metadata")) * 3000
        assert readable_legacy({"c.pb": cipher, "p.pb": plain}) == ["p.pb"]


# ---------------------------------------------------------------------------
# Claim 8: logs/chunks/ is an exact split of the whole transcript file.
# ---------------------------------------------------------------------------


def chunk_mismatches(pairs) -> list[str]:
    """(name, whole bytes, [chunk bytes in name order]) -> names that differ."""
    return [name for name, whole, chunks in pairs if b"".join(chunks) != whole]


class TestChunksDuplicateTheWholeFile:
    def test_corpus_holds(self, dbs):
        pairs = []
        for store in DB_STORES:
            brain = ROOT / store / "brain"
            if not brain.is_dir():
                continue
            for d in sorted(brain.iterdir()):
                for kind in ("transcript_full", "transcript"):
                    chunks = d / ".system_generated" / "logs" / "chunks" / kind
                    whole = d / ".system_generated" / "logs" / f"{kind}.jsonl"
                    if chunks.is_dir() and whole.exists():
                        pairs.append((f"{store}/{d.name}/{kind}", whole.read_bytes(),
                                      [p.read_bytes() for p in sorted(chunks.glob("*.jsonl"))]))
        assert pairs, "no chunked transcripts found"
        assert chunk_mismatches(pairs) == [], (
            "chunks/ is no longer a duplicate of the whole file; the lake excludes it and would lose data")

    def test_oracle(self):
        assert chunk_mismatches([("ok", b"ab", [b"a", b"b"]), ("bad", b"ab", [b"a"])]) == ["bad"]


# ---------------------------------------------------------------------------
# Claim 9: subagent parent and depth are stated, together.
# ---------------------------------------------------------------------------


def subagent_statement_violations(rows) -> list[str]:
    """(conversation_id, parent_conversation_id, nesting_depth): depth > 0 iff a parent is named."""
    return [cid for cid, parent, depth in rows if (depth > 0) != (parent != "")]


class TestSubagentLinksAreStated:
    def test_corpus_holds(self, dbs):
        rows = []
        for store in DB_STORES:
            path = ROOT / store / "conversation_summaries.db"
            if path.exists():
                conn = _ro(path)
                rows += conn.execute(
                    "SELECT conversation_id, parent_conversation_id, nesting_depth FROM conversation_summaries").fetchall()
                conn.close()
        assert any(depth > 0 for _c, _p, depth in rows), "no subagent rows to check"
        assert subagent_statement_violations(rows) == []

    def test_oracle(self):
        rows = [("a", "", 0), ("b", "a", 1), ("c", "", 2), ("d", "a", 0)]
        assert subagent_statement_violations(rows) == ["c", "d"]


# ---------------------------------------------------------------------------
# Claim 10: conversation ids do not collide across the stores the lake reads,
# and the backup store adds nothing.
# ---------------------------------------------------------------------------


def colliding(ids_by_store: dict[str, set[str]]) -> set[str]:
    seen: dict[str, str] = {}
    out = set()
    for store, ids in ids_by_store.items():
        for i in ids:
            if i in seen:
                out.add(i)
            seen[i] = store
    return out


class TestStoresAreDisjoint:
    def test_db_conversation_ids_are_unique_across_stores(self, dbs):
        ids = {s: {p.stem for p in (ROOT / s / "conversations").glob("*.db")} for s in DB_STORES}
        assert colliding(ids) == set()

    def test_backup_holds_no_db_and_mirrors_the_ide_copy(self):
        backup, ide = ROOT / "antigravity-backup", ROOT / "antigravity-ide"
        if not backup.is_dir():
            pytest.skip("no antigravity-backup store")
        assert not list((backup / "conversations").glob("*.db"))
        if ide.is_dir():
            for p in sorted((backup / "conversations").glob("*.pb")):
                twin = ide / "conversations" / p.name
                assert twin.exists() and twin.read_bytes() == p.read_bytes(), p.name

    def test_oracle(self):
        assert colliding({"a": {"x", "y"}, "b": {"y", "z"}}) == {"y"}


# ---------------------------------------------------------------------------
# Claim 11: tool calls moved to GENERIC (132) steps, the hub in July 2026 and
# the CLI in August; both encodings are in the corpus and a decoder must read
# both. The canary fires if the per-tool encoding comes back.
# ---------------------------------------------------------------------------

LEGACY_TOOL_STEP_TYPES = {5, 7, 8, 9, 21, 25}
# Measured: the hub's last per-tool step is in July 2026; the CLI wrote 212 in
# August next to 1,292 GENERIC ones, and none since.
GENERIC_CUTOVER = datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()


def legacy_after_cutover(rows) -> int:
    """(step_type, metadata) rows: legacy tool step types created after the cutover."""
    return sum(1 for step_type, meta in rows
               if step_type in LEGACY_TOOL_STEP_TYPES
               and (timestamp_seconds(get(meta, 1)) or 0) >= GENERIC_CUTOVER)


class TestGenericToolSteps:
    def test_corpus_holds(self, dbs):
        n = 0
        for db in dbs:
            conn = _ro(db)
            n += legacy_after_cutover(conn.execute("SELECT step_type, metadata FROM steps"))
            conn.close()
        assert n == 0, (
            f"{n} per-tool steps created after the GENERIC cutover: Antigravity writes the "
            "per-tool encoding again; update docs/ANTIGRAVITY_CONTRACT.md claim 11")

    def test_oracle(self):
        after = f_timestamp(1, int(GENERIC_CUTOVER) + 60)
        before = f_timestamp(1, int(GENERIC_CUTOVER) - 60)
        assert legacy_after_cutover([(8, after), (8, before), (132, after)]) == 1


# ---------------------------------------------------------------------------
# Claim 14: every step states its creation time, and reverts re-use indices
# ---------------------------------------------------------------------------


def missing_created_at(rows) -> list[int]:
    """(idx, metadata) rows whose Step metadata carries no field 1."""
    return [idx for idx, meta in rows if not meta or get(meta, 1) is None]


class TestStepCreationTimeIsStated:
    """The lake's revert detection compares Step metadata field 1 per idx
    (parsers/antigravity/__init__.py::supersedes). A step without it is
    skipped there, so if Antigravity stops writing it, reverts go back to
    being overwritten silently. This canary is what would say so."""

    def test_corpus_holds(self, dbs):
        bad, n = [], 0
        for db in dbs:
            conn = _ro(db)
            rows = conn.execute("SELECT idx, metadata FROM steps").fetchall()
            conn.close()
            n += len(rows)
            bad += [(db.stem, i) for i in missing_created_at(rows)]
        assert n > 0
        assert bad == [], f"{len(bad)} of {n} steps state no creation time, e.g. {bad[:3]}"

    def test_oracle(self):
        rows = [(0, f_timestamp(1, 1_790_000_000)), (1, f_bytes(12, b"exec-1")), (2, None)]
        assert missing_created_at(rows) == [1, 2]
