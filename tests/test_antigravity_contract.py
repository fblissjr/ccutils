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
    # mode=ro as well: immutable=1 alone creates a file that is not there.
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


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


def unfilled_columns(rows) -> set[tuple[int, str]]:
    """(idx, column) where the column is NULL and the payload states its field.

    The payload still carries the step there, so this is the one way a column
    may differ from its field. A column that states what the payload lacks, or
    states something else, is never in this set."""
    out = set()
    for idx, _step_type, _status, _has_sub, *copies, payload in rows:
        stated = {f for f, _wt, _v in parse(payload)}
        out |= {(idx, col) for (col, f), value in zip(_COPIED_FIELDS.items(), copies)
                if value is None and f in stated}
    return out


# Only `permissions` has been seen left NULL: 40 of 120,475 steps on the
# owner's Mac, 2026-11-20, all GENERIC steps of the desktop app's store.
UNFILLED_ALLOWED = ("permissions",)
MAX_UNFILLED = 0.005


def copy_verdict(rows) -> tuple[list[tuple[int, str]], int]:
    """(violations that contradict the claim, columns left NULL that it allows)."""
    found = payload_copy_violations(rows)
    tolerated = {v for v in unfilled_columns(rows) if v[1] in UNFILLED_ALLOWED} if found else set()
    return [v for v in found if v not in tolerated], len(tolerated)


def copy_violation_shapes(rows) -> collections.Counter:
    """How each copied column disagrees with its Step field, as shapes: no
    bytes of either side. Keyed (column, step_type, status, shape)."""
    shapes: collections.Counter = collections.Counter()
    for _idx, step_type, status, _has_sub, *copies, payload in rows:
        occurrences: dict[int, list] = collections.defaultdict(list)
        for f, _wt, v in parse(payload):
            occurrences[f].append(v)
        for (col, f), value in zip(_COPIED_FIELDS.items(), copies):
            found = occurrences.get(f) or []
            if value == (found[-1] if found else None):
                continue
            if value is None:
                shape = f"column NULL, field stated {len(found)}x"
            elif not found:
                shape = "column stated, field absent"
            elif value in found:
                shape = f"column equals an earlier occurrence of {len(found)}"
            elif isinstance(value, bytes) and value == b"".join(v for v in found if isinstance(v, bytes)):
                shape = f"column is the {len(found)} occurrences joined"
            else:
                sizes = "longer" if len(value) > len(found[-1]) else "shorter" if len(value) < len(found[-1]) else "same length"
                shape = f"both stated {len(found)}x, column {sizes}"
            shapes[(col, step_type, status, shape)] += 1
    return shapes


class TestBlobColumnsAreCopies:
    def test_corpus_holds(self, dbs):
        bad = []
        shapes: collections.Counter = collections.Counter()
        n = unfilled = 0
        for db in dbs:
            conn = _ro(db)
            rows = conn.execute(
                "SELECT idx, step_type, status, has_subtrajectory, metadata, error_details, "
                "permissions, task_details, step_payload FROM steps").fetchall()
            conn.close()
            n += len(rows)
            contradictions, tolerated = copy_verdict(rows)
            unfilled += tolerated
            bad += [(db.parent.parent.name, db.stem, *v) for v in contradictions]
            if contradictions or tolerated:
                shapes += copy_violation_shapes(rows)
        assert n > 0
        print(f"\n{n} steps; a column left NULL beside its stated field: {unfilled}; "
              f"(column, step_type, status, shape): {dict(shapes)}")
        stores = collections.Counter(b[0] for b in bad)
        assert bad == [], (f"{len(bad)} of {n} steps: a column states what its Step field does not. "
                           f"By store: {dict(stores)}. (column, step_type, status, shape): {dict(shapes)}")
        assert unfilled / n <= MAX_UNFILLED, f"{unfilled} of {n} steps leave a copied column NULL"

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
        twice = good + f_bytes(133, b"first") + f_bytes(133, b"second")
        shaped_rows = [
            (0, 15, 3, 0, meta, None, None, None, good),                       # agrees: no shape
            (1, 15, 3, 0, meta, None, b"first", None, twice),                  # the column kept the first
            (2, 15, 3, 0, meta, None, b"firstsecond", None, twice),            # the column kept both
            (3, 15, 3, 0, meta, None, None, None, twice),                      # the column kept nothing
            (4, 15, 3, 0, meta, None, b"x", None, good),                       # the payload states none
            (5, 15, 3, 0, meta, None, b"longer than it", None, twice),
        ]
        # A column left NULL beside a stated field is the one disagreement in
        # which the payload still carries everything; every other kind is not.
        assert unfilled_columns(shaped_rows) == {(3, "permissions")}
        assert unfilled_columns(rows) == set()
        # Only that one is allowed, and only for `permissions`: a metadata
        # column left NULL, or any column that states something else, fails.
        assert copy_verdict(shaped_rows) == ([(1, "permissions"), (2, "permissions"), (4, "permissions"),
                                              (5, "permissions")], 1)
        assert copy_verdict([(0, 15, 3, 0, None, None, None, None, good)]) == ([(0, "metadata")], 0)
        assert copy_verdict([(0, 15, 3, 0, meta, None, None, None, good)]) == ([], 0)
        assert copy_violation_shapes(shaped_rows) == collections.Counter({
            ("permissions", 15, 3, "column equals an earlier occurrence of 2"): 1,
            ("permissions", 15, 3, "column is the 2 occurrences joined"): 1,
            ("permissions", 15, 3, "column NULL, field stated 2x"): 1,
            ("permissions", 15, 3, "column stated, field absent"): 1,
            ("permissions", 15, 3, "both stated 2x, column longer"): 1,
        })


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

# A revert re-uses step indices and the transcript appends, so an index can
# carry lines from a branch the database no longer holds. A line belongs to a
# step when the index AND the creation second agree. Measured 2026-11-20 on a
# CLI-only store where reverts are common: joined that way no line names a
# type other than its step_type's dominant name, a handful of steps have lines
# at their index and none at their second, and on every index written more
# than once the last line is the database's step. Joined on the index alone
# the same store disagrees on about a quarter of a percent of lines, which is
# what this canary used to measure. Run with -s for the counts.
MAX_NAME_DISAGREEMENT = 0.001
MAX_STEPS_WITHOUT_THEIR_LINE = 0.001
MIN_LAST_LINE_IS_CURRENT = 0.99


def _seconds(iso: str | None) -> int | None:
    try:
        return int(datetime.strptime(iso or "", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return None


def lines_at(joined: list[dict], seconds: float | None) -> list[dict]:
    """The lines at a step's index that were written for THAT step."""
    if seconds is None:
        return []
    return [line for line in joined if _seconds(line.get("created_at")) == int(seconds)]


def name_disagreement(pairs) -> tuple[dict[int, str], float]:
    """(step_type, stated name) pairs -> (dominant name per type, disagreement rate)."""
    by_type: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for step_type, name in pairs:
        by_type[step_type][name] += 1
    dominant = {t: c.most_common(1)[0][0] for t, c in by_type.items()}
    total = sum(sum(c.values()) for c in by_type.values())
    off = sum(sum(c.values()) - c[dominant[t]] for t, c in by_type.items())
    return dominant, (off / total if total else 0.0)


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
        names = []
        with_lines = without_their_line = rewritten = last_is_current = 0
        for db in dbs:
            lines = _transcript(db)
            if not lines:
                continue
            conn = _ro(db)
            for idx, step_type, metadata in conn.execute("SELECT idx, step_type, metadata FROM steps"):
                joined = lines.get(idx) or []
                if not joined:
                    continue
                with_lines += 1
                mine = lines_at(joined, timestamp_seconds(get(metadata, 1)))
                without_their_line += not mine
                names += [(step_type, line.get("type")) for line in mine]
                if len(joined) > 1:
                    rewritten += 1
                    last_is_current += bool(mine) and mine[-1] is joined[-1]
            conn.close()
        assert len(names) > 1000, "too few joined lines to say anything"
        dominant, rate = name_disagreement(names)
        print(f"\nsteps with a transcript line {with_lines}; none at their creation second "
              f"{without_their_line}; indices written more than once {rewritten}, last line current "
              f"{last_is_current}; name disagreement {rate:.4%}; names {dominant}")
        assert rate <= MAX_NAME_DISAGREEMENT, f"{rate:.4%} of lines disagree with their type's name"
        assert len(set(dominant.values())) == len(dominant), f"two step types share a name: {dominant}"
        assert without_their_line / with_lines <= MAX_STEPS_WITHOUT_THEIR_LINE
        if rewritten:
            assert last_is_current / rewritten >= MIN_LAST_LINE_IS_CURRENT

    def test_oracle(self):
        pairs = [(15, "PLANNER_RESPONSE")] * 99 + [(15, "GENERIC")]
        dominant, rate = name_disagreement(pairs)
        assert dominant == {15: "PLANNER_RESPONSE"} and rate == pytest.approx(0.01)
        old = {"type": "GENERIC", "created_at": "2026-09-21T14:13:20Z"}       # the abandoned branch
        new = {"type": "USER_INPUT", "created_at": "2026-09-21T14:43:20Z"}    # what the database holds
        assert lines_at([old, new], 1_790_001_800.9) == [new]
        assert lines_at([old, new], 1_790_000_000) == [old]
        assert lines_at([old, new], 5.0) == [] and lines_at([old, {"type": "X"}], None) == []


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


# ---------------------------------------------------------------------------
# Claims 17 and 18: what each step field holds, read by number
# ---------------------------------------------------------------------------

# Every reading the shaped layer makes (parsers/antigravity/shaped.py) against
# what the brain transcript states for the same step, joined as in claim 6.
# A sample of the databases: the readings are per-field facts, not rare
# events, and the full scan (scripts under internal/antigravity_audit_2026-11-20)
# agreed with the sample. Equal where the transcript states the value as is;
# contained where the transcript wraps it. Run with -s for the counts.
READING_SAMPLE_EVERY = 12
MIN_EQUAL = 0.999          # type, source, response text, thinking, token counts
MIN_CONTAINED = 0.98       # user, system and checkpoint text inside the line's content
MIN_TOOL_OUTPUT = 0.90     # the transcript truncates some tool output (claim 6)
MIN_STATUS = 0.95          # a line can be written before the step's last status


def reading_comparisons(step: dict, line: dict) -> list[tuple[str, bool]]:
    """(reading, agrees) for one decoded step and the transcript line written for it."""
    from ccutils.parsers.antigravity import shaped
    out = [
        ("type", shaped.STEP_TYPE_NAMES.get(step["step_type"]) == line.get("type")),
        ("source", shaped.SOURCE_NAMES.get(step["source"]) == line.get("source")),
    ]
    cleared = step["status"] == shaped.STATUS_CLEARED
    if not cleared:
        out.append(("status", shaped.STATUS_NAMES.get(step["status"]) == line.get("status")))
    content = line.get("content") or ""
    if step["step_type"] == shaped.PLANNER_RESPONSE and not cleared:
        if step["text"] is not None or line.get("content"):
            out.append(("response text", step["text"] == line.get("content")))
        if step["thinking"] is not None or line.get("thinking"):
            out.append(("thinking", step["thinking"] == line.get("thinking")))
    if step["step_type"] == shaped.USER_INPUT and step["text"] is not None:
        out.append(("wrapped text", step["text"] in content))
    if step["step_type"] in (shaped.SYSTEM_MESSAGE, shaped.CHECKPOINT) and step["text"] is not None and not cleared:
        out.append(("wrapped text", step["text"] in content))
    if step["step_type"] == shaped.ERROR_MESSAGE and step["text"] is not None and not cleared and line.get("error"):
        out.append(("wrapped text", step["text"] in line["error"]))
    if step["step_type"] == shaped.GENERIC and step["tool_output"] is not None:
        out.append(("tool output", step["tool_output"] in content))
    for name in ("input_tokens", "output_tokens", "cache_read_tokens"):
        if name in line and step[name] is not None:
            out.append(("tokens", step[name] == line[name]))
    return out


class TestStepFieldReadings:
    def test_corpus_holds(self, dbs):
        from ccutils.parsers.antigravity import shaped
        sample = sorted(dbs, key=lambda p: p.stat().st_size)[::READING_SAMPLE_EVERY]
        seen: collections.Counter = collections.Counter()
        agree: collections.Counter = collections.Counter()
        unnamed: collections.Counter = collections.Counter()
        steps = unparsed = tool_steps = tool_steps_named = 0
        for db in sample:
            lines = _transcript(db)
            if not lines:
                continue
            conn = _ro(db)
            for idx, payload in conn.execute("SELECT idx, step_payload FROM steps"):
                steps += 1
                try:
                    step = shaped.read_step(payload)
                except ValueError:
                    unparsed += 1
                    continue
                if step["step_type"] == shaped.GENERIC:
                    tool_steps += 1
                    tool_steps_named += bool(step["tool_call_id"] and step["tool_name"])
                mine = lines_at(lines.get(idx) or [], shaped.created_second(step["created_at"]))
                if not mine:
                    continue
                for name, ok in reading_comparisons(step, mine[-1]):
                    seen[name] += 1
                    agree[name] += ok
                    if name == "type" and not ok:
                        unnamed[(step["step_type"], mine[-1].get("type"))] += 1
            conn.close()
        print(f"\n{len(sample)} databases, {steps} steps, {unparsed} unparsed; tool steps with id and name "
              f"{tool_steps_named} of {tool_steps}; " + ", ".join(f"{k} {agree[k]}/{seen[k]}" for k in sorted(seen))
              + f"; (step type, the transcript's name) where the name map disagrees: {dict(unnamed)}")
        assert steps > 1000 and unparsed == 0
        assert tool_steps and tool_steps_named == tool_steps
        floors = {"type": MIN_EQUAL, "source": MIN_EQUAL, "response text": MIN_EQUAL, "thinking": MIN_EQUAL,
                  "tokens": MIN_EQUAL, "wrapped text": MIN_CONTAINED, "tool output": MIN_TOOL_OUTPUT,
                  "status": MIN_STATUS}
        for name, floor in floors.items():
            assert seen[name], f"nothing to compare for {name}"
            assert agree[name] / seen[name] >= floor, f"{name}: {agree[name]} of {seen[name]} agree"

    def test_oracle(self):
        from ccutils.parsers.antigravity import shaped
        step = dict.fromkeys(("text", "thinking", "tool_output", "input_tokens", "output_tokens",
                              "cache_read_tokens"), None)
        step.update(step_type=shaped.PLANNER_RESPONSE, status=3, source=2, text="hello", input_tokens=10)
        line = {"type": "PLANNER_RESPONSE", "source": "MODEL", "status": "DONE", "content": "hello",
                "input_tokens": 11}
        assert dict(reading_comparisons(step, line)) == {
            "type": True, "source": True, "status": True, "response text": True, "tokens": False}
        # A cleared step is not compared on what it no longer holds.
        step.update(status=shaped.STATUS_CLEARED, text=None, input_tokens=None)
        assert dict(reading_comparisons(step, line)) == {"type": True, "source": True}
        tool = dict(step, step_type=shaped.GENERIC, status=3, tool_output="out")
        assert ("tool output", True) in reading_comparisons(tool, dict(line, type="GENERIC", content="Output:\nout\n"))
        assert ("tool output", False) in reading_comparisons(tool, dict(line, type="GENERIC", content="cut"))


# ---------------------------------------------------------------------------
# Claim 19: a background task's number is the index of the step that started it.
# ---------------------------------------------------------------------------

MIN_TASK_LAUNCH = 0.99   # a revert leaves a log whose index now holds another step


def task_launch_rate(tools: collections.Counter, task_tools: tuple[str, ...]) -> float:
    """The share of task logs whose step is a call of a tool that starts tasks."""
    total = sum(tools.values())
    return sum(tools[t] for t in task_tools) / total if total else 0.0


class TestTaskIsNamedAfterItsStep:
    def test_corpus_holds(self, dbs):
        import re
        from ccutils.parsers.antigravity import shaped
        tools: collections.Counter = collections.Counter()
        for db in dbs:
            tasks = db.parent.parent / "brain" / db.stem / ".system_generated" / "tasks"
            if not tasks.is_dir():
                continue
            indices = sorted(int(m.group(1)) for f in tasks.iterdir()
                             if (m := re.fullmatch(r"task-(\d+)\.log", f.name)))
            if not indices:
                continue
            conn = _ro(db)
            payloads = dict(conn.execute(
                f"SELECT idx, step_payload FROM steps WHERE idx IN ({', '.join(map(str, indices))})"))
            conn.close()
            for index in indices:
                payload = payloads.get(index)
                tools[shaped.read_step(payload)["tool_name"] if payload else "(no step at that index)"] += 1
        if not tools:
            pytest.skip("no background task logs in this store")
        rate = task_launch_rate(tools, shaped.TASK_TOOLS)
        print(f"\n{sum(tools.values())} task logs; the step at their number calls: {dict(tools.most_common())}")
        assert rate >= MIN_TASK_LAUNCH, f"only {rate:.2%} of task logs sit at a step that starts tasks: {dict(tools)}"

    def test_oracle(self):
        tools = collections.Counter({"run_command": 97, "schedule": 2, None: 1})
        assert task_launch_rate(tools, ("run_command", "schedule")) == pytest.approx(0.99)
        assert task_launch_rate(tools, ("run_command",)) == pytest.approx(0.97)
        assert task_launch_rate(collections.Counter(), ("run_command",)) == 0.0


# ---------------------------------------------------------------------------
# Claim 20: the metadata blob states each workspace with its repository.
# ---------------------------------------------------------------------------

MIN_FIRST_WORKSPACE_IS_THE_INDEX_ROWS = 0.99
MIN_SLUG_IS_THE_REMOTES_TAIL = 0.95   # a remote can have been renamed since the slug was taken


def slug_is_remote_tail(slug: str, remote: str) -> bool:
    """`owner/repo` is how the remote URL ends, whatever the host and scheme."""
    tail = remote.rstrip("/")
    tail = tail[:-4] if tail.endswith(".git") else tail
    return tail.lower().endswith(("/" + slug.lower(), ":" + slug.lower()))


class TestWorkspacesAreStated:
    def test_corpus_holds(self, dbs):
        from ccutils.parsers.antigravity import shaped
        index: dict[str, str | None] = {}
        for store in DB_STORES:
            path = ROOT / store / "conversation_summaries.db"
            if not path.exists():
                continue
            conn = _ro(path)
            for conv, uris in conn.execute("SELECT conversation_id, workspace_uris FROM conversation_summaries"):
                try:
                    stated = json.loads(uris) if uris else []
                except json.JSONDecodeError:
                    stated = []
                index[conv] = stated[0] if stated else None
            conn.close()
        blobs = unparsed = with_workspace = compared = same = with_remote = tail = 0
        for db in dbs:
            conn = _ro(db)
            rows = conn.execute("SELECT data FROM trajectory_metadata_blob").fetchall()
            conn.close()
            for (data,) in rows:
                if not data:
                    continue
                blobs += 1
                try:
                    workspaces = shaped.read_workspaces(data)
                except ValueError:
                    unparsed += 1
                    continue
                if not workspaces:
                    continue
                with_workspace += 1
                if index.get(db.stem):
                    compared += 1
                    same += workspaces[0]["workspace_uri"] == index[db.stem]
                for w in workspaces:
                    if w["repo_slug"] and w["git_remote"]:
                        with_remote += 1
                        tail += slug_is_remote_tail(w["repo_slug"], w["git_remote"])
        print(f"\n{blobs} blobs, {unparsed} unparsed, {with_workspace} state a workspace; first workspace is the "
              f"index row's {same}/{compared}; slug is the remote's tail {tail}/{with_remote}")
        assert blobs and unparsed == 0
        assert compared and same / compared >= MIN_FIRST_WORKSPACE_IS_THE_INDEX_ROWS
        assert with_remote and tail / with_remote >= MIN_SLUG_IS_THE_REMOTES_TAIL

    def test_oracle(self):
        assert slug_is_remote_tail("handle/My-Repo", "git@github.com:handle/my-repo.git")
        assert slug_is_remote_tail("user/repo", "ssh://git@10.0.0.5:2222/user/repo")
        assert slug_is_remote_tail("user/repo", "https://example.org/user/repo/")
        assert not slug_is_remote_tail("user/repo", "git@github.com:someone/user-repo.git")
        assert not slug_is_remote_tail("repo", "git@github.com:user/other.git")

