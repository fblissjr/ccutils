"""Lake table shapes for Antigravity, and the format version that guards them.

Every lake table is the envelope (`ccutils.parsers.lake.envelope_fields`),
then `conversation_id` for tables inside a conversation unit, then the source
columns exactly as the SQLite file declares them, in declaration order.
Nothing is decoded or promoted at Tier 1; that is Phase 1's job
(docs/HARNESS_ARCHITECTURE.md).

The expected columns below are copied from the real `sqlite_master`, and
their storage types were measured with `typeof()` over every row of every
file (docs/ANTIGRAVITY_CONTRACT.md, claims 1 and 4). SQLite does not enforce
declared types, so the writer checks every value against these and fails the
unit on a mismatch rather than coercing it into a plausible column.
"""

from __future__ import annotations

import hashlib

import pyarrow as pa

from ccutils.parsers.lake import envelope_fields
from ccutils.parsers.antigravity import stores

# Bump when the writer's output changes: a table shape, a classification
# rule, the allowlist, the store list. Pinned to format_fingerprint() by
# tests/test_antigravity_lake.py, so a change without a bump fails.
#
# 2: `source_present` in the envelope; rows and tables the source lost are
#    carried forward (CARRY_KEYS); reverts supersede (IDX_TABLES, claim 14);
#    brain/<id>/.git archived as the `brain_git` unit.
#
# 3: a column the contract does not declare takes its kind from the data
#    (`typeof`), not from its declared type; one holding several storage
#    classes is written as bytes plus a `<name>__storage` column (MIXED).
LAKE_FORMAT_VERSION = 3

TEXT, INTEGER, BLOB, REAL = "text", "integer", "blob", "real"

ARROW_TYPES = {TEXT: pa.string(), INTEGER: pa.int64(), BLOB: pa.large_binary(), REAL: pa.float64()}
PYTHON_TYPES = {TEXT: str, INTEGER: int, BLOB: bytes, REAL: float}

# SQLite types values, not columns: one column can hold a blob in one row and
# text in the next (the hub's full-text index does). Such a column cannot be
# one Arrow type, so it is written as two columns: the value's bytes under
# its own name, and the storage class each row had under this suffix.
MIXED = "mixed"
MIXED_STORAGE_SUFFIX = "__storage"


def encode_mixed(value) -> tuple[bytes | None, str | None]:
    """(bytes, storage class) for one value of a MIXED column."""
    if value is None:
        return None, None
    if isinstance(value, bytes):
        return value, BLOB
    if isinstance(value, str):
        return value.encode("utf-8"), TEXT
    if isinstance(value, int):
        return str(value).encode("ascii"), INTEGER
    if isinstance(value, float):
        return repr(value).encode("ascii"), REAL
    raise TypeError(f"not a SQLite storage class: {type(value).__name__}")


def decode_mixed(data: bytes | None, storage: str | None):
    """The value `encode_mixed` was given."""
    if storage is None:
        return None
    if storage == BLOB:
        return data
    if storage == TEXT:
        return data.decode("utf-8")
    if storage == INTEGER:
        return int(data)
    if storage == REAL:
        return float(data)
    raise ValueError(f"unknown storage class {storage!r}")

CONVERSATION_TABLES: dict[str, tuple[tuple[str, str], ...]] = {
    "trajectory_meta": (
        ("trajectory_id", TEXT), ("cascade_id", TEXT), ("trajectory_type", INTEGER), ("source", INTEGER),
    ),
    "steps": (
        ("idx", INTEGER), ("step_type", INTEGER), ("status", INTEGER), ("has_subtrajectory", INTEGER),
        ("metadata", BLOB), ("error_details", BLOB), ("permissions", BLOB), ("task_details", BLOB),
        ("render_info", BLOB), ("step_payload", BLOB), ("step_format", INTEGER),
    ),
    "gen_metadata": (("idx", INTEGER), ("data", BLOB), ("size", INTEGER)),
    "executor_metadata": (("idx", INTEGER), ("data", BLOB)),
    "parent_references": (("idx", INTEGER), ("data", BLOB)),
    "trajectory_metadata_blob": (("id", TEXT), ("data", BLOB)),
    "battle_mode_infos": (("idx", INTEGER), ("data", BLOB)),
}

SUMMARIES_TABLES: dict[str, tuple[tuple[str, str], ...]] = {
    "conversation_summaries": (
        ("conversation_id", TEXT), ("title", TEXT), ("preview", TEXT), ("step_count", INTEGER),
        ("last_modified_time", TEXT), ("workspace_uris", TEXT), ("status", TEXT), ("source", TEXT),
        ("project_id", TEXT), ("agent_name", TEXT), ("parent_conversation_id", TEXT),
        ("nesting_depth", INTEGER), ("battle_id", TEXT), ("winning_conversation_id", TEXT),
        ("not_fully_idle", INTEGER), ("killed", INTEGER), ("last_user_input_time", TEXT),
        ("last_user_input_step_index", INTEGER), ("app_data_dir", TEXT), ("raw_summary", BLOB),
        ("group_id", TEXT),
    ),
}

# Per unit kind, (table, key) pairs whose rows are carried forward when the
# source drops them. Each key is unique within its table: a summaries row per
# conversation, a files row per path.
CARRY_KEYS: dict[str, tuple[tuple[str, str], ...]] = {
    "conversation": (("files", "relpath"),),
    "brain_git": (("git_files", "relpath"),),
    "summaries": (("conversation_summaries", "conversation_id"),),
    "store_files": (("store_files", "relpath"),),
    "implicit": (("implicit_files", "relpath"),),
    "config_projects": (("config_projects", "relpath"),),
}

# Conversation tables keyed on `idx`. A snapshot that lost an idx from any of
# them is not a continuation of the old one and supersedes it.
IDX_TABLES: tuple[str, ...] = (
    "steps", "gen_metadata", "executor_metadata", "parent_references", "battle_mode_infos",
)

# Step metadata field 1 is the step's creation time (claim 6). A step whose
# creation time changed under the same idx was re-created by a revert (claim 14).
STEP_CREATED_AT_FIELD = 1

FILE_FIELDS: tuple[pa.Field, ...] = (
    pa.field("relpath", pa.string(), nullable=False),
    pa.field("kind", pa.string(), nullable=False),
    pa.field("size_bytes", pa.int64()),
    pa.field("mtime", pa.timestamp("ns", tz="UTC")),
    pa.field("sha256", pa.string()),
    pa.field("content", pa.large_binary()),
)


def storage_kind(declared: str) -> str:
    """Storage kind for an undeclared column that holds no value to read it from.

    The writer reads the kind of a column the contract does not know from the
    data (`typeof`); this is the fallback for an empty or all-NULL column,
    where the declaration is all there is. Follows SQLite's affinity rules, except that NUMERIC-affinity names map to
    INTEGER (every NUMERIC column measured holds integers) and dates to TEXT
    (gorm writes them as strings)."""
    d = declared.upper()
    if "INT" in d:
        return INTEGER
    if any(k in d for k in ("CHAR", "CLOB", "TEXT", "DATE", "TIME")):
        return TEXT
    if "BLOB" in d or not d:
        return BLOB
    if any(k in d for k in ("REAL", "FLOA", "DOUB")):
        return REAL
    return INTEGER


def table_schema(columns: list[tuple[str, str]], *, with_conversation_id: bool) -> pa.Schema:
    fields = list(envelope_fields())
    if with_conversation_id:
        fields.append(pa.field("conversation_id", pa.string(), nullable=False))
    for name, kind in columns:
        if kind == MIXED:
            fields.append(pa.field(name, pa.large_binary()))
            fields.append(pa.field(name + MIXED_STORAGE_SUFFIX, pa.string()))
        else:
            fields.append(pa.field(name, ARROW_TYPES[kind]))
    return pa.schema(fields)


def files_schema() -> pa.Schema:
    return pa.schema(list(envelope_fields()) + [pa.field("conversation_id", pa.string())] + list(FILE_FIELDS))


def envelope_fields_list():  # re-exported for tests that build expected column lists
    return envelope_fields()


def format_fingerprint() -> str:
    """Hash of everything that decides what the writer emits.

    Read at call time from the stores module so a changed rule changes it."""
    parts = [
        repr(CONVERSATION_TABLES), repr(SUMMARIES_TABLES),
        repr([str(f) for f in FILE_FIELDS]), repr([str(f) for f in envelope_fields()]),
        repr(stores.KNOWN_STORES), repr(stores.BACKUP_STORES),
        repr(stores.STORE_FILE_PATTERNS), repr(stores.GLOBAL_FILE_PATTERNS),
        repr(stores.STORE_FILE_KINDS), repr(stores.EXCLUDED_DIRS),
        repr(stores.CONVERSATION_FILE_RULES), repr(stores.UNCLASSIFIED_MAX_BYTES),
        repr(stores.GIT_DIR), repr(stores.GIT_FILE_RULES),
        repr(CARRY_KEYS), repr(IDX_TABLES), repr(STEP_CREATED_AT_FIELD),
        repr(("undeclared-column-kind", "typeof", MIXED, MIXED_STORAGE_SUFFIX)),
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()
