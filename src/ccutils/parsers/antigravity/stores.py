# path-privacy: skip-file -- references universal Antigravity install and data paths (not personal)
"""Where Antigravity keeps things, and what the lake may read.

Everything here is declarative on purpose: the store names, the read
allowlist and the file classification rules are hashed into the lake format
fingerprint (schema.format_fingerprint), so changing any of them without
bumping LAKE_FORMAT_VERSION fails a test.

The data root is ``<HOME>/.gemini``. It also holds OAuth tokens
(``oauth_creds.json``, ``jetski-standalone-oauth-token``), a browser profile and
MCP config with credentials, which is why reads are an allowlist that fails
closed rather than a glob with exclusions. See docs/ANTIGRAVITY_CONTRACT.md
for the measurements behind each rule.
"""

from __future__ import annotations

import os
import plistlib
import re
from pathlib import Path

UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_UUID_RE = re.compile(rf"^{UUID}$")

# Stores are the hub/IDE (`antigravity`), the agy CLI (`antigravity-cli`) and
# the IDE app's own data dir (`antigravity-ide`, today a migration snapshot).
# `antigravity-backup` is the copy the hub names IDE_BACKUP_DATA_DIR in its
# own paths.js. Any other store-shaped directory is reported, never ingested:
# guessing that a new directory is a store is how a backup gets double-counted.
KNOWN_STORES: tuple[str, ...] = ("antigravity", "antigravity-cli", "antigravity-ide")
BACKUP_STORES: tuple[str, ...] = ("antigravity-backup",)

HUB_INFO_PLIST = Path("/Applications/Antigravity.app/Contents/Info.plist")

# The read allowlist: full-match regexes over store-relative POSIX paths.
STORE_FILE_PATTERNS: tuple[str, ...] = (
    rf"conversations/{UUID}\.db",
    rf"conversations/{UUID}\.db-wal",
    rf"conversations/{UUID}\.pb",
    r"conversation_summaries\.db",
    r"conversation_summaries\.db-wal",
    rf"implicit/{UUID}\.pb",
    rf"annotations/{UUID}\.pbtxt",
    r"history\.jsonl",
    r"(agyhub|jetbox)_summaries_proto\.pb",
    rf"brain/{UUID}/.+",
    rf"browser_recordings/{UUID}/.+",
)
# Relative to the data root, outside any store.
GLOBAL_FILE_PATTERNS: tuple[str, ...] = (
    r"config/projects/[^/]+\.json",
)

# How a store-level file is labelled in store_files.parquet.
STORE_FILE_KINDS: tuple[tuple[str, str], ...] = (
    (rf"annotations/{UUID}\.pbtxt", "annotation"),
    (r"history\.jsonl", "cli_history"),
    (r"(agyhub|jetbox)_summaries_proto\.pb", "summaries_proto"),
)

# What happens to a file under brain/<id>/ or browser_recordings/<id>/.
BYTES = "bytes"            # content archived
INVENTORY = "inventory"    # path, size, mtime and sha256 only
CAPPED = "capped"          # bytes up to UNCLASSIFIED_MAX_BYTES, inventory above

UNCLASSIFIED_MAX_BYTES = 1024 * 1024

# Directories never walked, relative to the conversation dir. `.git` is the
# brain snapshot history (extracted in a later phase, not mirrored as blobs);
# `logs/chunks/` is an exact concatenation of the whole transcript files.
EXCLUDED_DIRS: dict[str, tuple[str, ...]] = {
    "brain": (r"\.git", r"\.system_generated/logs/chunks"),
    "browser_recordings": (),
}

# First full match wins. Relative to the conversation dir.
CONVERSATION_FILE_RULES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "brain": (
        (r"\.system_generated/logs/transcript_full\.jsonl", "transcript_full", BYTES),
        (r"\.system_generated/logs/transcript\.jsonl", "transcript", BYTES),
        (r"\.system_generated/messages/.+", "agent_message", BYTES),
        (r"\.system_generated/steps/\d+/.+", "step_output", BYTES),
        (r"\.system_generated/tasks/[^/]+\.log", "task_log", BYTES),
        (r"\.agents/(.+/)?agent\.md", "agent_definition", BYTES),
        (r"\.user_uploaded/.+", "user_upload", INVENTORY),
        (r"\.tempmediaStorage/.+", "media", INVENTORY),
        (r".+\.(png|jpe?g|webp|gif|mp4|webm|mov)", "media", INVENTORY),
        # Older conversations keep artifacts under artifacts/, newer at the root.
        (r"(artifacts/)?[^/]+\.md\.metadata\.json", "artifact_metadata", BYTES),
        (r"(artifacts/)?[^/]+\.md\.resolved(\.\d+)?", "artifact_resolved", BYTES),
        (r"(artifacts/)?[^/]+\.md", "artifact", BYTES),
        (r".+", "unclassified", CAPPED),
    ),
    "browser_recordings": (
        (r"metadata\.json", "recording_metadata", BYTES),
        (r".+", "recording_frame", INVENTORY),
    ),
}


def default_root() -> Path:
    return Path.home() / ".gemini"


def is_uuid(name: str) -> bool:
    return bool(_UUID_RE.match(name))


def is_allowed(root: Path, path: Path) -> bool:
    """True when the lake may open `path` (both resolved) under data root `root`."""
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        return False
    for pattern in GLOBAL_FILE_PATTERNS:
        if re.fullmatch(pattern, rel):
            return True
    store, _, rest = rel.partition("/")
    if store not in KNOWN_STORES or not rest:
        return False
    return any(re.fullmatch(p, rest) for p in STORE_FILE_PATTERNS)


def is_excluded_dir(rel_dir: str, area: str) -> bool:
    return any(re.fullmatch(p, rel_dir) for p in EXCLUDED_DIRS[area])


def classify_conversation_file(rel: str, area: str) -> tuple[str, str] | None:
    """(kind, keep) for a file relative to its conversation dir; None if excluded."""
    parts = rel.split("/")
    for i in range(1, len(parts)):
        if is_excluded_dir("/".join(parts[:i]), area):
            return None
    for pattern, kind, keep in CONVERSATION_FILE_RULES[area]:
        if re.fullmatch(pattern, rel):
            return kind, keep
    return None


def store_file_kind(rel: str) -> str | None:
    for pattern, kind in STORE_FILE_KINDS:
        if re.fullmatch(pattern, rel):
            return kind
    return None


def looks_like_store(path: Path) -> bool:
    return (path / "conversations").is_dir() or (path / "conversation_summaries.db").is_file()


def discover_stores(root: Path, only: tuple[str, ...] | None = None) -> tuple[list[str], list[str]]:
    """Known stores present under `root`, plus notes about what was skipped."""
    notes: list[str] = []
    found = [s for s in KNOWN_STORES if (root / s).is_dir() and looks_like_store(root / s)]
    if only is not None:
        unknown = [s for s in only if s not in KNOWN_STORES]
        if unknown:
            raise ValueError(f"not an Antigravity store: {', '.join(unknown)} (known: {', '.join(KNOWN_STORES)})")
        found = [s for s in found if s in only]
    try:
        children = sorted(e.name for e in os.scandir(root) if e.is_dir(follow_symlinks=False))
    except FileNotFoundError:
        return [], [f"no Antigravity data root at {root}"]
    for name in children:
        if name in KNOWN_STORES or not looks_like_store(root / name):
            continue
        if name in BACKUP_STORES:
            notes.append(f"{name}: skipped, the app's own backup copy")
        else:
            notes.append(f"{name}: looks like a store but is not a known one; not ingested")
    return found, notes


def walk_conversation_dir(store_root: Path, area: str, conv_id: str) -> list[str]:
    """Store-relative paths of every file the lake reads for one conversation.

    Excluded directories are pruned, never entered. Symlinks, to files or
    directories, are listed (so they are inventoried) but never followed."""
    base = store_root / area / conv_id
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        rel_dir = Path(dirpath).relative_to(base).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        keep = []
        for d in sorted(dirnames):
            rd = f"{rel_dir}/{d}" if rel_dir else d
            if (Path(dirpath) / d).is_symlink():
                out.append(f"{area}/{conv_id}/{rd}")
            elif not is_excluded_dir(rd, area):
                keep.append(d)
        dirnames[:] = keep
        for f in sorted(filenames):
            rf = f"{rel_dir}/{f}" if rel_dir else f
            if (Path(dirpath) / f).is_symlink() or classify_conversation_file(rf, area) is not None:
                out.append(f"{area}/{conv_id}/{rf}")
    return out


def app_version(plist: Path | None) -> str | None:
    """CFBundleShortVersionString of the installed hub, when there is one."""
    if plist is None or not plist.is_file():
        return None
    try:
        with open(plist, "rb") as f:
            return str(plistlib.load(f).get("CFBundleShortVersionString"))
    except (OSError, plistlib.InvalidFileException):
        return None
