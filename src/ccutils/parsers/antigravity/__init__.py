"""Antigravity (Google's Gemini agentic IDE/hub and its `agy` CLI) as a lake source.

`ccutils lake antigravity` mirrors the native store under ``<HOME>/.gemini``
into ``<lake_root>/antigravity/`` through the generic runner in
`ccutils.parsers.lake`. Units:

- ``conversation`` (per store, per conversation uuid): every table of
  ``conversations/<id>.db`` plus ``files.parquet`` for the legacy encrypted
  ``conversations/<id>.pb``, ``brain/<id>/**`` (except ``.git``) and
  ``browser_recordings/<id>/**``.
- ``brain_git`` (per store, per conversation uuid): ``brain/<id>/.git/**``,
  the snapshot history, as bytes.
- ``summaries`` (per store): ``conversation_summaries.db``.
- ``store_files`` (per store): annotations, the CLI prompt history and the
  summaries protobuf.
- ``implicit`` (per store): ``implicit/*.pb`` ciphertext.
- ``config_projects`` (global): ``config/projects/*.json``, which names the
  project ids conversations carry.

The store map and every measured claim behind these choices are in
docs/ANTIGRAVITY_CONTRACT.md.

Archive rules on top of the runner's: rows the source drops are carried
forward per `schema.CARRY_KEYS`, and a conversation snapshot supersedes the
old one when it lost an idx or re-created a step (`supersedes`).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

import pyarrow.parquet as pq

from ccutils.parsers.lake import Discovery, Envelope, LakeUnit, UnitWrite
from ccutils.parsers.antigravity import schema, stores, wire
from ccutils.parsers.antigravity.writer import (
    FileSpec,
    LakeWriteError,
    mirror_sqlite,
    snapshot_sqlite,
    write_files,
)

GLOBAL_STORE = "_global"

__all__ = ["AntigravitySource", "LakeWriteError", "GLOBAL_STORE"]


class AntigravitySource:
    harness_id = "antigravity"
    lake_format_version = schema.LAKE_FORMAT_VERSION
    record_sources = frozenset({
        "antigravity_conversation_db",
        "antigravity_summaries_db",
        "antigravity_conversation_file",
        "antigravity_store_file",
        "antigravity_encrypted",
        "antigravity_config_projects",
    })

    def __init__(self, *, stores: Sequence[str] | None = None, app_bundle: Path | None = stores.HUB_INFO_PLIST):
        self.only = tuple(stores) if stores is not None else None
        self.app_bundle = app_bundle

    # -- discovery ---------------------------------------------------------

    def discover(self, root: Path | None) -> Discovery:
        root = Path(root) if root is not None else stores.default_root()
        found, notes = stores.discover_stores(root, self.only)
        units: list[LakeUnit] = []
        for store in found:
            units += self._store_units(root / store, store, notes)
        projects = sorted(
            p.relative_to(root).as_posix()
            for p in (root / "config" / "projects").glob("*.json")
            if p.is_file() and not p.is_symlink()
        ) if (root / "config" / "projects").is_dir() else []
        if projects:
            units.append(LakeUnit(
                store=GLOBAL_STORE, unit_kind="config_projects", unit_id="config_projects",
                source_root=root, source_relpaths=tuple(projects),
                out_relpath=f"{GLOBAL_STORE}/config_projects",
                carry_keys=schema.CARRY_KEYS["config_projects"],
            ))
        context = {"data_root": str(root), "stores": ",".join(found)}
        version = stores.app_version(self.app_bundle)
        if version:
            context["app_version"] = version
        return Discovery(units=units, notes=notes, context=context,
                         scanned_stores=frozenset(found) | {GLOBAL_STORE})

    def _store_units(self, store_root: Path, store: str, notes: list[str]) -> list[LakeUnit]:
        units: list[LakeUnit] = []
        conv_files: dict[str, list[str]] = {}

        conv_dir = store_root / "conversations"
        if conv_dir.is_dir():
            for name in sorted(p.name for p in conv_dir.iterdir()):
                stem, dot, ext = name.partition(".")
                if not stores.is_uuid(stem):
                    notes.append(f"{store}/conversations/{name}: not a conversation file; skipped")
                    continue
                if ext in ("db", "pb", "db-wal"):
                    conv_files.setdefault(stem, []).append(f"conversations/{name}")
                elif ext not in ("db-shm", "db-journal"):
                    notes.append(f"{store}/conversations/{name}: unexpected extension; skipped")

        skipped_dirs: list[str] = []
        git_units: list[LakeUnit] = []
        for area in ("brain", "browser_recordings"):
            area_dir = store_root / area
            if not area_dir.is_dir():
                continue
            for d in sorted(p for p in area_dir.iterdir() if p.is_dir() and not p.is_symlink()):
                if not stores.is_uuid(d.name):
                    skipped_dirs.append(f"{area}/{d.name}")
                    continue
                conv_files.setdefault(d.name, []).extend(stores.walk_conversation_dir(store_root, area, d.name))
                git = d / stores.GIT_DIR
                if area == "brain" and git.is_dir() and not git.is_symlink():
                    git_rels = stores.walk_git_dir(store_root, d.name)
                    if git_rels:
                        git_units.append(LakeUnit(
                            store=store, unit_kind="brain_git", unit_id=d.name,
                            source_root=store_root, source_relpaths=tuple(git_rels),
                            out_relpath=f"{store}/brain_git/{d.name}",
                            carry_keys=schema.CARRY_KEYS["brain_git"],
                        ))
        if skipped_dirs:
            notes.append(f"{store}: skipped non-conversation dirs {', '.join(skipped_dirs)}")

        for conv_id in sorted(conv_files):
            rels = tuple(sorted(set(conv_files[conv_id])))
            units.append(LakeUnit(
                store=store, unit_kind="conversation", unit_id=conv_id,
                source_root=store_root, source_relpaths=rels,
                out_relpath=f"{store}/conversations/{conv_id}",
                carry_keys=schema.CARRY_KEYS["conversation"],
            ))
        units += git_units

        summaries = store_root / "conversation_summaries.db"
        if summaries.is_file():
            rels = ["conversation_summaries.db"]
            if summaries.with_name("conversation_summaries.db-wal").exists():
                rels.append("conversation_summaries.db-wal")
            units.append(LakeUnit(
                store=store, unit_kind="summaries", unit_id="conversation_summaries",
                source_root=store_root, source_relpaths=tuple(rels), out_relpath=f"{store}/summaries",
                carry_keys=schema.CARRY_KEYS["summaries"],
            ))

        store_level = []
        for p in sorted(store_root.glob("*")) + sorted((store_root / "annotations").glob("*")):
            rel = p.relative_to(store_root).as_posix()
            if p.is_file() and not p.is_symlink() and stores.store_file_kind(rel):
                store_level.append(rel)
        if store_level:
            units.append(LakeUnit(
                store=store, unit_kind="store_files", unit_id="store_files",
                source_root=store_root, source_relpaths=tuple(store_level),
                out_relpath=f"{store}/store_files",
                carry_keys=schema.CARRY_KEYS["store_files"],
            ))

        implicit = sorted(
            f"implicit/{p.name}" for p in (store_root / "implicit").glob("*.pb")
            if stores.is_uuid(p.stem) and p.is_file() and not p.is_symlink()
        ) if (store_root / "implicit").is_dir() else []
        if implicit:
            units.append(LakeUnit(
                store=store, unit_kind="implicit", unit_id="implicit",
                source_root=store_root, source_relpaths=tuple(implicit),
                out_relpath=f"{store}/implicit",
                carry_keys=schema.CARRY_KEYS["implicit"],
            ))
        return units

    # -- writing -----------------------------------------------------------

    def write_unit(self, unit: LakeUnit, out_dir: Path, envelope: Envelope) -> UnitWrite:
        if unit.unit_kind == "conversation":
            return self._write_conversation(unit, out_dir, envelope)
        if unit.unit_kind == "summaries":
            return self._write_sqlite(unit, out_dir, envelope, "conversation_summaries.db",
                                      schema.SUMMARIES_TABLES, "antigravity_summaries_db", None)
        if unit.unit_kind == "brain_git":
            prefix = f"brain/{unit.unit_id}/{stores.GIT_DIR}/"
            specs = [FileSpec(rel, stores.git_file_kind(rel[len(prefix):]), stores.BYTES,
                              "antigravity_conversation_file") for rel in unit.source_relpaths]
            rows = write_files(out_dir / "git_files.parquet", unit.source_root, specs, envelope, unit.unit_id)
            return UnitWrite(tables={"git_files": rows})
        specs_by_kind = {
            "store_files": ("store_files", "antigravity_store_file", lambda rel: stores.store_file_kind(rel)),
            "implicit": ("implicit_files", "antigravity_encrypted", lambda rel: "implicit_pb"),
            "config_projects": ("config_projects", "antigravity_config_projects", lambda rel: "config_project"),
        }
        table, record_source, kind_of = specs_by_kind[unit.unit_kind]
        specs = [FileSpec(rel, kind_of(rel) or "unknown", stores.BYTES, record_source) for rel in unit.source_relpaths]
        rows = write_files(out_dir / f"{table}.parquet", unit.source_root, specs, envelope, None)
        return UnitWrite(tables={table: rows})

    def _write_sqlite(self, unit, out_dir, envelope, rel, expected, record_source, conversation_id):
        scratch = out_dir / ".snapshot"
        try:
            copy, method = snapshot_sqlite(unit.source_root / rel, scratch)
            tables, notes = mirror_sqlite(copy, out_dir, expected, envelope, record_source, rel, conversation_id)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        return UnitWrite(tables=tables, notes=notes, snapshot=method)

    def _write_conversation(self, unit: LakeUnit, out_dir: Path, envelope: Envelope) -> UnitWrite:
        conv_id = unit.unit_id
        db_rel = f"conversations/{conv_id}.db"
        if db_rel in unit.source_relpaths:
            written = self._write_sqlite(unit, out_dir, envelope, db_rel, schema.CONVERSATION_TABLES,
                                         "antigravity_conversation_db", conv_id)
        else:
            written = UnitWrite(tables={})
        specs = []
        for rel in unit.source_relpaths:
            if rel.startswith("conversations/"):
                if rel.endswith(".pb"):
                    specs.append(FileSpec(rel, "legacy_pb", stores.BYTES, "antigravity_encrypted"))
                continue  # the .db and its -wal are mirrored as tables above
            area, _, rest = rel.partition("/")
            within = rest.partition("/")[2]
            classified = stores.classify_conversation_file(within, area)
            kind, keep = classified if classified else ("symlink", "symlink")
            specs.append(FileSpec(rel, kind, keep, "antigravity_conversation_file"))
        written.tables["files"] = write_files(out_dir / "files.parquet", unit.source_root, specs, envelope, conv_id)
        return written

    # -- continuity --------------------------------------------------------

    def supersedes(self, unit: LakeUnit, old_dir: Path, new_dir: Path) -> str | None:
        """Why a new conversation snapshot must not overwrite the old one.

        Two ways a conversation stops being a continuation: an idx the old
        snapshot held is gone from a table that still exists, or a step under
        the same idx was re-created (its creation time changed) -- a revert,
        which re-uses indices and so leaves the idx set whole (claim 14). A
        table missing from the new snapshot entirely is not a reason: the
        runner carries it forward whole."""
        if unit.unit_kind != "conversation":
            return None
        for table in schema.IDX_TABLES:
            old, new = old_dir / f"{table}.parquet", new_dir / f"{table}.parquet"
            if not (old.exists() and new.exists()):
                continue
            lost = _idx(old) - _idx(new)
            if lost:
                return f"{table}: idx {_sample(lost)} gone from the source"
        old, new = old_dir / "steps.parquet", new_dir / "steps.parquet"
        if old.exists() and new.exists():
            before, after = _created_at(old), _created_at(new)
            recreated = [i for i in before.keys() & after.keys()
                         if before[i] is not None and after[i] is not None and before[i] != after[i]]
            if recreated:
                return f"steps: idx {_sample(recreated)} re-created (a revert re-used them)"
        return None


def _idx(path: Path) -> set[int]:
    return set(pq.read_table(path, columns=["idx"]).column(0).to_pylist())


def _created_at(path: Path) -> dict[int, bytes | None]:
    """idx -> raw bytes of Step metadata's creation time, for current rows only.

    Rows carried forward from an earlier snapshot are left out: they describe
    what the source used to hold, not what it holds now."""
    t = pq.read_table(path, columns=["idx", "metadata", "source_present"])
    out: dict[int, bytes | None] = {}
    for idx, meta, present in zip(*(t.column(c).to_pylist() for c in ("idx", "metadata", "source_present"))):
        if present is False:
            continue
        try:
            out[idx] = wire.field_bytes(meta, schema.STEP_CREATED_AT_FIELD) if meta else None
        except ValueError:
            out[idx] = None
    return out


def _sample(values) -> str:
    ordered = sorted(values)
    return ", ".join(map(str, ordered[:5])) + (f" (+{len(ordered) - 5} more)" if len(ordered) > 5 else "")
