"""DDL migration runner backed by the `meta_schema_version` table.

Why this exists separately from `dim_etl_version`:
- `dim_etl_version` tracks ETL business-rules versions (data semantics).
- `meta_schema_version` tracks DDL-level migrations (table shapes).

Both must be tracked. A migration that adds a column doesn't change the
ETL business rules. A business-rule change that re-derives a column
doesn't change the schema. Conflating them obscures both signals.

Migration files live in `src/ccutils/schemas/migrations/m_<id>.py`.
Each file exports a module-level `MIGRATION` instance. The runner is
deliberately simple: read applied ids, apply pending ones in id order,
record each as it succeeds. No down migrations -- if a change needs to
be undone, write a new forward migration.

Usage:
    from ccutils.schemas.migrations import apply_pending_migrations, all_migrations
    apply_pending_migrations(conn, all_migrations())
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable


@dataclass
class Migration:
    """One DDL migration.

    id: stable, sortable identifier (YYYYMMDD_NNNN_short_name format).
    description: one-line summary stored in meta_schema_version.
    up: function that takes a DuckDB conn and applies the change.
    """

    id: str
    description: str
    up: Callable[..., None]


def applied_migration_ids(conn) -> set[str]:
    """Return the set of migration ids already applied to this database."""
    rows = conn.execute(
        "SELECT migration_id FROM meta_schema_version"
    ).fetchall()
    return {r[0] for r in rows}


def _record_migration(conn, migration: Migration, ccutils_version: str | None) -> None:
    conn.execute(
        """
        INSERT INTO meta_schema_version (migration_id, description, ccutils_version)
        VALUES (?, ?, ?)
        """,
        [migration.id, migration.description, ccutils_version],
    )


def apply_pending_migrations(
    conn,
    migrations: list[Migration],
    *,
    ccutils_version: str | None = None,
) -> list[str]:
    """Apply migrations not yet recorded in meta_schema_version, in id order.

    Returns the list of migration ids actually applied (empty if all were
    already applied). Raises if a migration's up() raises; the failing
    migration is NOT recorded so a retry will re-attempt it.
    """
    from ccutils.etl.lineage import PARSER_VERSION
    if ccutils_version is None:
        ccutils_version = PARSER_VERSION

    already = applied_migration_ids(conn)
    pending = sorted(
        (m for m in migrations if m.id not in already),
        key=lambda m: m.id,
    )
    applied: list[str] = []
    for m in pending:
        m.up(conn)
        _record_migration(conn, m, ccutils_version)
        applied.append(m.id)
    return applied


def all_migrations() -> list[Migration]:
    """Discover every Migration defined in this package.

    Imports every `m_*.py` sibling module and collects its module-level
    `MIGRATION` attribute. New migrations become live by dropping the file
    into this directory -- no explicit registry to update.
    """
    discovered: list[Migration] = []
    for info in pkgutil.iter_modules(__path__, prefix=__name__ + "."):
        if not info.name.rsplit(".", 1)[-1].startswith("m_"):
            continue
        module = importlib.import_module(info.name)
        migration = getattr(module, "MIGRATION", None)
        if isinstance(migration, Migration):
            discovered.append(migration)
    return discovered
