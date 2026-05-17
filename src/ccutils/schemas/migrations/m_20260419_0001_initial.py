"""Baseline migration -- records that this database has the v0.15 lineage
+ meta tables (`dim_etl_version`, `fact_etl_runs`, `meta_schema_version`).

The DDL itself is created by `create_star_schema()` in `schemas/star/schema.py`.
This migration's job is just to record "this database has the baseline shape"
so future migrations have something to anchor against.
"""

from ccutils.schemas.migrations import Migration


def _up(_conn) -> None:
    pass  # No-op: schema is already created by create_star_schema().


MIGRATION = Migration(
    id="20260419_0001_initial",
    description="v0.15 baseline: lineage + meta tables in place",
    up=_up,
)
