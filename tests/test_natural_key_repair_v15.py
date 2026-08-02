"""An existing warehouse must heal itself of duplicate natural keys.

Claim: `lineage_upsert` now RAISES on a duplicate natural key, which stops
new duplicates being written but does nothing about rows already in a
warehouse built before that change. Those rows are not inert -- several
populators build their inbound table by reading a TARGET fact table, so a
pre-existing duplicate propagates into a new inbound batch and trips the
raise.

The severe case is `populate_delegation_completion`: its inbound is the
WHOLE `fact_agent_delegations` table (by design -- it is a cross-session
pass), and it runs from `run_post_session_reconciliation` OUTSIDE any
per-session try/except, after every session has been processed. So one stale
duplicate anywhere in the warehouse killed the entire `ccutils all` /
`ccutils local` invocation at the end of the run.

Reproduced against a real pre-fix warehouse holding 3 duplicate
`delegation_key` rows: the reconciliation raised.

Delete these and upgrading to the raise silently bricks every warehouse
built before it, with no operator-facing remedy but a full rebuild.
"""

import duckdb
import pytest

from ccutils import create_star_schema
from ccutils.schemas.star.schema import NATURAL_KEYS


def _seed_duplicate(path):
    """Write two live rows sharing one natural key, as a pre-fix warehouse
    would hold."""
    conn = duckdb.connect(str(path))
    for i in (1, 2):
        conn.execute(
            """
            INSERT INTO fact_tool_results (
                created_by_version_key, last_updated_by_version_key,
                etl_run_id, record_source, hash_diff,
                tool_use_id, entry_id, message_id, session_id, tool_name
            ) VALUES ('v', 'v', 'r', 'claude_code_jsonl', ?, 'toolu_dup',
                      ?, 'm1', 's1', 'Bash')
            """,
            [f"h{i}", f"e{i}"],
        )
    conn.close()


class TestNaturalKeyRepair:
    def test_duplicates_are_repaired_on_open(self, tmp_path):
        db = tmp_path / "w.duckdb"
        create_star_schema(db).close()
        _seed_duplicate(db)

        # Re-opening the warehouse must heal it.
        conn = create_star_schema(db)
        live = conn.execute(
            "SELECT COUNT(*) FROM fact_tool_results "
            "WHERE tool_use_id = 'toolu_dup' AND NOT is_deleted"
        ).fetchone()[0]
        assert live == 1

    def test_repair_soft_deletes_rather_than_destroys(self, tmp_path):
        """Non-destructive: the losing row is retained with is_deleted set,
        matching the warehouse's lineage convention. A DELETE would discard
        data the operator never chose to lose."""
        db = tmp_path / "w.duckdb"
        create_star_schema(db).close()
        _seed_duplicate(db)

        conn = create_star_schema(db)
        total, deleted = conn.execute(
            "SELECT COUNT(*), COUNT(*) FILTER (WHERE is_deleted) "
            "FROM fact_tool_results WHERE tool_use_id = 'toolu_dup'"
        ).fetchone()
        assert (total, deleted) == (2, 1)

    def test_repair_is_deterministic(self, tmp_path):
        """Two warehouses seeded identically must keep the same row, or a
        rebuild is not reproducible."""
        survivors = []
        for name in ("a", "b"):
            db = tmp_path / f"{name}.duckdb"
            create_star_schema(db).close()
            _seed_duplicate(db)
            conn = create_star_schema(db)
            survivors.append(
                conn.execute(
                    "SELECT entry_id FROM fact_tool_results "
                    "WHERE tool_use_id = 'toolu_dup' AND NOT is_deleted"
                ).fetchone()[0]
            )
            conn.close()
        assert survivors[0] == survivors[1]

    def test_clean_warehouse_is_untouched(self, tmp_path):
        """Non-vacuity: the repair must not soft-delete anything in a
        warehouse that has no duplicates. Without this, a repair that marked
        everything deleted would satisfy every test above."""
        db = tmp_path / "w.duckdb"
        create_star_schema(db).close()
        conn = duckdb.connect(str(db))
        conn.execute(
            """
            INSERT INTO fact_tool_results (
                created_by_version_key, last_updated_by_version_key,
                etl_run_id, record_source, hash_diff,
                tool_use_id, entry_id, message_id, session_id, tool_name
            ) VALUES ('v','v','r','claude_code_jsonl','h','toolu_a',
                      'e1','m1','s1','Bash')
            """
        )
        conn.close()

        conn = create_star_schema(db)
        deleted = conn.execute(
            "SELECT COUNT(*) FROM fact_tool_results WHERE is_deleted"
        ).fetchone()[0]
        assert deleted == 0

    def test_natural_keys_map_covers_every_upsert_target(self):
        """The map is the single source of truth the repair walks, and later
        the audit command. If a populator declares a natural_key for a table
        absent from it, that table silently stops being repaired or checked.
        """
        import pathlib
        import re

        etl = pathlib.Path("src/ccutils/etl")
        src = "\n".join(
            p.read_text() for p in etl.rglob("*.py")
        )
        # table="..." followed within a few lines by natural_key="..."
        pairs = set(
            re.findall(
                r'table="(\w+)",\s*\n\s*inbound_table="[^"]+",\s*\n\s*natural_key="(\w+)"',
                src,
            )
        )
        assert pairs, "regex found no lineage_upsert call sites -- fix the test"
        missing = {t: k for t, k in pairs if NATURAL_KEYS.get(t) != k}
        assert not missing, (
            f"NATURAL_KEYS disagrees with these populators: {missing}"
        )
