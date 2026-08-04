"""`lineage_upsert` asserts the natural key it declares; it does not fix it.

Claim these tests encode: a populator that hands `lineage_upsert` two inbound
rows sharing a natural_key has a broken projection, and the helper must say
so rather than resolve it. The INSERT guards with `NOT EXISTS (SELECT 1 FROM
tgt WHERE tgt.key = im.key)`, which consults only the TARGET, so without this
assertion both rows insert and the declared uniqueness quietly stops being
true.

Measured on a real 2,344-session corpus, 6 of 13 facts were violating their
own declared key -- fact_tool_results 29, fact_file_operations 8,
fact_tool_uses 7, fact_tool_chain_steps 7, fact_agent_delegations 3,
fact_errors 1.

This file previously asserted the opposite: that the helper silently
collapsed duplicates. That was the wrong layer. Only the populator knows
whether collapsing is safe and which row should survive, and a generic
collapse applied one fact's judgment to all 13 while being invisible in the
step counts. The collapse now lives in the projections that declare the grain
(see tests/test_projection_grain_v15.py); this asserts the backstop.

Delete these and a future populator with a fanned-out join silently
double-counts instead of failing.
"""

import pytest

from ccutils import create_star_schema
from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert


PAYLOAD = ["entry_id", "message_id", "tool_name", "timestamp", "is_error"]
HASH = ["entry_id", "tool_name", "is_error"]
TS = "2026-08-02 10:00:00"


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


def _inbound(conn, rows):
    """Build an inbound temp table shaped like the tool-results populator's."""
    conn.execute("DROP TABLE IF EXISTS _inbound_x")
    conn.execute(
        """
        CREATE TEMP TABLE _inbound_x (
            tool_use_id VARCHAR, session_id VARCHAR, entry_id VARCHAR,
            message_id VARCHAR, tool_name VARCHAR, timestamp TIMESTAMP,
            is_error BOOLEAN
        )
        """
    )
    conn.executemany(
        "INSERT INTO _inbound_x VALUES (?, ?, ?, ?, ?, ?, ?)", rows
    )


def _upsert(conn):
    run = EtlRun.start(conn, source_path="test")
    lineage_upsert(
        conn,
        run=run,
        table="fact_tool_results",
        inbound_table="_inbound_x",
        natural_key="tool_use_id",
        payload_cols=PAYLOAD,
        hash_cols=HASH,
    )
    return run


class TestNaturalKeyIsAsserted:
    def test_duplicate_key_raises(self, conn):
        _inbound(
            conn,
            [
                ("toolu_dup", "s1", "e1", "m1", "Bash", TS, False),
                ("toolu_dup", "s1", "e2", "m1", "Bash", TS, False),
            ],
        )
        with pytest.raises(ValueError) as exc:
            _upsert(conn)
        msg = str(exc.value)
        # The message must be actionable: name the key and point at the fix.
        assert "tool_use_id" in msg
        assert "fact_tool_results" in msg
        assert "projection" in msg.lower()

    def test_nothing_is_written_when_it_raises(self, conn):
        """Failing loud must not half-apply the batch."""
        _inbound(
            conn,
            [
                ("toolu_dup", "s1", "e1", "m1", "Bash", TS, False),
                ("toolu_dup", "s1", "e2", "m1", "Bash", TS, False),
            ],
        )
        with pytest.raises(ValueError):
            _upsert(conn)
        n = conn.execute(
            "SELECT COUNT(*) FROM fact_tool_results WHERE tool_use_id = 'toolu_dup'"
        ).fetchone()[0]
        assert n == 0

    def test_clean_batch_still_inserts(self, conn):
        """Non-vacuity: the assertion must not reject valid batches. Without
        this, a helper that raised unconditionally would pass everything
        above."""
        _inbound(
            conn,
            [
                ("toolu_a", "s1", "e1", "m1", "Bash", TS, False),
                ("toolu_b", "s1", "e2", "m1", "Read", TS, False),
                ("toolu_c", "s1", "e3", "m1", "Edit", TS, False),
            ],
        )
        run = _upsert(conn)
        keys = conn.execute(
            "SELECT tool_use_id, COUNT(*) FROM fact_tool_results "
            "WHERE NOT is_deleted GROUP BY 1 ORDER BY 1"
        ).fetchall()
        assert keys == [("toolu_a", 1), ("toolu_b", 1), ("toolu_c", 1)]

        read, inserted = conn.execute(
            "SELECT rows_read, rows_inserted FROM fact_etl_steps "
            "WHERE etl_run_id = ? AND step_name = 'upsert:fact_tool_results'",
            [run.etl_run_id],
        ).fetchone()
        assert (read, inserted) == (3, 3)

    def test_null_keys_do_not_trip_the_assertion(self, conn):
        """NULL is not a duplicate of NULL for this purpose. Every fact's
        natural-key column is NOT NULL at the DDL level, so a NULL key fails
        on INSERT anyway -- the assertion must not pre-empt that with a
        misleading 'duplicate key' message."""
        _inbound(
            conn,
            [
                ("toolu_a", "s1", "e1", "m1", "Bash", TS, False),
                (None, "s1", "e2", "m1", "Read", TS, False),
                (None, "s1", "e3", "m1", "Edit", TS, False),
            ],
        )
        with pytest.raises(Exception) as exc:
            _upsert(conn)
        assert "duplicate" not in str(exc.value).lower()

    def test_re_run_is_idempotent(self, conn):
        rows = [("toolu_a", "s1", "e1", "m1", "Bash", TS, False)]
        _inbound(conn, rows)
        _upsert(conn)
        _inbound(conn, rows)
        run2 = _upsert(conn)

        inserted, updated = conn.execute(
            "SELECT rows_inserted, rows_updated FROM fact_etl_steps "
            "WHERE etl_run_id = ? AND step_name = 'upsert:fact_tool_results'",
            [run2.etl_run_id],
        ).fetchone()
        assert (inserted, updated) == (0, 0)


class TestUpdateAddressesOneRowPerKey:
    """The UPDATE step must touch exactly one physical row per natural key.

    Claim: a natural key names ONE logical row. Matching `tgt.key = im.key`
    alone addresses every physical row sharing the key -- including twins
    `_repair_duplicate_natural_keys` soft-deleted on open -- and the SET
    includes `is_deleted = FALSE`, so a single hash change resurrects all
    of them. Observed on a real pre-fix warehouse: the open-time repair
    soft-deleted 29 duplicate tool_use_ids; the next batch run added a new
    hash column, every row's hash changed, and all 29 twins came back live
    -- killing the run in `populate_delegation_completion`, after every
    session had been processed. A fresh rebuild can never catch this.
    """

    def _plant_soft_deleted_twin(self, conn, key, twin_entry):
        """Simulate a repaired warehouse: a second physical row for `key`,
        soft-deleted by the repair, with a stale hash."""
        conn.execute(
            "INSERT INTO fact_tool_results SELECT * REPLACE ("
            "  ? AS entry_id, TRUE AS is_deleted, "
            "  current_timestamp AS deleted_at, 'stale-twin-hash' AS hash_diff)"
            "FROM fact_tool_results WHERE tool_use_id = ?",
            [twin_entry, key],
        )

    def test_repaired_twin_stays_dead_when_the_live_rows_hash_changes(self, conn):
        _inbound(conn, [("toolu_a", "s1", "e1", "m1", "Bash", TS, False)])
        _upsert(conn)
        self._plant_soft_deleted_twin(conn, "toolu_a", "e_twin")

        # Content change -> hash differs from BOTH physical rows.
        _inbound(conn, [("toolu_a", "s1", "e1", "m1", "Bash", TS, True)])
        _upsert(conn)

        live = conn.execute(
            "SELECT entry_id, is_error FROM fact_tool_results "
            "WHERE tool_use_id = 'toolu_a' AND NOT is_deleted"
        ).fetchall()
        assert live == [("e1", True)], (
            f"expected the one live row updated in place, got {live} -- "
            "a resurrected twin means the UPDATE addressed the key, not a row"
        )
        twin = conn.execute(
            "SELECT is_deleted FROM fact_tool_results WHERE entry_id = 'e_twin'"
        ).fetchone()[0]
        assert twin is True, "the repaired twin must stay soft-deleted"

    def test_soft_deleted_row_still_revives_when_its_key_returns(self, conn):
        """Non-vacuity for the fix: revival through the UPDATE is the ONLY
        way a soft-deleted row comes back (the INSERT's NOT EXISTS matches
        deleted rows too). Restricting the UPDATE must not break it."""
        _inbound(conn, [("toolu_gone", "s1", "e1", "m1", "Bash", TS, False)])
        _upsert(conn)
        conn.execute(
            "UPDATE fact_tool_results SET is_deleted = TRUE, "
            "deleted_at = current_timestamp WHERE tool_use_id = 'toolu_gone'"
        )

        _inbound(conn, [("toolu_gone", "s1", "e1", "m1", "Bash", TS, True)])
        _upsert(conn)

        n = conn.execute(
            "SELECT COUNT(*) FROM fact_tool_results "
            "WHERE tool_use_id = 'toolu_gone' AND NOT is_deleted"
        ).fetchone()[0]
        assert n == 1, "a returning key must revive its (only) soft-deleted row"
