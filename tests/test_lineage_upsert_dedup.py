"""`lineage_upsert` must hold the natural key it declares.

Claim these tests encode: a populator that hands `lineage_upsert` two inbound
rows sharing a natural_key must not produce two target rows. The helper's
docstring states "one row per natural key" as an inbound contract, but nothing
enforced it: the INSERT guards with `NOT EXISTS (SELECT 1 FROM tgt WHERE
tgt.key = im.key)`, which only consults the TARGET, so two inbound rows with
the same key both pass the check and both insert.

Delete these tests and the declared uniqueness of every fact's natural key
stops being a guarantee and becomes a hope. Measured on a real 2,344-session
corpus before the fix: 6 of 13 facts violated their own declared key --
fact_tool_results 29 keys, fact_file_operations 8, fact_tool_uses 7,
fact_tool_chain_steps 7, fact_agent_delegations 3, fact_errors 1.

The duplication is real source data, not a synthetic worry: a single Claude
Code session can record one tool_use_id under two distinct entry uuids.
"""

import pytest

from ccutils import create_star_schema
from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert


PAYLOAD = ["entry_id", "message_id", "tool_name", "timestamp", "is_error"]
HASH = ["entry_id", "tool_name", "is_error"]


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


def _rows(conn, tool_use_id="toolu_dup"):
    return conn.execute(
        "SELECT entry_id, tool_name, is_error FROM fact_tool_results "
        "WHERE tool_use_id = ? AND NOT is_deleted ORDER BY entry_id",
        [tool_use_id],
    ).fetchall()


TS = "2026-08-02 10:00:00"


class TestIntraBatchDeduplication:
    def test_identical_duplicate_rows_insert_once(self, conn):
        """Two byte-identical inbound rows for one key -> one target row."""
        _inbound(
            conn,
            [
                ("toolu_dup", "s1", "e1", "m1", "Bash", TS, False),
                ("toolu_dup", "s1", "e1", "m1", "Bash", TS, False),
            ],
        )
        _upsert(conn)
        assert len(_rows(conn)) == 1

    def test_distinct_entries_sharing_one_tool_use_id_insert_once(self, conn):
        """The real corpus shape: two source entries, one tool_use_id.

        22 of the 29 violating keys had two DISTINCT entry_ids, so this is
        not the trivial identical-row case.
        """
        _inbound(
            conn,
            [
                ("toolu_dup", "s1", "e1", "m1", "Bash", TS, False),
                ("toolu_dup", "s1", "e2", "m1", "Bash", TS, False),
            ],
        )
        _upsert(conn)
        assert len(_rows(conn)) == 1

    def test_conflicting_payloads_resolve_deterministically(self, conn, tmp_path):
        """Same key, DIFFERENT payloads -> one row, and the same one every run.

        8 of the 29 violating keys carried two different payload hashes, so
        the helper must make a choice; an arbitrary choice that varies between
        runs would make the warehouse non-reproducible.
        """
        rows = [
            ("toolu_dup", "s1", "e1", "m1", "Bash", TS, True),
            ("toolu_dup", "s1", "e2", "m1", "Bash", TS, False),
        ]
        _inbound(conn, rows)
        _upsert(conn)
        first = _rows(conn)
        assert len(first) == 1

        # Same inbound, a fresh warehouse: the surviving row must match.
        other = create_star_schema(tmp_path / "other.duckdb")
        _inbound(other, list(reversed(rows)))
        _upsert(other)
        assert _rows(other) == first

    def test_deduplication_does_not_drop_distinct_keys(self, conn):
        """Non-vacuity: dedup must not be 'keep one row overall'.

        Without this, a helper that collapsed the entire inbound batch to a
        single row would pass every test above.
        """
        _inbound(
            conn,
            [
                ("toolu_a", "s1", "e1", "m1", "Bash", TS, False),
                ("toolu_a", "s1", "e2", "m1", "Bash", TS, False),
                ("toolu_b", "s1", "e3", "m1", "Read", TS, False),
                ("toolu_c", "s1", "e4", "m1", "Edit", TS, False),
            ],
        )
        _upsert(conn)
        keys = conn.execute(
            "SELECT tool_use_id, COUNT(*) FROM fact_tool_results "
            "WHERE NOT is_deleted GROUP BY 1 ORDER BY 1"
        ).fetchall()
        assert keys == [("toolu_a", 1), ("toolu_b", 1), ("toolu_c", 1)]

    def test_step_row_counts_stay_honest(self, conn):
        """rows_read reports what arrived; rows_inserted what landed.

        The gap is the only signal that a populator is emitting duplicates,
        so silently reporting rows_read == rows_inserted would hide it.
        """
        _inbound(
            conn,
            [
                ("toolu_dup", "s1", "e1", "m1", "Bash", TS, False),
                ("toolu_dup", "s1", "e2", "m1", "Bash", TS, False),
                ("toolu_b", "s1", "e3", "m1", "Read", TS, False),
            ],
        )
        run = _upsert(conn)
        read, inserted = conn.execute(
            "SELECT rows_read, rows_inserted FROM fact_etl_steps "
            "WHERE etl_run_id = ? AND step_name = 'upsert:fact_tool_results'",
            [run.etl_run_id],
        ).fetchone()
        assert read == 3
        assert inserted == 2

    def test_second_run_is_still_idempotent(self, conn):
        """Dedup must not break the re-run-is-a-no-op property."""
        rows = [
            ("toolu_dup", "s1", "e1", "m1", "Bash", TS, False),
            ("toolu_dup", "s1", "e2", "m1", "Bash", TS, False),
        ]
        _inbound(conn, rows)
        _upsert(conn)
        _inbound(conn, rows)
        run2 = _upsert(conn)

        assert len(_rows(conn)) == 1
        inserted, updated = conn.execute(
            "SELECT rows_inserted, rows_updated FROM fact_etl_steps "
            "WHERE etl_run_id = ? AND step_name = 'upsert:fact_tool_results'",
            [run2.etl_run_id],
        ).fetchone()
        assert inserted == 0
        assert updated == 0
