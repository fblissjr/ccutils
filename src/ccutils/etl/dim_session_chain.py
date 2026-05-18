"""Populate dim_session_chain from dim_session.

Grain: one row per distinct Claude Code slug. A chain groups sessions
that share a slug (set by the user via /save / continuation). Stays
minimal -- matches the dim_tool / dim_model pattern (no lineage block).

Idempotent delete-and-reload: chains are recomputed from current
dim_session each run, since adding a new session to an existing slug
should update session_count and last_session_key on the existing chain.
Cheap to recompute -- one row per slug.

dim_session.chain_key gets pointed at the chain after rebuild so
JOIN dim_session_chain works without slug-on-slug.

Run AFTER _upsert_minimal_dimensions (which populates dim_session.slug).
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun


def populate_dim_session_chain(conn, *, run: EtlRun) -> None:
    """Rebuild dim_session_chain from dim_session + repoint chain_keys."""
    conn.execute("DELETE FROM dim_session_chain")
    conn.execute(
        """
        INSERT INTO dim_session_chain (
            chain_key, slug, project_key,
            first_session_key, last_session_key,
            session_count, first_timestamp, last_timestamp,
            total_duration_seconds
        )
        WITH chains AS (
            SELECT
                md5(slug) AS chain_key,
                slug,
                ANY_VALUE(project_key) AS project_key,
                COUNT(*) AS session_count,
                MIN(first_timestamp) AS first_timestamp,
                MAX(last_timestamp) AS last_timestamp
            FROM dim_session
            WHERE slug IS NOT NULL AND slug <> ''
            GROUP BY slug
        ),
        first_session AS (
            SELECT DISTINCT ON (slug)
                slug, session_key AS first_session_key
            FROM dim_session
            WHERE slug IS NOT NULL AND slug <> '' AND first_timestamp IS NOT NULL
            ORDER BY slug, first_timestamp ASC
        ),
        last_session AS (
            SELECT DISTINCT ON (slug)
                slug, session_key AS last_session_key
            FROM dim_session
            WHERE slug IS NOT NULL AND slug <> '' AND last_timestamp IS NOT NULL
            ORDER BY slug, last_timestamp DESC
        )
        SELECT
            c.chain_key,
            c.slug,
            c.project_key,
            fs.first_session_key,
            ls.last_session_key,
            c.session_count,
            c.first_timestamp,
            c.last_timestamp,
            CASE WHEN c.first_timestamp IS NOT NULL AND c.last_timestamp IS NOT NULL
                THEN CAST(EXTRACT(EPOCH FROM (c.last_timestamp - c.first_timestamp))
                          AS INTEGER)
                ELSE NULL
            END AS total_duration_seconds
        FROM chains c
        LEFT JOIN first_session fs USING (slug)
        LEFT JOIN last_session ls USING (slug)
        """
    )
    # Repoint dim_session.chain_key at the new chain rows. Sessions
    # without a slug stay NULL.
    conn.execute(
        """
        UPDATE dim_session
        SET chain_key = md5(slug)
        WHERE slug IS NOT NULL AND slug <> ''
        """
    )
    # dim_session_chain matches the dim_tool/dim_model pattern; no
    # version stamp.
    _ = run
