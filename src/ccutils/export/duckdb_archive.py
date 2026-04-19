"""DuckDB archive generation.

This module provides functions for creating DuckDB database archives
from Claude Code session files. Supports both simple (4-table) and
star (28 tables + 14 views) schemas.
"""

import os
import time
from pathlib import Path

import duckdb

from ..parsers import find_all_sessions
from ..schemas import (
    create_duckdb_schema,
    export_session_to_duckdb,
    create_star_schema,
    run_star_schema_etl,
    export_star_schema_to_json,
)
from ..schemas.star.utils import generate_dimension_key


def generate_duckdb_archive(
    source_folder,
    output_dir,
    schema_type="simple",
    include_agents=False,
    include_thinking=False,
    truncate_output=2000,
    progress_callback=None,
    max_workers=1,
    batch_size=10,
    private=False,
):
    """Generate DuckDB archive for all sessions.

    Supports both simple (4-table) and star (28 tables + 14 views) schemas.
    Uses a stage-and-load pattern for efficient batch processing:
    - Stage: Parse sessions (parallelizable with max_workers)
    - Load: Bulk insert in batches (batch_size sessions per transaction)

    Args:
        source_folder: Path to Claude projects folder
        output_dir: Path for output
        schema_type: "simple" (4 tables) or "star" (dimensional model)
        include_agents: Whether to include agent sessions
        include_thinking: Whether to include thinking blocks
        truncate_output: Max chars for tool output
        progress_callback: Optional callback with signature:
            callback(project_name, session_name, current, total, stats)
            where stats is a dict with 'rows_inserted', 'db_size_mb', 'rate'
        max_workers: Number of parallel workers for staging (default: 1)
        batch_size: Sessions per transaction batch (default: 10)

    Returns:
        dict with statistics including row counts
    """
    source_folder = Path(source_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "archive.duckdb"

    # Create appropriate schema
    if schema_type == "star":
        conn = create_star_schema(db_path)
        etl_func = run_star_schema_etl
    else:
        conn = create_duckdb_schema(db_path)
        etl_func = export_session_to_duckdb

    projects = find_all_sessions(source_folder, include_agents=include_agents)

    total_session_count = sum(len(p["sessions"]) for p in projects)
    processed_count = 0
    successful_sessions = 0
    failed_sessions = []

    # Stats tracking
    start_time = time.time()

    # Flatten sessions for processing
    session_tasks = []
    for project in projects:
        project_name = project["name"]
        for session in project["sessions"]:
            session_tasks.append((project_name, session["path"]))

    # Process sessions
    if max_workers > 1 and len(session_tasks) > 1:
        # Parallel processing - stage then load in batches
        _process_parallel(
            conn,
            session_tasks,
            etl_func,
            include_thinking,
            truncate_output,
            batch_size,
            progress_callback,
            db_path,
            start_time,
            failed_sessions,
            schema_type,
            private,
        )
        successful_sessions = len(session_tasks) - len(failed_sessions)
    else:
        # Sequential processing (original behavior)
        for project_name, session_path in session_tasks:
            try:
                etl_func(
                    conn,
                    session_path,
                    project_name,
                    include_thinking=include_thinking,
                    truncate_output=truncate_output,
                    private=private,
                )
                successful_sessions += 1
            except Exception as e:
                failed_sessions.append(
                    {
                        "project": project_name,
                        "session": session_path.stem,
                        "error": str(e),
                    }
                )

            processed_count += 1
            if progress_callback:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                db_size = _get_db_size_mb(db_path)
                stats = {
                    "rows_inserted": _count_rows(conn, schema_type),
                    "db_size_mb": db_size,
                    "rate": rate,
                }
                progress_callback(
                    project_name,
                    session_path.stem,
                    processed_count,
                    total_session_count,
                    stats,
                )

    # Post-ETL batch processing for star schema
    if schema_type == "star":
        finalize_star_schema(conn)

    # Get final row counts
    final_row_count = _count_rows(conn, schema_type)
    final_db_size = _get_db_size_mb(db_path)

    conn.close()

    return {
        "total_projects": len(projects),
        "total_sessions": successful_sessions,
        "failed_sessions": failed_sessions,
        "output_dir": output_dir,
        "db_path": db_path,
        "schema_type": schema_type,
        "rows_inserted": final_row_count,
        "db_size_mb": final_db_size,
    }


def _process_parallel(
    conn,
    session_tasks,
    etl_func,
    include_thinking,
    truncate_output,
    batch_size,
    progress_callback,
    db_path,
    start_time,
    failed_sessions,
    schema_type,
    private=False,
):
    """Process sessions in batches with progress reporting.

    Note: DuckDB connections are not thread-safe for writes, so we
    process in batches and serialize the actual DB writes.
    """
    total = len(session_tasks)
    processed = 0
    rows_total = 0

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = session_tasks[batch_start:batch_end]

        # Process batch - serialize DB writes
        for project_name, session_path in batch:
            try:
                etl_func(
                    conn,
                    session_path,
                    project_name,
                    include_thinking=include_thinking,
                    truncate_output=truncate_output,
                    private=private,
                )
            except Exception as e:
                failed_sessions.append(
                    {
                        "project": project_name,
                        "session": session_path.stem,
                        "error": str(e),
                    }
                )

            processed += 1
            if progress_callback:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                db_size = _get_db_size_mb(db_path)
                # Count rows periodically (expensive, so estimate)
                if processed % 5 == 0:
                    rows_total = _count_rows(conn, schema_type)
                stats = {
                    "rows_inserted": rows_total,
                    "db_size_mb": db_size,
                    "rate": rate,
                }
                progress_callback(
                    project_name,
                    session_path.stem,
                    processed,
                    total,
                    stats,
                )


def _count_rows(conn, schema_type):
    """Count total rows across relevant tables."""
    if schema_type == "star":
        tables = [
            "fact_messages",
            "fact_tool_calls",
            "fact_content_blocks",
            "fact_session_summary",
            "fact_token_usage",
            "fact_turn_durations",
            "fact_diagnostics",
            "fact_stop_events",
        ]
    else:
        tables = ["messages", "tool_calls", "sessions"]

    total = 0
    for table in tables:
        try:
            result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            total += result[0] if result else 0
        except Exception:
            pass
    return total


def _get_db_size_mb(db_path):
    """Get database file size in MB."""
    try:
        size_bytes = os.path.getsize(db_path)
        return round(size_bytes / (1024 * 1024), 2)
    except Exception:
        return 0.0


def generate_star_json_archive(
    source_folder,
    output_dir,
    include_agents=False,
    include_thinking=False,
    truncate_output=2000,
    progress_callback=None,
    max_workers=1,
    batch_size=10,
    private=False,
):
    """Generate star schema JSON archive for all sessions.

    Creates a JSON directory structure with dimensions/ and facts/ subdirs.

    Args:
        source_folder: Path to Claude projects folder
        output_dir: Path for output
        include_agents: Whether to include agent sessions
        include_thinking: Whether to include thinking blocks
        truncate_output: Max chars for tool output
        progress_callback: Optional progress callback
        max_workers: Number of parallel workers (default: 1)
        batch_size: Sessions per batch (default: 10)

    Returns:
        dict with statistics
    """
    import tempfile

    source_folder = Path(source_folder)
    output_dir = Path(output_dir)

    # First build the DuckDB, then export to JSON
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Generate DuckDB with star schema
        stats = generate_duckdb_archive(
            source_folder,
            tmp_path,
            schema_type="star",
            include_agents=include_agents,
            include_thinking=include_thinking,
            truncate_output=truncate_output,
            progress_callback=progress_callback,
            max_workers=max_workers,
            batch_size=batch_size,
            private=private,
        )

        # Export to JSON
        db_path = tmp_path / "archive.duckdb"
        conn = duckdb.connect(str(db_path))
        export_star_schema_to_json(conn, output_dir)
        conn.close()

    stats["output_dir"] = output_dir
    stats["db_path"] = None  # No DuckDB file for JSON export
    return stats


def finalize_star_schema(conn, history_path=None, private=False):
    """Run post-ETL processing for star schema.

    Must be called after all sessions have been loaded via run_star_schema_etl().
    Populates cross-session tables that require all data to be present:
    - dim_session.depth_level (parent-child depth calculation)
    - dim_session_chain (session chain grouping by slug)
    - fact_agent_delegations (agent-to-parent Task tool linking)
    - fact_plan_revisions (ExitPlanMode revision chain + outcomes)
    - bridge_session_file (cross-session file operation aggregation)
    - fact_session_summary._incl_agents rollup (bottom-up agent metric aggregation)
    - dim_prompt (from history.jsonl, if path provided)

    Safe to call multiple times -- each step clears before repopulating.

    Args:
        conn: DuckDB connection
        history_path: Optional path to ~/.claude/history.jsonl
        private: If True, sanitize project paths in history data
    """
    _calculate_session_depths(conn)
    _build_session_chains(conn)
    _link_agent_delegations(conn)
    _link_plan_revisions(conn)
    _build_session_file_bridge(conn)
    _rollup_agent_metrics(conn)

    if history_path:
        from ..schemas.star.history_etl import load_history

        load_history(conn, history_path, private=private)


def _calculate_session_depths(conn):
    """Calculate depth_level for all sessions based on parent-child relationships.

    Iteratively sets depth levels:
    - Root sessions (no parent) get depth 0
    - Children get parent's depth + 1
    - Caps at 100 iterations for safety
    - Orphans (unresolvable parent) stay at 0
    """
    # Root sessions already default to 0
    # Iteratively resolve children
    for _ in range(100):
        updated = conn.execute(
            """
            UPDATE dim_session child
            SET depth_level = parent.depth_level + 1
            FROM dim_session parent
            WHERE child.parent_session_key = parent.session_key
              AND child.parent_session_key IS NOT NULL
              AND child.is_agent = TRUE
              AND parent.depth_level IS NOT NULL
              AND child.depth_level = 0
              AND parent.depth_level >= 0
              AND child.session_key != parent.session_key
            """
        )
        if updated.fetchone() is None:
            break
        # Check if any rows were actually changed
        remaining = conn.execute(
            """
            SELECT COUNT(*) FROM dim_session child
            JOIN dim_session parent ON child.parent_session_key = parent.session_key
            WHERE child.is_agent = TRUE
              AND child.depth_level = 0
              AND parent.depth_level > 0
            """
        ).fetchone()
        if remaining is None or remaining[0] == 0:
            break


def _build_session_chains(conn):
    """Build session chain records from sessions sharing the same slug.

    Groups sessions by slug, creates a dim_session_chain record for each group,
    and updates dim_session.chain_key for all sessions in the chain.
    """
    # Find all distinct slugs with their sessions
    slug_groups = conn.execute(
        """
        SELECT slug, COUNT(*) as cnt,
               MIN(first_timestamp) as first_ts,
               MAX(last_timestamp) as last_ts,
               MIN(session_key) as first_sk,
               MAX(session_key) as last_sk,
               MIN(project_key) as proj_key
        FROM dim_session
        WHERE slug IS NOT NULL AND slug != ''
        GROUP BY slug
        """
    ).fetchall()

    for row in slug_groups:
        slug = row[0]
        session_count = row[1]
        first_ts = row[2]
        last_ts = row[3]
        first_sk = row[4]
        last_sk = row[5]
        proj_key = row[6]

        chain_key = generate_dimension_key(slug)

        # Calculate total duration
        total_duration = 0
        if first_ts and last_ts:
            try:
                total_duration = int((last_ts - first_ts).total_seconds())
            except (TypeError, AttributeError):
                pass

        # Insert chain record
        if not conn.execute(
            "SELECT 1 FROM dim_session_chain WHERE chain_key = ?", [chain_key]
        ).fetchone():
            conn.execute(
                """INSERT INTO dim_session_chain
                   (chain_key, slug, project_key, first_session_key,
                    last_session_key, session_count, first_timestamp,
                    last_timestamp, total_duration_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    chain_key,
                    slug,
                    proj_key,
                    first_sk,
                    last_sk,
                    session_count,
                    first_ts,
                    last_ts,
                    total_duration,
                ],
            )

        # Update all sessions in this chain
        conn.execute(
            "UPDATE dim_session SET chain_key = ? WHERE slug = ?",
            [chain_key, slug],
        )


def _link_agent_delegations(conn):
    """Link agent sessions to their parent's Task tool_use calls.

    Uses a two-tier matching strategy:
    1. Deterministic: progress records provide tool_use_id -> agent_id links
       (confidence 1.0, zero ambiguity)
    2. Fallback heuristic: timestamp proximity matching for older data without
       progress records (confidence 0.5-0.8)

    The deterministic path uses stg_task_agent_map (populated from progress
    records during ETL) joined with dim_session.agent_id to find the exact
    tool_call_id for each agent session.

    Idempotent: clears fact_agent_delegations before repopulating.
    """
    conn.execute("DELETE FROM fact_agent_delegations")
    import json

    # Build deterministic lookup: agent_id -> tool_use_id
    # from progress records captured during ETL
    deterministic_map = {}
    try:
        map_rows = conn.execute(
            """
            SELECT tam.tool_use_id, tam.agent_id, ds.session_key
            FROM stg_task_agent_map tam
            JOIN dim_session ds ON ds.agent_id = tam.agent_id AND ds.is_agent = TRUE
            """
        ).fetchall()
        for row in map_rows:
            # agent_session_key -> tool_use_id
            deterministic_map[row[2]] = row[0]
    except Exception:
        # Table may not exist in older databases
        pass

    # Find all agent sessions with parents
    agents = conn.execute(
        """
        SELECT a.session_key, a.parent_session_key, a.first_timestamp, a.last_timestamp
        FROM dim_session a
        WHERE a.is_agent = TRUE AND a.parent_session_key IS NOT NULL
        """
    ).fetchall()

    for agent_row in agents:
        agent_session_key = agent_row[0]
        parent_session_key = agent_row[1]
        agent_first_ts = agent_row[2]
        agent_last_ts = agent_row[3]

        # Find Task tool calls in the parent session
        task_calls = conn.execute(
            """
            SELECT ftc.tool_call_id, ftc.timestamp, ftc.input_json,
                   ftc.is_error, ftc.output_text, ftc.date_key, ftc.time_key
            FROM fact_tool_calls ftc
            JOIN dim_tool dt ON ftc.tool_key = dt.tool_key
            WHERE ftc.session_key = ? AND dt.tool_name = 'Task'
            ORDER BY ftc.timestamp
            """,
            [parent_session_key],
        ).fetchall()

        if not task_calls:
            continue

        # Tier 1: Deterministic match via progress records
        best_match = None
        match_confidence = None

        if agent_session_key in deterministic_map:
            target_tool_id = deterministic_map[agent_session_key]
            for tc in task_calls:
                if tc[0] == target_tool_id:
                    best_match = tc
                    match_confidence = 1.0
                    break

        # Tier 2: Fallback heuristic - timestamp proximity
        if best_match is None:
            best_distance = float("inf")

            for tc in task_calls:
                tc_timestamp = tc[1]
                if tc_timestamp and agent_first_ts:
                    try:
                        distance = abs((agent_first_ts - tc_timestamp).total_seconds())
                        if distance < best_distance:
                            best_distance = distance
                            best_match = tc
                    except (TypeError, AttributeError):
                        continue

            if best_match is not None:
                match_confidence = 1.0
                if len(task_calls) > 1:
                    close_calls = sum(
                        1
                        for tc in task_calls
                        if tc[1]
                        and agent_first_ts
                        and abs((agent_first_ts - tc[1]).total_seconds()) < 60
                    )
                    if close_calls > 1:
                        match_confidence = 0.5
                    else:
                        match_confidence = 0.8

        if best_match is None:
            continue

        # Parse Task input
        tool_call_id = best_match[0]
        tc_timestamp = best_match[1]
        input_json_str = best_match[2]
        is_error = best_match[3]
        output_text = best_match[4]
        date_key = best_match[5]
        time_key = best_match[6]

        task_description = None
        task_prompt = None
        subagent_type = None

        if input_json_str:
            try:
                input_data = json.loads(input_json_str)
                task_description = input_data.get("description")
                task_prompt = input_data.get("prompt")
                subagent_type = input_data.get("subagent_type")
            except (json.JSONDecodeError, TypeError):
                pass

        # Determine completion status
        completion_status = "unknown"
        if is_error:
            completion_status = "error"
        elif output_text:
            completion_status = "completed"

        delegation_key = generate_dimension_key(parent_session_key, agent_session_key)

        # Denormalize agent metrics from fact_session_summary
        agent_tool_calls = None
        agent_errors = None
        agent_duration_seconds = None
        agent_estimated_tokens = None
        agent_summary = conn.execute(
            """SELECT total_tool_calls, total_errors, session_duration_seconds,
                      total_estimated_tokens
               FROM fact_session_summary WHERE session_key = ?""",
            [agent_session_key],
        ).fetchone()
        if agent_summary:
            agent_tool_calls = agent_summary[0]
            agent_errors = agent_summary[1]
            agent_duration_seconds = agent_summary[2]
            agent_estimated_tokens = agent_summary[3]

        conn.execute(
            """INSERT INTO fact_agent_delegations
               (delegation_key, parent_session_key, agent_session_key,
                task_tool_call_id, date_key, time_key,
                task_description, task_prompt, subagent_type,
                agent_output, completion_status,
                delegation_timestamp, completion_timestamp, match_confidence,
                agent_tool_calls, agent_errors, agent_duration_seconds,
                agent_estimated_tokens)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                delegation_key,
                parent_session_key,
                agent_session_key,
                tool_call_id,
                date_key,
                time_key,
                task_description,
                task_prompt,
                subagent_type,
                output_text[:2000] if output_text else None,
                completion_status,
                tc_timestamp,
                agent_last_ts,
                match_confidence,
                agent_tool_calls,
                agent_errors,
                agent_duration_seconds,
                agent_estimated_tokens,
            ],
        )


# Substring match signals approval in the ExitPlanMode tool_result body.
# Claude Code emits "User has approved your plan. You can now start coding."
# when the user accepts a plan. If Claude Code changes the signal in future
# versions, update this constant.
PLAN_APPROVAL_SIGNATURE = "approved your plan"


def _link_plan_revisions(conn):
    """Walk ExitPlanMode tool calls per session and build the plan revision chain.

    For each session, links successive ExitPlanMode invocations parent->child,
    classifies the outcome of each revision, and captures any user feedback
    message that followed a rejection.

    Outcome precedence (first match wins):
      - 'superseded' : a later ExitPlanMode exists in the same session
      - 'accepted'   : tool_result body contains the approval signature
      - 'rejected'   : a user text message followed the tool_result
      - 'pending'    : session ended with no follow-up signal

    Always reads the full tool_result body from fact_content_blocks.content_json
    (not fact_tool_calls.output_text, which is truncated to 2000 chars).

    Idempotent: clears fact_plan_revisions before repopulating.
    """
    import json

    from ..schemas.star.extractors import estimate_tokens

    conn.execute("DELETE FROM fact_plan_revisions")

    plan_rows = conn.execute(
        """
        SELECT ftc.session_key,
               ftc.tool_call_id,
               ftc.invoke_message_id,
               ftc.result_message_id,
               ftc.timestamp,
               ftc.input_json,
               ftc.date_key,
               ftc.time_key,
               ds.project_key
        FROM fact_tool_calls ftc
        JOIN dim_tool dt ON ftc.tool_key = dt.tool_key
        JOIN dim_session ds ON ftc.session_key = ds.session_key
        WHERE dt.tool_name = 'ExitPlanMode'
        ORDER BY ftc.session_key, ftc.timestamp
        """
    ).fetchall()

    if not plan_rows:
        return

    # Group by session (preserves timestamp order within each session)
    sessions = {}
    for row in plan_rows:
        sessions.setdefault(row[0], []).append(row)

    insert_rows = []
    for session_key, revisions in sessions.items():
        prev_revision_key = None
        for idx, rev in enumerate(revisions):
            (
                _sess_key,
                tool_call_id,
                invoke_message_id,
                result_message_id,
                plan_timestamp,
                input_json,
                date_key,
                time_key,
                project_key,
            ) = rev
            del _sess_key

            revision_number = idx + 1
            revision_key = generate_dimension_key(session_key, tool_call_id)

            # Extract plan text from input_json
            plan_text = None
            if input_json:
                try:
                    plan_text = json.loads(input_json).get("plan")
                except (json.JSONDecodeError, TypeError):
                    plan_text = None
            plan_char_count = len(plan_text) if plan_text else 0
            plan_estimated_tokens = estimate_tokens(plan_text) if plan_text else 0

            is_last_in_session = idx == len(revisions) - 1

            # Pull full tool_result body (untruncated) from content_json
            result_content_text = None
            resolved_timestamp = None
            if result_message_id:
                cb_row = conn.execute(
                    """
                    SELECT fcb.content_json, fm.timestamp
                    FROM fact_content_blocks fcb
                    JOIN fact_messages fm ON fcb.message_id = fm.message_id
                    WHERE fcb.message_id = ?
                      AND fcb.block_type = 'tool_result'
                    """,
                    [result_message_id],
                ).fetchall()
                # If multiple tool_results in the same message, pick the one
                # whose tool_use_id matches our tool_call_id.
                for content_json, msg_ts in cb_row:
                    try:
                        block = json.loads(content_json)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if block.get("tool_use_id") == tool_call_id:
                        content = block.get("content", "")
                        if isinstance(content, list):
                            result_content_text = " ".join(
                                str(item.get("text", ""))
                                for item in content
                                if isinstance(item, dict)
                            )
                        else:
                            result_content_text = str(content)
                        resolved_timestamp = msg_ts
                        break
                # Fallback: no tool_use_id match, take the first tool_result
                if result_content_text is None and cb_row:
                    content_json, msg_ts = cb_row[0]
                    try:
                        block = json.loads(content_json)
                        content = block.get("content", "")
                        if isinstance(content, list):
                            result_content_text = " ".join(
                                str(item.get("text", ""))
                                for item in content
                                if isinstance(item, dict)
                            )
                        else:
                            result_content_text = str(content)
                        resolved_timestamp = msg_ts
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Classify outcome
            if not is_last_in_session:
                outcome = "superseded"
                outcome_signal = "next_plan"
            elif result_content_text and PLAN_APPROVAL_SIGNATURE in result_content_text:
                outcome = "accepted"
                outcome_signal = "tool_result_approve"
            else:
                outcome = None
                outcome_signal = None

            # If not yet classified, look for a following user message
            user_feedback_message_id = None
            user_feedback_text = None
            if outcome is None:
                ref_ts = resolved_timestamp or plan_timestamp
                follow_up = conn.execute(
                    """
                    SELECT message_id, content_text
                    FROM fact_messages
                    WHERE session_key = ?
                      AND message_type = 'user'
                      AND timestamp > ?
                    ORDER BY timestamp
                    LIMIT 1
                    """,
                    [session_key, ref_ts],
                ).fetchone()
                if follow_up:
                    user_feedback_message_id = follow_up[0]
                    user_feedback_text = (follow_up[1] or "")[:2000]
                    outcome = "rejected"
                    outcome_signal = "next_user_msg"
                else:
                    outcome = "pending"
                    outcome_signal = "session_end"

            # For superseded, also capture the next user message as feedback if any
            if outcome == "superseded":
                ref_ts = resolved_timestamp or plan_timestamp
                follow_up = conn.execute(
                    """
                    SELECT message_id, content_text
                    FROM fact_messages
                    WHERE session_key = ?
                      AND message_type = 'user'
                      AND timestamp > ?
                    ORDER BY timestamp
                    LIMIT 1
                    """,
                    [session_key, ref_ts],
                ).fetchone()
                if follow_up:
                    user_feedback_message_id = follow_up[0]
                    user_feedback_text = (follow_up[1] or "")[:2000]

            seconds_to_resolution = None
            if resolved_timestamp and plan_timestamp:
                try:
                    seconds_to_resolution = (
                        resolved_timestamp - plan_timestamp
                    ).total_seconds()
                except (TypeError, AttributeError):
                    seconds_to_resolution = None

            insert_rows.append(
                (
                    revision_key,
                    session_key,
                    project_key,
                    date_key,
                    time_key,
                    tool_call_id,
                    invoke_message_id,
                    result_message_id,
                    revision_number,
                    prev_revision_key,
                    plan_text,
                    plan_char_count,
                    plan_estimated_tokens,
                    outcome,
                    outcome_signal,
                    user_feedback_message_id,
                    user_feedback_text,
                    plan_timestamp,
                    resolved_timestamp,
                    seconds_to_resolution,
                )
            )
            prev_revision_key = revision_key

    if insert_rows:
        conn.executemany(
            """INSERT INTO fact_plan_revisions
               (revision_key, session_key, project_key, date_key, time_key,
                tool_call_id, invoke_message_id, result_message_id,
                revision_number, parent_revision_key,
                plan_text, plan_char_count, plan_estimated_tokens,
                outcome, outcome_signal,
                user_feedback_message_id, user_feedback_text,
                plan_timestamp, resolved_timestamp, seconds_to_resolution)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            insert_rows,
        )


def _build_session_file_bridge(conn):
    """Build bridge table aggregating file operations by session and file.

    Aggregates fact_file_operations into per-(session, file) summaries
    with operation counts broken down by type.

    Idempotent: clears bridge_session_file before repopulating.
    """
    conn.execute("DELETE FROM bridge_session_file")
    conn.execute(
        """
        INSERT INTO bridge_session_file
        SELECT
            md5(ffo.session_key || '|' || ffo.file_key) AS session_file_key,
            ffo.session_key,
            ffo.file_key,
            MIN(ffo.timestamp) AS first_operation_timestamp,
            MAX(ffo.timestamp) AS last_operation_timestamp,
            COUNT(*) AS operation_count,
            SUM(CASE WHEN ffo.operation_type = 'read' THEN 1 ELSE 0 END) AS read_count,
            SUM(CASE WHEN ffo.operation_type = 'write' THEN 1 ELSE 0 END) AS write_count,
            SUM(CASE WHEN ffo.operation_type = 'edit' THEN 1 ELSE 0 END) AS edit_count,
            SUM(COALESCE(ffo.file_size_chars, 0)) AS total_chars_written
        FROM fact_file_operations ffo
        WHERE ffo.file_key IS NOT NULL
        GROUP BY ffo.session_key, ffo.file_key
        """
    )


def _rollup_agent_metrics(conn):
    """Roll up agent metrics into parent session summaries.

    Walks the session hierarchy bottom-up (deepest agents first) and
    accumulates descendant metrics into each parent's _incl_agents columns.
    Uses dim_session.depth_level (set by _calculate_session_depths) to
    process in correct order.

    The _incl_agents columns are initialized to the session's own values
    during ETL. This function adds descendant contributions on top.

    Idempotent: resets _incl_agents to own values before re-accumulating.
    """
    # Reset all _incl_agents columns to own values
    conn.execute(
        """
        UPDATE fact_session_summary
        SET total_estimated_tokens_incl_agents = total_estimated_tokens,
            total_tool_calls_incl_agents = total_tool_calls,
            total_errors_incl_agents = total_errors,
            total_duration_incl_agents = session_duration_seconds
        """
    )

    # Find max depth to iterate bottom-up
    max_depth_row = conn.execute(
        "SELECT MAX(depth_level) FROM dim_session WHERE is_agent = TRUE"
    ).fetchone()
    max_depth = max_depth_row[0] if max_depth_row and max_depth_row[0] else 0

    # Walk bottom-up: deepest agents first, rolling into their parents
    for depth in range(max_depth, 0, -1):
        conn.execute(
            """
            UPDATE fact_session_summary parent_fss
            SET total_estimated_tokens_incl_agents =
                    parent_fss.total_estimated_tokens_incl_agents + agent_totals.sum_tokens,
                total_tool_calls_incl_agents =
                    parent_fss.total_tool_calls_incl_agents + agent_totals.sum_tool_calls,
                total_errors_incl_agents =
                    parent_fss.total_errors_incl_agents + agent_totals.sum_errors,
                total_duration_incl_agents =
                    parent_fss.total_duration_incl_agents + agent_totals.sum_duration
            FROM (
                SELECT ds_agent.parent_session_key AS parent_sk,
                       SUM(COALESCE(agent_fss.total_estimated_tokens_incl_agents, 0)) AS sum_tokens,
                       SUM(COALESCE(agent_fss.total_tool_calls_incl_agents, 0)) AS sum_tool_calls,
                       SUM(COALESCE(agent_fss.total_errors_incl_agents, 0)) AS sum_errors,
                       SUM(COALESCE(agent_fss.total_duration_incl_agents, 0)) AS sum_duration
                FROM dim_session ds_agent
                JOIN fact_session_summary agent_fss
                    ON ds_agent.session_key = agent_fss.session_key
                WHERE ds_agent.is_agent = TRUE
                  AND ds_agent.depth_level = ?
                  AND ds_agent.parent_session_key IS NOT NULL
                GROUP BY ds_agent.parent_session_key
            ) agent_totals
            WHERE parent_fss.session_key = agent_totals.parent_sk
            """,
            [depth],
        )
