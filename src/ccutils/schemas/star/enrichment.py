"""LLM enrichment pipeline for star schema."""

from datetime import datetime

from .utils import generate_dimension_key


def run_llm_enrichment(
    conn,
    enrich_func,
    model_name="claude-3-haiku-20240307",
    batch_size=10,
    session_key=None,
):
    """Run LLM enrichment on messages that haven't been enriched yet.

    This function provides a framework for enriching messages with LLM-derived
    classifications like intent, sentiment, and topics.

    Args:
        conn: DuckDB connection
        enrich_func: Function(messages) -> list of enrichment results.
                    Each result should be a dict with:
                    - message_id: str
                    - intent: str (should match dim_intent.intent_name)
                    - sentiment: str (should match dim_sentiment.sentiment_name)
                    - topics: list[str] (should match dim_topic.topic_name values)
                    - complexity_score: float (0-1)
                    - confidence_score: float (0-1)
        model_name: Name of the model used for enrichment (for tracking)
        batch_size: Number of messages to process at once
        session_key: Optional session key to limit enrichment to one session

    Returns:
        dict with counts: messages_enriched, topics_assigned
    """
    query = """
        SELECT m.message_id, m.session_key, m.content_text, mt.message_type
        FROM fact_messages m
        JOIN dim_message_type mt ON m.message_type_key = mt.message_type_key
        LEFT JOIN fact_message_enrichment e ON m.message_id = e.message_id
        WHERE e.message_id IS NULL
          AND m.content_text IS NOT NULL
          AND LENGTH(m.content_text) > 0
    """
    params = []
    if session_key:
        query += " AND m.session_key = ?"
        params.append(session_key)
    query += f" LIMIT {batch_size}"

    messages = conn.execute(query, params).fetchall()

    if not messages:
        return {"messages_enriched": 0, "topics_assigned": 0}

    message_data = [
        {
            "message_id": row[0],
            "session_key": row[1],
            "content_text": row[2],
            "message_type": row[3],
        }
        for row in messages
    ]

    enrichment_results = enrich_func(message_data)

    intent_lookup = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT intent_name, intent_key FROM dim_intent"
        ).fetchall()
    }
    sentiment_lookup = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT sentiment_name, sentiment_key FROM dim_sentiment"
        ).fetchall()
    }
    topic_lookup = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT topic_name, topic_key FROM dim_topic"
        ).fetchall()
    }

    messages_enriched = 0
    topics_assigned = 0
    enriched_at = datetime.now()

    for result in enrichment_results:
        message_id = result.get("message_id")
        if not message_id:
            continue

        msg_session_key = None
        for md in message_data:
            if md["message_id"] == message_id:
                msg_session_key = md["session_key"]
                break

        intent_name = result.get("intent", "question")
        sentiment_name = result.get("sentiment", "neutral")
        intent_key = intent_lookup.get(intent_name, intent_lookup.get("question"))
        sentiment_key = sentiment_lookup.get(
            sentiment_name, sentiment_lookup.get("neutral")
        )

        enrichment_id = generate_dimension_key(message_id, "enrichment")
        conn.execute(
            """INSERT INTO fact_message_enrichment
               (enrichment_id, message_id, session_key, intent_key, sentiment_key,
                complexity_score, confidence_score, enrichment_model, enriched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                enrichment_id,
                message_id,
                msg_session_key,
                intent_key,
                sentiment_key,
                result.get("complexity_score", 0.5),
                result.get("confidence_score", 0.5),
                model_name,
                enriched_at,
            ],
        )
        messages_enriched += 1

        topics = result.get("topics", [])
        for idx, topic_name in enumerate(topics):
            topic_key = topic_lookup.get(topic_name)
            if topic_key:
                message_topic_id = generate_dimension_key(message_id, "topic", str(idx))
                relevance = 1.0 - (idx * 0.1) if idx < 10 else 0.1
                conn.execute(
                    """INSERT INTO fact_message_topics
                       (message_topic_id, message_id, topic_key, relevance_score)
                       VALUES (?, ?, ?, ?)""",
                    [message_topic_id, message_id, topic_key, relevance],
                )
                topics_assigned += 1

    return {"messages_enriched": messages_enriched, "topics_assigned": topics_assigned}


def run_session_insights_enrichment(
    conn,
    insight_func,
    model_name="claude-3-haiku-20240307",
    session_key=None,
):
    """Generate LLM-based insights for sessions.

    Args:
        conn: DuckDB connection
        insight_func: Function(session_data) -> insight dict with:
                     - summary_text: str
                     - key_decisions: str
                     - outcome_status: str (success, partial, failed, unknown)
                     - task_completed: bool
                     - primary_intent: str (should match dim_intent.intent_name)
                     - complexity_score: float (0-1)
        model_name: Name of the model used for enrichment
        session_key: Optional session key to process only one session

    Returns:
        dict with count: sessions_enriched
    """
    query = """
        SELECT s.session_key, s.session_id,
               ss.total_messages, ss.total_tool_calls,
               ss.session_duration_seconds
        FROM dim_session s
        JOIN fact_session_summary ss ON s.session_key = ss.session_key
        LEFT JOIN fact_session_insights i ON s.session_key = i.session_key
        WHERE i.session_key IS NULL
    """
    params = []
    if session_key:
        query += " AND s.session_key = ?"
        params.append(session_key)

    sessions = conn.execute(query, params).fetchall()

    if not sessions:
        return {"sessions_enriched": 0}

    intent_lookup = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT intent_name, intent_key FROM dim_intent"
        ).fetchall()
    }

    sessions_enriched = 0
    enriched_at = datetime.now()

    for row in sessions:
        sess_key = row[0]
        session_id = row[1]
        total_messages = row[2]
        total_tool_calls = row[3]
        duration_seconds = row[4]

        messages = conn.execute(
            """SELECT content_text, mt.message_type
               FROM fact_messages m
               JOIN dim_message_type mt ON m.message_type_key = mt.message_type_key
               WHERE m.session_key = ?
               ORDER BY m.timestamp
               LIMIT 50""",
            [sess_key],
        ).fetchall()

        session_data = {
            "session_key": sess_key,
            "session_id": session_id,
            "total_messages": total_messages,
            "total_tool_calls": total_tool_calls,
            "duration_seconds": duration_seconds,
            "messages": [
                {"content": row[0], "type": row[1]} for row in messages if row[0]
            ],
        }

        insight = insight_func(session_data)

        primary_intent_name = insight.get("primary_intent", "question")
        primary_intent_key = intent_lookup.get(
            primary_intent_name, intent_lookup.get("question")
        )

        insight_id = generate_dimension_key(sess_key, "insight")
        conn.execute(
            """INSERT INTO fact_session_insights
               (insight_id, session_key, summary_text, key_decisions, outcome_status,
                task_completed, primary_intent_key, complexity_score,
                enrichment_model, enriched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                insight_id,
                sess_key,
                insight.get("summary_text", ""),
                insight.get("key_decisions", ""),
                insight.get("outcome_status", "unknown"),
                insight.get("task_completed", False),
                primary_intent_key,
                insight.get("complexity_score", 0.5),
                model_name,
                enriched_at,
            ],
        )
        sessions_enriched += 1

    return {"sessions_enriched": sessions_enriched}


def run_goal_task_enrichment(conn, classify_func, session_key=None):
    """Populate Goal > Task > Attempt hierarchy via LLM classification.

    This function queries sessions that don't yet have goal/task/attempt
    assignments, calls user-provided classify_func for classification,
    and populates dim_goal, dim_task, dim_attempt tables.

    Args:
        conn: DuckDB connection
        classify_func: Function(session_data) -> dict with:
                       - goal: dict with goal_description, goal_status (optional)
                       - task: dict with task_description, task_type, task_status (optional)
                       - attempt: dict with attempt_description, approach, outcome (optional)
                       Each key is optional; only provided keys are processed.
        session_key: Optional session key to process only one session

    Returns:
        dict with counts: goals_created, tasks_created, attempts_created, sessions_linked
    """
    query = """
        SELECT s.session_key, s.session_id,
               s.first_timestamp, s.last_timestamp, s.chain_key,
               ss.total_messages, ss.total_tool_calls,
               ss.session_duration_seconds
        FROM dim_session s
        JOIN fact_session_summary ss ON s.session_key = ss.session_key
        WHERE s.goal_key IS NULL AND s.task_key IS NULL
    """
    params = []
    if session_key:
        query += " AND s.session_key = ?"
        params.append(session_key)

    sessions = conn.execute(query, params).fetchall()

    if not sessions:
        return {
            "goals_created": 0,
            "tasks_created": 0,
            "attempts_created": 0,
            "sessions_linked": 0,
        }

    goals_created = 0
    tasks_created = 0
    attempts_created = 0
    sessions_linked = 0
    now = datetime.now()

    for row in sessions:
        sess_key = row[0]
        session_id = row[1]
        first_ts = row[2]
        last_ts = row[3]
        chain_key = row[4]
        total_messages = row[5]
        total_tool_calls = row[6]
        duration = row[7]

        # Get session messages for classification
        messages = conn.execute(
            """SELECT content_text, mt.message_type
               FROM fact_messages m
               JOIN dim_message_type mt ON m.message_type_key = mt.message_type_key
               WHERE m.session_key = ?
               ORDER BY m.timestamp
               LIMIT 50""",
            [sess_key],
        ).fetchall()

        session_data = {
            "session_key": sess_key,
            "session_id": session_id,
            "total_messages": total_messages,
            "total_tool_calls": total_tool_calls,
            "duration_seconds": duration,
            "messages": [{"content": r[0], "type": r[1]} for r in messages if r[0]],
        }

        result = classify_func(session_data)
        if not result:
            continue

        goal_key = None
        task_key = None
        attempt_key = None

        # Process goal
        goal_data = result.get("goal")
        if goal_data and goal_data.get("goal_description"):
            goal_key = generate_dimension_key(goal_data["goal_description"])
            if not conn.execute(
                "SELECT 1 FROM dim_goal WHERE goal_key = ?", [goal_key]
            ).fetchone():
                conn.execute(
                    """INSERT INTO dim_goal
                       (goal_key, goal_description, goal_status,
                        created_at, completed_at, source)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    [
                        goal_key,
                        goal_data["goal_description"],
                        goal_data.get("goal_status", "active"),
                        now,
                        None,
                        "llm_enrichment",
                    ],
                )
                goals_created += 1

        # Process task
        task_data = result.get("task")
        if task_data and task_data.get("task_description"):
            task_key = generate_dimension_key(task_data["task_description"])
            if not conn.execute(
                "SELECT 1 FROM dim_task WHERE task_key = ?", [task_key]
            ).fetchone():
                conn.execute(
                    """INSERT INTO dim_task
                       (task_key, goal_key, task_description, task_type,
                        task_status, created_at, completed_at, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        task_key,
                        goal_key,
                        task_data["task_description"],
                        task_data.get("task_type", "unknown"),
                        task_data.get("task_status", "active"),
                        now,
                        None,
                        "llm_enrichment",
                    ],
                )
                tasks_created += 1

        # Process attempt
        attempt_data = result.get("attempt")
        if attempt_data and attempt_data.get("attempt_description"):
            attempt_key = generate_dimension_key(
                sess_key, attempt_data["attempt_description"]
            )
            if not conn.execute(
                "SELECT 1 FROM dim_attempt WHERE attempt_key = ?", [attempt_key]
            ).fetchone():
                conn.execute(
                    """INSERT INTO dim_attempt
                       (attempt_key, task_key, session_key, chain_key,
                        attempt_description, approach, outcome,
                        started_at, ended_at, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        attempt_key,
                        task_key,
                        sess_key,
                        chain_key,
                        attempt_data["attempt_description"],
                        attempt_data.get("approach"),
                        attempt_data.get("outcome", "unknown"),
                        first_ts,
                        last_ts,
                        "llm_enrichment",
                    ],
                )
                attempts_created += 1

        # Update session with hierarchy links
        if goal_key or task_key or attempt_key:
            conn.execute(
                """UPDATE dim_session
                   SET goal_key = COALESCE(?, goal_key),
                       task_key = COALESCE(?, task_key),
                       attempt_key = COALESCE(?, attempt_key)
                   WHERE session_key = ?""",
                [goal_key, task_key, attempt_key, sess_key],
            )
            sessions_linked += 1

    return {
        "goals_created": goals_created,
        "tasks_created": tasks_created,
        "attempts_created": attempts_created,
        "sessions_linked": sessions_linked,
    }
