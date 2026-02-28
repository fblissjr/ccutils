"""ColBERT embedding pipeline for semantic matching.

Uses mxbai-edge-colbert-v0-32m (32M params, 64-dim projections)
via PyLate for late-interaction retrieval.

Usage:
    from ccutils.schemas.star.embeddings import EmbeddingPipeline

    pipeline = EmbeddingPipeline()  # lazy model loading
    pipeline.embed_sessions(conn)   # embed session content
    pipeline.match_delegations(conn)  # improve agent-delegation matching
    pipeline.cluster_sessions(conn)   # cluster sessions by similarity
"""

import hashlib
from datetime import datetime

from .utils import generate_dimension_key


def _check_pylate():
    """Check if pylate is available."""
    try:
        import pylate  # noqa: F401

        return True
    except ImportError:
        return False


class EmbeddingPipeline:
    """ColBERT embedding pipeline for semantic matching across sessions.

    Uses lazy model loading -- the model is only loaded when first needed.
    Requires the 'colbert' optional dependency: pip install ccutils[colbert]

    Args:
        model_name: HuggingFace model ID for ColBERT model
    """

    DEFAULT_MODEL = "mixedbread-ai/mxbai-edge-colbert-v0-32m"
    EMBEDDING_DIM = 64

    def __init__(self, model_name=None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    @property
    def model(self):
        """Lazy-load the ColBERT model."""
        if self._model is None:
            if not _check_pylate():
                raise ImportError(
                    "pylate is required for the embedding pipeline. "
                    "Install it with: uv add ccutils[colbert]"
                )
            from pylate import models

            self._model = models.ColBERT(self.model_name)
        return self._model

    def embed_sessions(self, conn, content_type="first_user_message", batch_size=32):
        """Embed session content into fact_session_embeddings.

        Args:
            conn: DuckDB connection
            content_type: What to embed - 'summary' or 'first_user_message'
            batch_size: Number of sessions to process at once

        Returns:
            dict with count: sessions_embedded
        """
        # Find sessions not yet embedded for this content_type
        existing = conn.execute(
            """SELECT session_key FROM fact_session_embeddings
               WHERE content_type = ? AND embedding_model = ?""",
            [content_type, self.model_name],
        ).fetchall()
        existing_keys = {r[0] for r in existing}

        # Always use first_user_message (summary mode removed --
        # it depended on LLM enrichment tables that no longer exist)
        sessions = conn.execute(
            """SELECT fm.session_key, fm.content_text
               FROM fact_messages fm
               WHERE fm.message_type = 'user'
                 AND fm.content_text IS NOT NULL
                 AND LENGTH(fm.content_text) > 0
               ORDER BY fm.timestamp"""
        ).fetchall()
        # Take first user message per session
        texts = {}
        for r in sessions:
            if r[0] not in existing_keys and r[0] not in texts:
                texts[r[0]] = r[1]

        if not texts:
            return {"sessions_embedded": 0}

        session_keys = list(texts.keys())
        session_texts = [texts[k] for k in session_keys]

        # Encode in batches
        embedded_count = 0
        now = datetime.now()

        for i in range(0, len(session_texts), batch_size):
            batch_keys = session_keys[i : i + batch_size]
            batch_texts = session_texts[i : i + batch_size]

            # Get multi-vector embeddings then mean-pool
            embeddings = self.model.encode(batch_texts, is_query=False)

            for j, (sk, emb) in enumerate(zip(batch_keys, embeddings)):
                # Mean-pool the multi-vector embedding to a single vector
                import numpy as np

                mean_emb = np.mean(emb, axis=0).tolist()

                # Pad or truncate to EMBEDDING_DIM
                if len(mean_emb) < self.EMBEDDING_DIM:
                    mean_emb.extend([0.0] * (self.EMBEDDING_DIM - len(mean_emb)))
                else:
                    mean_emb = mean_emb[: self.EMBEDDING_DIM]

                content_hash = hashlib.md5(batch_texts[j].encode("utf-8")).hexdigest()

                emb_key = generate_dimension_key(sk, content_type, self.model_name)

                conn.execute(
                    """INSERT INTO fact_session_embeddings
                       (embedding_key, session_key, content_type,
                        embedding_model, embedding_dim, mean_embedding,
                        embedded_at, content_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        emb_key,
                        sk,
                        content_type,
                        self.model_name,
                        self.EMBEDDING_DIM,
                        mean_emb,
                        now,
                        content_hash,
                    ],
                )
                embedded_count += 1

        return {"sessions_embedded": embedded_count}

    def match_delegations(self, conn):
        """Re-score agent delegation matches using semantic similarity.

        Compares Task prompt embeddings with agent session content
        to improve match_confidence scores in fact_agent_delegations.

        Returns:
            dict with count: delegations_rescored
        """
        delegations = conn.execute(
            """SELECT fad.delegation_key, fad.task_prompt, fad.agent_session_key
               FROM fact_agent_delegations fad
               WHERE fad.task_prompt IS NOT NULL
                 AND LENGTH(fad.task_prompt) > 0"""
        ).fetchall()

        if not delegations:
            return {"delegations_rescored": 0}

        rescored = 0

        for deleg in delegations:
            deleg_key = deleg[0]
            task_prompt = deleg[1]
            agent_sk = deleg[2]

            # Get agent session's first user message
            agent_msg = conn.execute(
                """SELECT fm.content_text
                   FROM fact_messages fm
                   WHERE fm.session_key = ? AND fm.message_type = 'user'
                     AND fm.content_text IS NOT NULL
                   ORDER BY fm.timestamp LIMIT 1""",
                [agent_sk],
            ).fetchone()

            if not agent_msg or not agent_msg[0]:
                continue

            # Compute similarity using ColBERT late interaction
            query_emb = self.model.encode([task_prompt], is_query=True)
            doc_emb = self.model.encode([agent_msg[0]], is_query=False)

            from pylate import scores

            similarity = scores.colbert_scores(query_emb, doc_emb)
            score = float(similarity[0][0])

            # Normalize to 0-1 range and update
            confidence = min(max(score, 0.0), 1.0)

            conn.execute(
                """UPDATE fact_agent_delegations
                   SET match_confidence = ?
                   WHERE delegation_key = ?""",
                [confidence, deleg_key],
            )
            rescored += 1

        return {"delegations_rescored": rescored}

    def cluster_sessions(self, conn, n_clusters=None):
        """Cluster sessions by content similarity using embeddings.

        Uses mean-pooled embeddings from fact_session_embeddings
        to cluster sessions. Assigns cluster IDs as candidate task_key values.

        Args:
            conn: DuckDB connection
            n_clusters: Number of clusters (auto-detected if None)

        Returns:
            dict with counts: sessions_clustered, clusters_found
        """
        # Get all session embeddings
        rows = conn.execute(
            """SELECT session_key, mean_embedding
               FROM fact_session_embeddings
               WHERE mean_embedding IS NOT NULL"""
        ).fetchall()

        if len(rows) < 2:
            return {"sessions_clustered": 0, "clusters_found": 0}

        import numpy as np

        session_keys = [r[0] for r in rows]
        embeddings = np.array([list(r[1]) for r in rows])

        # Determine number of clusters
        if n_clusters is None:
            n_clusters = max(2, min(len(rows) // 5, 20))

        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
        except ImportError:
            # Fallback: no clustering without sklearn
            return {"sessions_clustered": 0, "clusters_found": 0}

        clustered = 0

        for cluster_id in range(n_clusters):
            cluster_sessions = [
                session_keys[i] for i, l in enumerate(labels) if l == cluster_id
            ]
            if not cluster_sessions:
                continue

            cluster_label = f"cluster_{cluster_id}"

            for sk in cluster_sessions:
                conn.execute(
                    """UPDATE dim_session
                       SET domain = COALESCE(domain, ?)
                       WHERE session_key = ? AND (domain IS NULL OR domain = 'unknown')""",
                    [cluster_label, sk],
                )
                clustered += 1

        return {"sessions_clustered": clustered, "clusters_found": n_clusters}
