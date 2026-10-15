import array
import json
import logging
import math
import struct
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from memory.db import DatabaseManager

logger = logging.getLogger("JARVIS.MemoryEmbeddings")


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class EmbeddingManager:
    """Local semantic embedding manager for J.A.R.V.I.S. using Ollama embeddinggemma:300m."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        ollama_base_url: Optional[str] = None,
        db_manager: Optional[DatabaseManager] = None,
        timeout: float = 3.0,
    ):
        self.model_name = model_name or getattr(settings, "EMBEDDING_MODEL", "embeddinggemma:300m")
        self.dimension = dimension or getattr(settings, "EMBEDDING_DIMENSION", 768)
        self.ollama_base_url = (ollama_base_url or getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.db_manager = db_manager or DatabaseManager()
        self.timeout = timeout
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Checks if local Ollama embedding service is online."""
        try:
            req = urllib.request.Request(
                f"{self.ollama_base_url}/api/tags",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                # Match full model name or base name (e.g. embeddinggemma:300m or embeddinggemma:latest)
                base = self.model_name.split(":")[0]
                self._available = any(self.model_name in m or base in m for m in models)
                return self._available
        except Exception:
            self._available = False
            return False

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generates a normalized float embedding vector for text using Ollama."""
        clean = text.strip() if text else ""
        if not clean:
            return None

        # 1. Try batch /api/embed endpoint first
        url_embed = f"{self.ollama_base_url}/api/embed"
        payload = json.dumps({"model": self.model_name, "input": [clean]}).encode("utf-8")
        req = urllib.request.Request(url_embed, data=payload, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                embeddings = data.get("embeddings", [])
                if embeddings and len(embeddings) > 0:
                    vec = embeddings[0]
                    if len(vec) == self.dimension:
                        return vec
        except Exception:
            pass

        # 2. Fallback to /api/embeddings legacy endpoint
        url_embeddings = f"{self.ollama_base_url}/api/embeddings"
        payload_legacy = json.dumps({"model": self.model_name, "prompt": clean}).encode("utf-8")
        req_legacy = urllib.request.Request(url_embeddings, data=payload_legacy, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req_legacy, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                vec = data.get("embedding", [])
                if len(vec) == self.dimension:
                    return vec
        except Exception as e:
            logger.debug(f"Ollama embedding request failed for '{self.model_name}': {e}")

        return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Batch generates embeddings for a list of strings."""
        if not texts:
            return []

        clean_texts = [t.strip() for t in texts]
        url = f"{self.ollama_base_url}/api/embed"
        payload = json.dumps({"model": self.model_name, "input": clean_texts}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=max(self.timeout, len(texts) * 1.5)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                embeddings = data.get("embeddings", [])
                if len(embeddings) == len(texts):
                    return embeddings
        except Exception as e:
            logger.debug(f"Batch embedding failed, falling back to sequential: {e}")

        # Sequential fallback
        return [self.embed_text(t) for t in clean_texts]

    def store_embedding(self, memory_id: str, vector: List[float]):
        """Persists a float vector into SQLite as a compact binary BLOB."""
        if not vector or len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")

        # Pack floats into binary format (little-endian single precision)
        packed_blob = struct.pack(f"{len(vector)}f", *vector)
        now = datetime.now(timezone.utc).isoformat()

        with self.db_manager.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_embeddings (
                    memory_id, model, dimensions, vector, updated_at
                ) VALUES (?, ?, ?, ?, ?);
                """,
                (memory_id, self.model_name, len(vector), packed_blob, now),
            )

    def get_embedding(self, memory_id: str) -> Optional[List[float]]:
        """Retrieves and unpacks a vector from SQLite by memory ID."""
        with self.db_manager.connection() as conn:
            row = conn.execute(
                "SELECT vector, dimensions FROM memory_embeddings WHERE memory_id = ?;",
                (memory_id,),
            ).fetchone()

            if not row or not row["vector"]:
                return None

            blob = row["vector"]
            dim = row["dimensions"]
            try:
                floats = list(struct.unpack(f"{dim}f", blob))
                return floats
            except Exception as e:
                logger.error(f"Error unpacking vector blob for memory '{memory_id}': {e}")
                return None

    def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 5,
        layer: Optional[str] = None,
        category: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Computes cosine similarity against all active stored embeddings and returns top-K matches."""
        if not query_vector:
            return []

        sql = """
            SELECT e.memory_id, e.vector, e.dimensions
            FROM memory_embeddings e
            JOIN memories m ON e.memory_id = m.id
            WHERE m.status = 'active'
        """
        params: List[Any] = []
        if layer:
            sql += " AND m.layer = ?"
            params.append(layer)
        if category:
            sql += " AND m.category = ?"
            params.append(category.lower())
        if project_id:
            sql += " AND m.project_id = ?"
            params.append(project_id)

        scored: List[Tuple[str, float]] = []

        with self.db_manager.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            for r in rows:
                blob = r["vector"]
                dim = r["dimensions"]
                try:
                    vec = struct.unpack(f"{dim}f", blob)
                    sim = cosine_similarity(query_vector, vec)
                    scored.append((r["memory_id"], sim))
                except Exception:
                    continue

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def sync_all_unembedded(self, batch_size: int = 20) -> int:
        """Finds active memories missing embeddings and generates them."""
        with self.db_manager.connection() as conn:
            missing_rows = conn.execute(
                """
                SELECT m.id, m.content
                FROM memories m
                LEFT JOIN memory_embeddings e ON m.id = e.memory_id
                WHERE m.status = 'active' AND e.memory_id IS NULL
                LIMIT ?;
                """,
                (batch_size,),
            ).fetchall()

        if not missing_rows:
            return 0

        synced = 0
        for row in missing_rows:
            vec = self.embed_text(row["content"])
            if vec:
                self.store_embedding(row["id"], vec)
                synced += 1

        logger.info(f"Synchronized {synced} missing memory embeddings.")
        return synced

    @staticmethod
    def calculate_rrf(
        fts_ids: Optional[List[str]] = None,
        vec_ids: Optional[List[str]] = None,
        k: int = 60,
        weight_fts: float = 0.5,
        weight_vec: float = 0.5,
        **kwargs: Any,
    ) -> List[Tuple[str, float]]:
        """Computes Reciprocal Rank Fusion (RRF) scores across FTS5 and vector rankings.
        
        Score(d) = sum(weight / (k + rank))
        """
        fts_list = fts_ids if fts_ids is not None else kwargs.get("fts_ranks", [])
        vec_list = vec_ids if vec_ids is not None else kwargs.get("vec_ranks", [])
        rrf_scores: Dict[str, float] = {}

        # 1. Process FTS5 ranks (1-indexed)
        for rank_idx, doc_id in enumerate(fts_list, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (weight_fts / (k + rank_idx))

        # 2. Process Vector ranks (1-indexed)
        for rank_idx, doc_id in enumerate(vec_list, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (weight_vec / (k + rank_idx))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs
