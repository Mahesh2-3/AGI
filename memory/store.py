"""Persistent, multi-layer memory store for Project J.A.R.V.I.S.

Backed by SQLite with WAL mode, FTS5 full-text indexing, ACID transactions,
fact supersession, deduplication, and backward-compatible JSON file support.
"""

import concurrent.futures
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional, Set
import uuid

from config.settings import settings
from memory.db import DatabaseManager, db_manager as default_db_manager
from memory.embeddings import EmbeddingManager

logger = logging.getLogger("JARVIS.MemoryStore")


@dataclass
class MemoryEntry:
    """Unified record across all five memory layers (working, semantic, episodic, project, procedural)."""

    id: str
    content: str
    category: str = "fact"  # "fact", "preference", "profile", "task", "correction", "event", "procedure"
    tags: List[str] = field(default_factory=list)
    importance: int = 3  # 1 (low) to 5 (critical)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    access_count: int = 0
    last_accessed: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Multi-layer architectural attributes
    layer: str = "semantic"  # "working", "semantic", "episodic", "project", "procedural"
    subject: Optional[str] = None
    predicate: Optional[str] = None
    project_id: Optional[str] = None
    confidence: float = 1.0
    status: str = "active"  # "active", "superseded", "expired", "archived"
    source: str = "user"  # "user", "agent_inference", "tool_output", "verified_fix"
    provenance: Optional[str] = None
    expires_at: Optional[str] = None
    superseded_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        tags_raw = data.get("tags", [])
        if isinstance(tags_raw, str):
            try:
                tags = json.loads(tags_raw)
            except Exception:
                tags = [tags_raw]
        else:
            tags = list(tags_raw) if tags_raw else []

        return cls(
            id=data.get("id", f"mem_{uuid.uuid4().hex[:8]}"),
            content=data.get("content", ""),
            category=data.get("category", "fact"),
            tags=tags,
            importance=int(data.get("importance", 3)),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            access_count=int(data.get("access_count", 0)),
            last_accessed=data.get("last_accessed", datetime.now(timezone.utc).isoformat()),
            layer=data.get("layer", "semantic"),
            subject=data.get("subject"),
            predicate=data.get("predicate"),
            project_id=data.get("project_id"),
            confidence=float(data.get("confidence", 1.0)),
            status=data.get("status", "active"),
            source=data.get("source", "user"),
            provenance=data.get("provenance"),
            expires_at=data.get("expires_at"),
            superseded_by=data.get("superseded_by"),
        )


class MemoryStore:
    """Production-grade multi-layer memory store for J.A.R.V.I.S.
    
    Persists to SQLite with WAL mode and FTS5 full-text search. Maintains an in-memory
    cache and JSON synchronization when configured with a JSON path for backward compatibility.
    """

    def __init__(
        self,
        db_path: Optional[Path | str] = None,
        file_path: Optional[Path | str] = None,
    ):
        self._lock = threading.RLock()
        is_default_db = (db_path is None and file_path is None)

        # Handle backward compatibility: file_path passed by tests or legacy configs
        if file_path:
            self.file_path = Path(file_path)
            # If a JSON path is given, derive a SQLite database alongside it
            if self.file_path.suffix.lower() == ".json":
                self.db_path = self.file_path.with_suffix(".db")
            else:
                self.db_path = self.file_path
        elif db_path:
            self.db_path = Path(db_path)
            self.file_path = None
        else:
            self.db_path = settings.DATABASE_PATH
            self.file_path = settings.MEMORY_FILE_PATH

        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.embeddings = EmbeddingManager(db_manager=self.db_manager)
        self._embed_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jarvis_embed_worker",
        )
        self.memories: Dict[str, MemoryEntry] = {}
        self.profile: Dict[str, Any] = {}

        # Auto-migrate legacy JSON memory if default DB or explicit JSON file path provided
        if is_default_db and self.file_path and self.file_path.exists():
            self.db_manager.migrate_from_json_if_needed(self.file_path)
        elif self.file_path and self.file_path.exists() and self.file_path.suffix.lower() == ".json":
            self.db_manager.migrate_from_json_if_needed(self.file_path)

        self._load()

    def _async_embed_memory(self, memory_id: str, content: str):
        """Asynchronously computes and persists vector embedding in background."""
        if hasattr(self, "_embed_executor") and self.embeddings:
            def _task():
                try:
                    vec = self.embeddings.embed_text(content)
                    if vec:
                        self.embeddings.store_embedding(memory_id, vec)
                except Exception as ex:
                    logger.debug(f"Async embedding failed for memory '{memory_id}': {ex}")

            self._embed_executor.submit(_task)

    def _load(self):
        """Loads memories and profile attributes from SQLite into the in-memory cache."""
        with self._lock:
            try:
                with self.db_manager.connection() as conn:
                    # 1. Load user profile
                    prof_rows = conn.execute("SELECT key, value FROM user_profile;").fetchall()
                    self.profile = {r["key"]: r["value"] for r in prof_rows}

                    # 2. Load active memories
                    mem_rows = conn.execute(
                        "SELECT * FROM memories WHERE status = 'active' ORDER BY importance DESC, updated_at DESC;"
                    ).fetchall()
                    self.memories = {}
                    for row in mem_rows:
                        entry = MemoryEntry.from_dict(dict(row))
                        self.memories[entry.id] = entry

                # If backed by a JSON file path in legacy mode, sync JSON on load
                self._sync_legacy_json()
            except Exception as e:
                logger.error(f"Error loading memories from database: {e}")

    def _sync_legacy_json(self):
        """Writes cache to legacy JSON file if file_path is configured with a .json extension."""
        if not self.file_path or self.file_path.suffix.lower() != ".json":
            return
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": "1.0",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "profile": self.profile,
                "memories": [entry.to_dict() for entry in self.memories.values()],
            }
            tmp = self.file_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.file_path)
        except Exception as e:
            logger.debug(f"Could not sync legacy JSON file: {e}")

    def _tokenize(self, text: str) -> Set[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())
        stop_words = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
            "is", "was", "are", "were", "of", "it", "that", "this", "my", "your",
            "me", "you", "i", "sir", "jarvis", "please", "be", "have", "has", "do"
        }
        return {w for w in words if len(w) > 1 and w not in stop_words}

    def _sanitize_fts_query(self, query: str) -> str:
        """Converts user natural language search into safe FTS5 query syntax."""
        tokens = re.findall(r"[a-zA-Z0-9_\.]+", query.lower())
        valid_tokens = [t for t in tokens if len(t) > 1 and t not in {"and", "or", "not"}]
        if not valid_tokens:
            return ""
        # Match any token with prefix expansion
        return " OR ".join(f'"{t}"*' for t in valid_tokens)

    def remember(
        self,
        content: str,
        category: str = "fact",
        tags: Optional[List[str]] = None,
        importance: int = 3,
        layer: str = "semantic",
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        project_id: Optional[str] = None,
        confidence: float = 1.0,
        source: str = "user",
        provenance: Optional[str] = None,
        expires_at: Optional[str] = None,
    ) -> MemoryEntry:
        """Stores a new fact, preference, or instruction into persistent memory.
        
        Performs deduplication and fact supersession:
        - If an identical memory exists, updates it cleanly.
        - If a memory with the same subject and predicate exists with a conflicting value,
          marks the older memory as superseded.
        """
        with self._lock:
            clean_content = content.strip()
            if not clean_content:
                raise ValueError("Memory content cannot be empty.")

            clean_tags = [t.strip().lower() for t in (tags or []) if t.strip()]
            importance = max(1, min(5, int(importance)))
            now = datetime.now(timezone.utc).isoformat()

            # Automatic profile extraction for common user attributes
            lower = clean_content.lower()
            name_match = re.search(r"\b(?:my name is|call me|i am|i'm|user(?:'s)? name is)\s+([a-zA-Z]+)\b", lower)
            if name_match:
                name = name_match.group(1).capitalize()
                self.set_profile_attribute("name", name)
                category = "profile"
                importance = 5
                subject = subject or "user"
                predicate = predicate or "name"
                if "name" not in clean_tags:
                    clean_tags.append("name")

            # Detect key-value preference patterns (e.g. "User preference terminal: kitty")
            pref_match = re.search(r"\buser preference ([a-zA-Z0-9_]+):\s*(.+)$", lower)
            if pref_match:
                p_key = pref_match.group(1).strip()
                p_val = pref_match.group(2).strip()
                subject = subject or "user"
                predicate = predicate or p_key
                category = "preference"

            new_id = f"mem_{uuid.uuid4().hex[:8]}"

            # 1. Deduplication check: by exact content or high token overlap
            content_tokens = self._tokenize(clean_content)
            existing_match: Optional[MemoryEntry] = None

            for entry in list(self.memories.values()):
                if entry.status != "active":
                    continue
                if entry.content.lower() == clean_content.lower():
                    existing_match = entry
                    break
                entry_tokens = self._tokenize(entry.content)
                if content_tokens and entry_tokens:
                    intersection = len(content_tokens & entry_tokens)
                    union = len(content_tokens | entry_tokens)
                    if union > 0 and (intersection / union) > 0.8:
                        existing_match = entry
                        break

            if existing_match:
                # Update existing record
                merged_tags = list(set(existing_match.tags + clean_tags))
                existing_match.content = clean_content
                existing_match.category = category
                existing_match.importance = max(existing_match.importance, importance)
                existing_match.updated_at = now
                existing_match.tags = merged_tags
                if subject:
                    existing_match.subject = subject
                if predicate:
                    existing_match.predicate = predicate
                if project_id:
                    existing_match.project_id = project_id

                with self.db_manager.connection() as conn:
                    conn.execute(
                        """
                        UPDATE memories
                        SET content = ?, category = ?, tags = ?, importance = ?,
                            updated_at = ?, subject = coalesce(?, subject),
                            predicate = coalesce(?, predicate), project_id = coalesce(?, project_id)
                        WHERE id = ?;
                        """,
                        (
                            clean_content,
                            category,
                            json.dumps(merged_tags),
                            existing_match.importance,
                            now,
                            subject,
                            predicate,
                            project_id,
                            existing_match.id,
                        ),
                    )
                self._sync_legacy_json()
                self._async_embed_memory(existing_match.id, existing_match.content)
                return existing_match

            # 2. Create and insert fresh record
            entry = MemoryEntry(
                id=new_id,
                content=clean_content,
                category=category,
                tags=clean_tags,
                importance=importance,
                created_at=now,
                updated_at=now,
                access_count=0,
                last_accessed=now,
                layer=layer,
                subject=subject,
                predicate=predicate,
                project_id=project_id,
                confidence=confidence,
                status="active",
                source=source,
                provenance=provenance,
                expires_at=expires_at,
            )

            with self.db_manager.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO memories (
                        id, layer, category, subject, predicate, content, tags,
                        project_id, importance, confidence, status, source, provenance,
                        created_at, updated_at, last_accessed, access_count, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        entry.id,
                        entry.layer,
                        entry.category,
                        entry.subject,
                        entry.predicate,
                        entry.content,
                        json.dumps(entry.tags),
                        entry.project_id,
                        entry.importance,
                        entry.confidence,
                        entry.status,
                        entry.source,
                        entry.provenance,
                        entry.created_at,
                        entry.updated_at,
                        entry.last_accessed,
                        entry.access_count,
                        entry.expires_at,
                    ),
                )

                # 3. Fact Supersession: check if conflicting active record exists for same subject & predicate
                # Safe to set superseded_by = entry.id now that entry.id is inserted
                if subject and predicate:
                    existing_preds = conn.execute(
                        """
                        SELECT id, content FROM memories
                        WHERE subject = ? AND predicate = ? AND status = 'active' AND id != ?;
                        """,
                        (subject, predicate, entry.id),
                    ).fetchall()
                    for old_row in existing_preds:
                        if old_row["content"].strip().lower() != clean_content.lower():
                            conn.execute(
                                """
                                UPDATE memories
                                SET status = 'superseded', superseded_by = ?, updated_at = ?
                                WHERE id = ?;
                                """,
                                (entry.id, now, old_row["id"]),
                            )
                            if old_row["id"] in self.memories:
                                self.memories.pop(old_row["id"], None)
                            logger.info(f"Superseded older memory '{old_row['id']}' with '{entry.id}'")

            self.memories[entry.id] = entry
            self._sync_legacy_json()
            self._async_embed_memory(entry.id, entry.content)
            return entry

    def recall(
        self,
        query: str,
        category: Optional[str] = None,
        project_id: Optional[str] = None,
        layer: Optional[str] = None,
        limit: int = 5,
        hybrid: bool = False,
    ) -> List[MemoryEntry]:
        """Searches persistent memory using fast FTS5 BM25 or hybrid Reciprocal Rank Fusion (FTS5 + Vector),
        falling back gracefully to token containment if needed."""
        with self._lock:
            q_clean = query.strip()
            now = datetime.now(timezone.utc).isoformat()
            if not q_clean:
                return self.list_all(category=category, project_id=project_id, limit=limit)

            fts_query = self._sanitize_fts_query(q_clean)
            fts_results: List[MemoryEntry] = []

            # 1. Primary: SQLite FTS5 Full-Text Match
            if fts_query:
                try:
                    sql = """
                        SELECT m.*, bm25(memories_fts) as rank
                        FROM memories_fts f
                        JOIN memories m ON f.id = m.id
                        WHERE memories_fts MATCH ?
                          AND m.status = 'active'
                    """
                    params: List[Any] = [fts_query]

                    if category:
                        sql += " AND m.category = ?"
                        params.append(category.lower())
                    if project_id:
                        sql += " AND m.project_id = ?"
                        params.append(project_id)
                    if layer:
                        sql += " AND m.layer = ?"
                        params.append(layer)

                    sql += " ORDER BY rank ASC, m.importance DESC LIMIT ?;"
                    params.append(limit * 2 if hybrid else limit)

                    with self.db_manager.connection() as conn:
                        rows = conn.execute(sql, params).fetchall()
                        for row in rows:
                            fts_results.append(MemoryEntry.from_dict(dict(row)))
                except Exception as e:
                    logger.debug(f"FTS5 query failed, falling back to LIKE: {e}")

            results: List[MemoryEntry] = []

            # 2. Hybrid Search (RRF): Fuse FTS5 and Semantic Vector Embeddings
            if hybrid and self.embeddings:
                try:
                    query_vec = self.embeddings.embed_text(q_clean)
                    if query_vec:
                        vec_matches = self.embeddings.search_similar(
                            query_vector=query_vec,
                            top_k=limit * 2,
                            layer=layer,
                            category=category,
                            project_id=project_id,
                        )
                        fts_ids = [m.id for m in fts_results]
                        vec_ids = [m[0] for m in vec_matches]

                        if vec_ids or fts_ids:
                            fused = self.embeddings.calculate_rrf(fts_ids=fts_ids, vec_ids=vec_ids)
                            seen_ids = set()
                            for mem_id, _score in fused:
                                if mem_id not in seen_ids:
                                    entry = self.get(mem_id)
                                    if entry and entry.status == "active":
                                        results.append(entry)
                                        seen_ids.add(mem_id)
                                        if len(results) >= limit:
                                            break
                except Exception as ex:
                    logger.debug(f"Hybrid vector search encountered issue, falling back: {ex}")

            # If hybrid did not yield results (e.g. no vectors or embeddings unavailable), use FTS5 results
            if not results and fts_results:
                results = fts_results[:limit]

            # 3. Fallback: Python/SQL token and substring containment if FTS returned nothing
            if not results:
                q_lower = q_clean.lower()
                q_tokens = self._tokenize(q_lower)
                scored: List[Tuple[float, MemoryEntry]] = []

                for entry in self.memories.values():
                    if entry.status != "active":
                        continue
                    if category and entry.category.lower() != category.lower():
                        continue
                    if project_id and entry.project_id != project_id:
                        continue
                    if layer and entry.layer != layer:
                        continue

                    score = 0.0
                    entry_lower = entry.content.lower()

                    if q_lower in entry_lower:
                        score += 15.0

                    entry_tokens = self._tokenize(entry.content)
                    if q_tokens and entry_tokens:
                        overlap = len(q_tokens & entry_tokens)
                        score += overlap * 3.0

                    for tag in entry.tags:
                        if tag in q_tokens or tag in q_lower:
                            score += 5.0

                    if score > 0:
                        score *= (1.0 + (entry.importance - 1) * 0.25)
                        scored.append((score, entry))

                scored.sort(key=lambda x: x[0], reverse=True)
                results = [item[1] for item in scored[:limit]]

            # 3. Update access tracking in SQLite & cache
            if results:
                with self.db_manager.connection() as conn:
                    for res in results:
                        res.access_count += 1
                        res.last_accessed = now
                        conn.execute(
                            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?;",
                            (now, res.id),
                        )
                self._sync_legacy_json()

            return results

    def recall_hybrid(
        self,
        query: str,
        category: Optional[str] = None,
        project_id: Optional[str] = None,
        layer: Optional[str] = None,
        limit: int = 5,
    ) -> List[MemoryEntry]:
        """Performs hybrid Reciprocal Rank Fusion (FTS5 + Vector) semantic retrieval."""
        return self.recall(
            query=query,
            category=category,
            project_id=project_id,
            layer=layer,
            limit=limit,
            hybrid=True,
        )

    def list_all(
        self,
        category: Optional[str] = None,
        project_id: Optional[str] = None,
        layer: Optional[str] = None,
        status: str = "active",
        limit: int = 50,
    ) -> List[MemoryEntry]:
        """Returns stored memories sorted by importance and update date."""
        with self._lock:
            sql = "SELECT * FROM memories WHERE status = ?"
            params: List[Any] = [status]

            if category:
                sql += " AND lower(category) = ?"
                params.append(category.lower())
            if project_id:
                sql += " AND project_id = ?"
                params.append(project_id)
            if layer:
                sql += " AND layer = ?"
                params.append(layer)

            sql += " ORDER BY importance DESC, updated_at DESC LIMIT ?;"
            params.append(limit)

            with self.db_manager.connection() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [MemoryEntry.from_dict(dict(r)) for r in rows]

    def forget(self, query_or_id: str) -> Optional[MemoryEntry]:
        """Deletes or archives a memory matching ID or content substring."""
        with self._lock:
            target_key = query_or_id.strip()
            removed: Optional[MemoryEntry] = None

            # 1. Exact ID match
            if target_key in self.memories:
                removed = self.memories.pop(target_key)
            else:
                # 2. Substring match
                target_lower = target_key.lower()
                for k, entry in list(self.memories.items()):
                    if target_lower in entry.content.lower():
                        removed = self.memories.pop(k)
                        break

            if removed:
                with self.db_manager.connection() as conn:
                    conn.execute("DELETE FROM memories WHERE id = ?;", (removed.id,))
                self._sync_legacy_json()
                return removed

            # Check database directly if not in memory cache
            with self.db_manager.connection() as conn:
                row = conn.execute("SELECT * FROM memories WHERE id = ?;", (target_key,)).fetchone()
                if not row:
                    row = conn.execute(
                        "SELECT * FROM memories WHERE lower(content) LIKE ? ORDER BY CASE WHEN status = 'active' THEN 0 ELSE 1 END LIMIT 1;",
                        (f"%{target_key.lower()}%",),
                    ).fetchone()

                if row:
                    entry = MemoryEntry.from_dict(dict(row))
                    conn.execute("DELETE FROM memories WHERE id = ?;", (entry.id,))
                    self.memories.pop(entry.id, None)
                    self._sync_legacy_json()
                    return entry

            return None

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieves a single memory entry by ID from cache or SQLite."""
        with self._lock:
            if memory_id in self.memories:
                return self.memories[memory_id]
            with self.db_manager.connection() as conn:
                row = conn.execute("SELECT * FROM memories WHERE id = ?;", (memory_id,)).fetchone()
                if row:
                    return MemoryEntry.from_dict(dict(row))
            return None

    def set_profile_attribute(self, key: str, value: Any):
        """Sets a persistent user profile attribute in SQLite."""
        with self._lock:
            clean_k = str(key).strip().lower().replace(" ", "_")
            str_val = str(value)
            now = datetime.now(timezone.utc).isoformat()
            self.profile[clean_k] = str_val

            with self.db_manager.connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO user_profile (key, value, updated_at) VALUES (?, ?, ?);",
                    (clean_k, str_val, now),
                )
            self._sync_legacy_json()

    def get_profile_attribute(self, key: str, default: Any = None) -> Any:
        with self._lock:
            clean_k = str(key).strip().lower().replace(" ", "_")
            return self.profile.get(clean_k, default)

    def get_profile(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.profile)

    def clear(self, category: Optional[str] = None) -> int:
        """Clears memories, optionally scoped to category."""
        with self._lock:
            with self.db_manager.connection() as conn:
                if category:
                    rows = conn.execute(
                        "SELECT id FROM memories WHERE lower(category) = ?;",
                        (category.lower(),),
                    ).fetchall()
                    count = len(rows)
                    conn.execute("DELETE FROM memories WHERE lower(category) = ?;", (category.lower(),))
                    for r in rows:
                        self.memories.pop(r["id"], None)
                else:
                    count_row = conn.execute("SELECT count(*) as c FROM memories;").fetchone()
                    count = count_row["c"] if count_row else 0
                    conn.execute("DELETE FROM memories;")
                    self.memories.clear()

            self._sync_legacy_json()
            return count

    def save_procedure(
        self,
        title: str,
        trigger_intent: str,
        steps: List[str],
        expected_outcome: str,
        preconditions: Optional[List[str]] = None,
        verification_method: Optional[str] = None,
        failure_modes: Optional[List[str]] = None,
        confidence: float = 0.8,
        verified: bool = False,
    ) -> Dict[str, Any]:
        """Saves a reusable procedural workflow or fix into the procedures table."""
        with self._lock:
            proc_id = f"proc_{uuid.uuid4().hex[:8]}"
            now = datetime.now(timezone.utc).isoformat()
            precond_json = json.dumps(preconditions or [])
            steps_json = json.dumps(steps)
            fail_json = json.dumps(failure_modes or [])

            # Check if an existing procedure with the same trigger_intent or title exists
            with self.db_manager.connection() as conn:
                existing = conn.execute(
                    "SELECT id FROM procedures WHERE lower(trigger_intent) = ? OR lower(title) = ? LIMIT 1;",
                    (trigger_intent.strip().lower(), title.strip().lower()),
                ).fetchone()

                if existing:
                    proc_id = existing["id"]
                    conn.execute(
                        """
                        UPDATE procedures
                        SET title = ?, trigger_intent = ?, preconditions = ?, steps = ?,
                            expected_outcome = ?, verification_method = ?, failure_modes = ?,
                            confidence = ?, verified = ?, updated_at = ?
                        WHERE id = ?;
                        """,
                        (
                            title.strip(),
                            trigger_intent.strip(),
                            precond_json,
                            steps_json,
                            expected_outcome.strip(),
                            verification_method,
                            fail_json,
                            confidence,
                            1 if verified else 0,
                            now,
                            proc_id,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO procedures (
                            id, title, trigger_intent, preconditions, steps,
                            expected_outcome, verification_method, failure_modes,
                            confidence, verified, success_count, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            proc_id,
                            title.strip(),
                            trigger_intent.strip(),
                            precond_json,
                            steps_json,
                            expected_outcome.strip(),
                            verification_method,
                            fail_json,
                            confidence,
                            1 if verified else 0,
                            0,
                            now,
                            now,
                        ),
                    )

            # Also create/update a procedural memory entry in memories table so it is searchable via FTS5
            content = f"Procedure: {title}. Trigger: {trigger_intent}. Steps: {'; '.join(steps)}"
            self.remember(
                content=content,
                category="procedure",
                layer="procedural",
                subject=trigger_intent,
                predicate="workflow",
                tags=["procedure", trigger_intent],
            )

            return {
                "id": proc_id,
                "title": title,
                "trigger_intent": trigger_intent,
                "steps": steps,
                "expected_outcome": expected_outcome,
                "verified": verified,
            }

    def recall_procedures(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Recalls procedures matching intent, title, or keywords."""
        with self._lock:
            q = query.strip().lower()
            with self.db_manager.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM procedures
                    WHERE lower(trigger_intent) LIKE ? OR lower(title) LIKE ?
                    ORDER BY verified DESC, success_count DESC, confidence DESC
                    LIMIT ?;
                    """,
                    (f"%{q}%", f"%{q}%", limit),
                ).fetchall()
                results = []
                for r in rows:
                    item = dict(r)
                    item["preconditions"] = json.loads(item["preconditions"])
                    item["steps"] = json.loads(item["steps"])
                    item["failure_modes"] = json.loads(item["failure_modes"])
                    results.append(item)
                return results

    def record_procedure_result(self, proc_id: str, success: bool):
        """Records the outcome of executing a procedure, updating success metrics."""
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            with self.db_manager.connection() as conn:
                if success:
                    conn.execute(
                        """
                        UPDATE procedures
                        SET success_count = success_count + 1, verified = 1, last_executed = ?, updated_at = ?
                        WHERE id = ?;
                        """,
                        (now, now, proc_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE procedures
                        SET confidence = max(0.1, confidence - 0.1), last_executed = ?, updated_at = ?
                        WHERE id = ?;
                        """,
                        (now, now, proc_id),
                    )

    def get_context_summary(
        self,
        max_items: int = 5,
        project_id: Optional[str] = None,
    ) -> str:
        """Constructs a compact, structured context block with safety boundaries
        suitable for direct injection into system prompt context."""
        with self._lock:
            lines = []

            # 1. Profile attributes
            if self.profile:
                prof_items = []
                if "name" in self.profile:
                    prof_items.append(f"Name: {self.profile['name']}")
                for k, v in self.profile.items():
                    if k != "name":
                        prof_items.append(f"{k.capitalize()}: {v}")
                if prof_items:
                    lines.append(f"• User Profile: {', '.join(prof_items)}")

            # 2. Active project context (if scoped)
            if project_id:
                with self.db_manager.connection() as conn:
                    p_row = conn.execute("SELECT * FROM projects WHERE id = ?;", (project_id,)).fetchone()
                    if p_row:
                        lines.append(f"• Active Project: {p_row['name']} (Stack: {p_row['tech_stack']})")

            # 3. Top semantic memories
            top_memories = self.list_all(project_id=project_id, limit=max_items)
            if top_memories:
                lines.append("• Stored Long-Term Memories:")
                for m in top_memories:
                    tag_str = f" [{', '.join(m.tags)}]" if m.tags else ""
                    lines.append(f"  - ({m.category.capitalize()}) {m.content}{tag_str}")

            if not lines:
                return ""

            return (
                "[PERSISTENT LONG-TERM MEMORY & USER PROFILE]\n"
                f"<recalled_context>\n"
                + "\n".join(lines)
                + "\n</recalled_context>"
            )


# Global singleton instance
memory_store = MemoryStore()
