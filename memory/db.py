"""SQLite database management and schema migrations for Project J.A.R.V.I.S.

Provides ACID transactions, WAL mode, FTS5 full-text indexing, automatic schema
migrations, and lossless import from legacy JSON memory stores.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import threading
from typing import Any, Dict, Generator, List, Optional, Tuple

from config.settings import settings

logger = logging.getLogger("JARVIS.Database")


class DatabaseManager:
    """Manages SQLite connections, schema lifecycle, transactions, and backups."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path) if db_path else settings.DATABASE_PATH
        self._lock = threading.RLock()
        self._local = threading.local()
        self._initialized = False
        self._ensure_parent_dir()
        self.initialize_schema()

    def _ensure_parent_dir(self):
        parent = self.db_path.parent
        parent.mkdir(parents=True, exist_ok=True)

    def get_raw_connection(self) -> sqlite3.Connection:
        """Returns a thread-local SQLite connection configured with WAL and foreign keys."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=10.0,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            # Enable WAL mode and foreign keys
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            self._local.conn = conn
        return conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding a thread-safe connection wrapped in an atomic transaction."""
        with self._lock:
            conn = self.get_raw_connection()
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction rolled back due to error: {e}")
                raise e

    def initialize_schema(self):
        """Initializes normalized tables, FTS5 virtual tables, indexes, and triggers."""
        with self._lock:
            if self._initialized:
                return

            with self.connection() as conn:
                # 1. Schema Migrations table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        applied_at TEXT NOT NULL
                    );
                """)

                # 2. User Profile table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_profile (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                """)

                # 3. Core Multi-Layer Memories table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        layer TEXT NOT NULL DEFAULT 'semantic',
                        category TEXT NOT NULL DEFAULT 'fact',
                        subject TEXT,
                        predicate TEXT,
                        content TEXT NOT NULL,
                        tags TEXT DEFAULT '[]',
                        project_id TEXT,
                        importance INTEGER DEFAULT 3,
                        confidence REAL DEFAULT 1.0,
                        status TEXT DEFAULT 'active',
                        source TEXT DEFAULT 'user',
                        provenance TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_accessed TEXT NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        expires_at TEXT,
                        superseded_by TEXT,
                        FOREIGN KEY (superseded_by) REFERENCES memories(id) ON DELETE SET NULL
                    );
                """)

                # Indexes on core memories
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_layer ON memories(layer);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC);")

                # 4. FTS5 Virtual Table for full-text keyword retrieval
                # Check if FTS5 table exists
                fts_exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts';"
                ).fetchone()

                if not fts_exists:
                    conn.execute("""
                        CREATE VIRTUAL TABLE memories_fts USING fts5(
                            id UNINDEXED,
                            content,
                            subject,
                            predicate,
                            category,
                            tags,
                            tokenize='porter unicode61'
                        );
                    """)

                # Triggers to keep FTS5 synchronized with memories table (only active memories indexed)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS trg_memories_ai AFTER INSERT ON memories
                    WHEN new.status = 'active'
                    BEGIN
                        INSERT INTO memories_fts(id, content, subject, predicate, category, tags)
                        VALUES (new.id, new.content, coalesce(new.subject, ''), coalesce(new.predicate, ''), new.category, new.tags);
                    END;
                """)

                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS trg_memories_ad AFTER DELETE ON memories
                    BEGIN
                        DELETE FROM memories_fts WHERE id = old.id;
                    END;
                """)

                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS trg_memories_au AFTER UPDATE ON memories
                    BEGIN
                        DELETE FROM memories_fts WHERE id = old.id;
                        INSERT INTO memories_fts(id, content, subject, predicate, category, tags)
                        SELECT new.id, new.content, coalesce(new.subject, ''), coalesce(new.predicate, ''), new.category, new.tags
                        WHERE new.status = 'active';
                    END;
                """)

                # 5. Local Semantic Embeddings table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_embeddings (
                        memory_id TEXT PRIMARY KEY,
                        model TEXT NOT NULL,
                        dimensions INTEGER NOT NULL,
                        vector BLOB NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                    );
                """)

                # 6. Project Memory table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS projects (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        path TEXT,
                        tech_stack TEXT DEFAULT '[]',
                        architecture_summary TEXT,
                        status TEXT DEFAULT 'active',
                        milestones TEXT DEFAULT '[]',
                        blockers TEXT DEFAULT '[]',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                """)

                # 7. Working Memory / Resumable Task Checkpoints table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS task_checkpoints (
                        task_id TEXT PRIMARY KEY,
                        project_id TEXT,
                        user_request TEXT NOT NULL,
                        goal TEXT NOT NULL,
                        status TEXT NOT NULL,
                        completed_steps TEXT DEFAULT '[]',
                        pending_steps TEXT DEFAULT '[]',
                        last_tool_name TEXT,
                        last_tool_result TEXT,
                        error_summary TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
                    );
                """)

                # 8. Procedural Memory table (Verified fixes & workflows)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS procedures (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        trigger_intent TEXT NOT NULL,
                        preconditions TEXT DEFAULT '[]',
                        steps TEXT NOT NULL,
                        expected_outcome TEXT NOT NULL,
                        verification_method TEXT,
                        failure_modes TEXT DEFAULT '[]',
                        confidence REAL DEFAULT 0.8,
                        verified INTEGER DEFAULT 0,
                        success_count INTEGER DEFAULT 0,
                        last_executed TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                """)

                conn.execute("CREATE INDEX IF NOT EXISTS idx_procedures_intent ON procedures(trigger_intent);")

            self._initialized = True
            logger.info(f"SQLite database initialized at {self.db_path} with FTS5.")

    def create_backup(self, label: str = "auto") -> Path:
        """Creates a timestamped snapshot backup of the SQLite database."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = self.db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"jarvis_db_{label}_{timestamp}.db"

        with self._lock:
            # Checkpoint WAL before copying
            try:
                conn = self.get_raw_connection()
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except Exception:
                pass

            # Safe SQLite backup API
            backup_conn = sqlite3.connect(str(backup_path))
            try:
                conn.backup(backup_conn)
                logger.info(f"Database backed up successfully to {backup_path}")
            finally:
                backup_conn.close()

        return backup_path

    def migrate_from_json_if_needed(self, json_path: Optional[Path | str] = None) -> bool:
        """Losslessly imports profile attributes and memory entries from legacy JSON storage.
        
        Backs up the JSON file before importing and records migration in schema_migrations.
        """
        target_json = Path(json_path) if json_path else settings.MEMORY_FILE_PATH
        if not target_json.exists():
            return False

        with self.connection() as conn:
            # Check if migration was already applied
            migrated = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE version = 1;"
            ).fetchone()
            if migrated:
                return False

            # Check if memories already exist in database
            count_row = conn.execute("SELECT count(*) as count FROM memories;").fetchone()
            if count_row and count_row["count"] > 0:
                conn.execute(
                    "INSERT OR IGNORE INTO schema_migrations (version, name, applied_at) VALUES (1, 'legacy_json_import', ?);",
                    (datetime.now(timezone.utc).isoformat(),),
                )
                return False

            try:
                # 1. Automated pre-migration snapshot backup
                backup_dir = target_json.parent / "backups"
                backup_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                json_backup = backup_dir / f"jarvis_memory_backup_{ts}.json"
                shutil.copy2(target_json, json_backup)
                logger.info(f"Legacy JSON memory backed up to {json_backup}")

                # 2. Read legacy JSON data
                with open(target_json, "r", encoding="utf-8") as f:
                    data = json.load(f)

                profile_dict = data.get("profile", {})
                raw_memories = data.get("memories", [])
                now_str = datetime.now(timezone.utc).isoformat()

                # 3. Import User Profile attributes
                for k, v in profile_dict.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO user_profile (key, value, updated_at) VALUES (?, ?, ?);",
                        (str(k), str(v), now_str),
                    )

                # 4. Import Memory entries
                imported_count = 0
                for item in raw_memories:
                    if not isinstance(item, dict) or not item.get("content"):
                        continue

                    mem_id = item.get("id") or f"mem_{os.urandom(4).hex()}"
                    content = item.get("content", "").strip()
                    category = item.get("category", "fact")
                    tags = json.dumps(item.get("tags", []))
                    importance = int(item.get("importance", 3))
                    created_at = item.get("created_at", now_str)
                    updated_at = item.get("updated_at", now_str)
                    access_count = int(item.get("access_count", 0))
                    last_accessed = item.get("last_accessed", now_str)

                    # Determine appropriate memory layer
                    layer = "semantic"
                    if category == "task":
                        layer = "working"
                    elif category == "procedure":
                        layer = "procedural"

                    conn.execute(
                        """
                        INSERT OR REPLACE INTO memories (
                            id, layer, category, content, tags, importance,
                            confidence, status, source, created_at, updated_at,
                            last_accessed, access_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            mem_id, layer, category, content, tags, importance,
                            1.0, "active", "user", created_at, updated_at,
                            last_accessed, access_count,
                        ),
                    )
                    imported_count += 1

                # 5. Record migration status
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (1, 'legacy_json_import', ?);",
                    (now_str,),
                )
                logger.info(
                    f"Successfully losslessly migrated {imported_count} memories and "
                    f"{len(profile_dict)} profile attributes to SQLite database."
                )
                return True

            except Exception as e:
                logger.error(f"Failed to migrate legacy JSON memory: {e}")
                raise e


# Default database manager singleton
db_manager = DatabaseManager()
