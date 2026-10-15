import json
from pathlib import Path
import tempfile
import pytest

from memory.db import DatabaseManager
from memory.store import MemoryEntry, MemoryStore


def test_sqlite_schema_and_tables_creation():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test_jarvis.db"
        mgr = DatabaseManager(db_path=db_path)

        with mgr.connection() as conn:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type in ('table', 'view');"
                ).fetchall()
            ]
            assert "memories" in tables
            assert "memories_fts" in tables
            assert "user_profile" in tables
            assert "projects" in tables
            assert "task_checkpoints" in tables
            assert "procedures" in tables
            assert "memory_embeddings" in tables
            assert "schema_migrations" in tables


def test_sqlite_fts5_synchronization_and_triggers():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "fts_test.db"
        store = MemoryStore(db_path=db_path)

        # 1. Insert triggers FTS insert
        entry = store.remember(
            "Hyprland configuration is located in ~/.config/hypr/hyprland.conf",
            category="fact",
            tags=["hyprland", "config"],
            subject="hyprland",
            predicate="config_path",
        )

        with store.db_manager.connection() as conn:
            fts_row = conn.execute(
                "SELECT * FROM memories_fts WHERE memories_fts MATCH 'hyprland';"
            ).fetchone()
            assert fts_row is not None
            assert "hyprland.conf" in fts_row["content"]

        # 2. Recall via FTS
        recalled = store.recall("hyprland config")
        assert len(recalled) == 1
        assert recalled[0].id == entry.id

        # 3. Update triggers FTS update
        store.remember(
            "Hyprland configuration is located in ~/.config/hypr/hyprland.conf with blur enabled",
            category="fact",
            tags=["hyprland", "blur"],
            subject="hyprland",
            predicate="config_path",
        )

        with store.db_manager.connection() as conn:
            fts_count = conn.execute(
                "SELECT count(*) as c FROM memories_fts WHERE memories_fts MATCH 'blur';"
            ).fetchone()["c"]
            assert fts_count == 1

        # 4. Forget triggers FTS delete
        store.forget("hyprland.conf")
        with store.db_manager.connection() as conn:
            fts_remaining = conn.execute(
                "SELECT count(*) as c FROM memories_fts WHERE memories_fts MATCH 'hyprland';"
            ).fetchone()["c"]
            assert fts_remaining == 0


def test_fact_supersession():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "supersede_test.db"
        store = MemoryStore(db_path=db_path)

        # Old preference
        old_entry = store.remember(
            "User preferred text editor is Neovim",
            category="preference",
            subject="user",
            predicate="text_editor",
        )
        assert old_entry.status == "active"

        # User updates preference to VSCode
        new_entry = store.remember(
            "User preferred text editor is Visual Studio Code",
            category="preference",
            subject="user",
            predicate="text_editor",
        )

        assert new_entry.status == "active"
        assert new_entry.id != old_entry.id

        # Verify old record is marked as superseded in SQLite
        with store.db_manager.connection() as conn:
            old_row = conn.execute(
                "SELECT * FROM memories WHERE id = ?;", (old_entry.id,)
            ).fetchone()
            assert old_row["status"] == "superseded"
            assert old_row["superseded_by"] == new_entry.id

        # Active recall only returns the new preference
        active_hits = store.recall("text editor")
        assert len(active_hits) == 1
        assert "Visual Studio Code" in active_hits[0].content


def test_json_migration_lossless():
    with tempfile.TemporaryDirectory() as td:
        json_path = Path(td) / "legacy_mem.json"
        db_path = Path(td) / "migrated.db"

        sample_data = {
            "version": "1.0",
            "profile": {
                "name": "Mahesh",
                "shell": "zsh",
            },
            "memories": [
                {
                    "id": "mem_arch_01",
                    "content": "Arch Linux kernel is linux-zen",
                    "category": "fact",
                    "tags": ["arch", "kernel"],
                    "importance": 4,
                },
                {
                    "id": "mem_arch_02",
                    "content": "Pacman parallel downloads set to 5",
                    "category": "preference",
                    "tags": ["pacman"],
                    "importance": 3,
                },
            ],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        mgr = DatabaseManager(db_path=db_path)
        mgr.migrate_from_json_if_needed(json_path)

        # Verify backup was created
        backup_files = list((Path(td) / "backups").glob("*.json"))
        assert len(backup_files) == 1

        # Verify database contents
        with mgr.connection() as conn:
            mem_count = conn.execute("SELECT count(*) as c FROM memories;").fetchone()["c"]
            assert mem_count == 2
            prof_name = conn.execute(
                "SELECT value FROM user_profile WHERE key = 'name';"
            ).fetchone()["value"]
            assert prof_name == "Mahesh"
            prof_shell = conn.execute(
                "SELECT value FROM user_profile WHERE key = 'shell';"
            ).fetchone()["value"]
            assert prof_shell == "zsh"


def test_database_backup_creation():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "backup_test.db"
        store = MemoryStore(db_path=db_path)
        store.remember("Critical system memory for backup test", importance=5)

        backup_path = store.db_manager.create_backup(label="unit_test")
        assert backup_path.exists()
        assert backup_path.stat().st_size > 0

        # Verify backup database can be read independently
        mgr_backup = DatabaseManager(db_path=backup_path)
        with mgr_backup.connection() as conn:
            row = conn.execute("SELECT count(*) as c FROM memories;").fetchone()
            assert row["c"] == 1
