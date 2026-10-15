from pathlib import Path
import tempfile
import pytest

from core.agent import JarvisAgent
from memory.context_builder import ContextBuilder
from memory.store import MemoryStore
from tools.base import ToolRegistry


def test_scenario_1_restart_persistence():
    """Scenario 1: Persistence across agent restarts without data loss."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "jarvis_prod.db"

        # Session 1: Jarvis stores user preferences, profile, and system facts
        store1 = MemoryStore(db_path=db_path)
        store1.set_profile_attribute("name", "Mahesh")
        store1.set_profile_attribute("role", "Engineer")
        m1 = store1.remember("Arch Linux system with linux-zen kernel", category="fact", tags=["arch", "kernel"])
        m2 = store1.remember("Preferred terminal is kitty", category="preference", tags=["terminal"])

        assert len(store1.list_all()) == 2
        assert store1.get_profile()["name"] == "Mahesh"

        # Simulate full agent shutdown & restart with fresh MemoryStore instance
        del store1
        store2 = MemoryStore(db_path=db_path)

        assert store2.get_profile()["name"] == "Mahesh"
        assert store2.get_profile()["role"] == "Engineer"
        assert len(store2.list_all()) == 2

        # Fast FTS5 recall after restart
        hits = store2.recall("kitty terminal")
        assert len(hits) == 1
        assert hits[0].id == m2.id
        assert hits[0].content == "Preferred terminal is kitty"


def test_scenario_2_semantic_and_hybrid_retrieval():
    """Scenario 2: Accurate keyword and hybrid retrieval."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "hybrid_e2e.db"
        store = MemoryStore(db_path=db_path)

        store.remember("Hyprland configuration enables active window blur", tags=["hyprland", "blur"])
        store.remember("PostgreSQL database port configured as 5433", tags=["database", "postgres"])
        store.remember("Python virtual environment path is /home/mahesh/core/AGI/.venv", tags=["python", "path"])

        # Exact keyword and phrase search via FTS5
        hits_db = store.recall("5433 postgres")
        assert len(hits_db) >= 1
        assert "5433" in hits_db[0].content

        hits_hypr = store.recall("blur window")
        assert len(hits_hypr) >= 1
        assert "Hyprland" in hits_hypr[0].content


def test_scenario_3_fact_supersession_and_conflict_resolution():
    """Scenario 3: Automatic supersession when user preferences evolve."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "supersede_e2e.db"
        store = MemoryStore(db_path=db_path)

        # 1. Initial preference
        old_mem = store.remember(
            "User preferred shell is bash",
            category="preference",
            subject="user",
            predicate="shell",
        )
        assert old_mem.status == "active"

        # 2. User tells Jarvis they switched to zsh
        new_mem = store.remember(
            "User preferred shell is zsh",
            category="preference",
            subject="user",
            predicate="shell",
        )
        assert new_mem.id != old_mem.id
        assert new_mem.status == "active"

        # 3. Verify old record is marked as superseded in SQLite
        old_record = store.get(old_mem.id)
        assert old_record is not None
        assert old_record.status == "superseded"
        assert old_record.superseded_by == new_mem.id

        # 4. Standard recall only returns the new, active preference
        recalled = store.recall("shell")
        assert len(recalled) == 1
        assert recalled[0].id == new_mem.id
        assert "zsh" in recalled[0].content


def test_scenario_4_task_checkpoint_resumption():
    """Scenario 4: Resumable working memory prevents repeating already-completed steps."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "resumption_e2e.db"
        store = MemoryStore(db_path=db_path)

        task_id = "task_pacman_update_01"
        user_goal = "Upgrade system packages and clear package cache"

        # Step 1 & 2 succeed, Step 3 fails due to lock file
        store.save_checkpoint(
            task_id=task_id,
            user_request=user_goal,
            goal=user_goal,
            status="failed",
            completed_steps=["sudo pacman -Syu --noconfirm", "paru -Sua --noconfirm"],
            pending_steps=["sudo pacman -Sc --noconfirm"],
            last_tool_name="run_command",
            last_tool_result="error: failed to init transaction (unable to lock database)",
            error_summary="db.lck file present in /var/lib/pacman/",
        )

        # Next user turn: user asks Jarvis to resume or continue
        builder = ContextBuilder(store=store, max_tokens=1000)
        resumed_context = builder.build_context(user_query="continue package upgrade", include_checkpoint=True)

        # Context contains completed steps and exact failure point
        assert "Active Task Checkpoint (Resumed / In-Progress):" in resumed_context
        assert "sudo pacman -Syu" in resumed_context
        assert "unable to lock database" in resumed_context

        # Agent can now remove the lock and complete the pending step without repeating full upgrade
        store.update_checkpoint_status(task_id=task_id, status="completed")
        assert store.get_latest_active_checkpoint() is None
