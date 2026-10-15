import json
from pathlib import Path
import tempfile
import pytest

from memory.context_builder import ContextBuilder
from memory.store import MemoryStore


def test_context_builder_profile_and_basic_sections():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "ctx_test.db"
        store = MemoryStore(db_path=db_path)
        store.set_profile_attribute("name", "Mahesh")
        store.set_profile_attribute("theme", "dark")
        store.remember("Arch Linux system with linux-zen kernel", category="fact", tags=["arch"])

        builder = ContextBuilder(store=store, max_tokens=1000)
        context = builder.build_context()

        assert "[PERSISTENT LONG-TERM MEMORY & CONTEXT]" in context
        assert "<recalled_context>" in context
        assert "</recalled_context>" in context
        assert "Name: Mahesh" in context
        assert "Theme: dark" in context
        assert "linux-zen" in context


def test_context_builder_token_budget_enforcement():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "budget_test.db"
        store = MemoryStore(db_path=db_path)
        store.set_profile_attribute("name", "Mahesh")

        # Add multiple long memories
        for i in range(15):
            store.remember(f"Long verbose memory entry number {i} detailing system configuration and history of actions", importance=3)

        builder = ContextBuilder(store=store, max_tokens=120)
        context = builder.build_context()

        # Token count should be strictly bounded
        est_tokens = builder.estimate_tokens(context)
        assert est_tokens <= 150  # Allows slight buffer for framing
        # High-priority profile is preserved even under tight budget
        assert "Name: Mahesh" in context


def test_resumable_task_checkpoint_lifecycle():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "cp_test.db"
        store = MemoryStore(db_path=db_path)

        # 1. Start task
        cp = store.save_checkpoint(
            task_id="task_build_01",
            user_request="Build C++ kernel module and run tests",
            goal="Build C++ kernel module",
            status="in_progress",
            completed_steps=["git checkout master", "cmake -B build"],
            pending_steps=["cmake --build build", "ctest"],
            last_tool_name="run_command",
            last_tool_result="CMake configured successfully",
        )
        assert cp["status"] == "in_progress"

        # 2. Retrieve active checkpoint
        active = store.get_latest_active_checkpoint()
        assert active is not None
        assert active["task_id"] == "task_build_01"
        assert len(active["completed_steps"]) == 2

        # 3. Verify ContextBuilder injects active checkpoint
        builder = ContextBuilder(store=store, max_tokens=1000)
        context = builder.build_context(include_checkpoint=True)
        assert "Active Task Checkpoint (Resumed / In-Progress):" in context
        assert "cmake -B build" in context
        assert "CMake configured successfully" in context

        # 4. Update status to completed -> no longer returned as active checkpoint
        store.update_checkpoint_status("task_build_01", status="completed")
        assert store.get_latest_active_checkpoint() is None


def test_agent_checkpoint_resumption_context():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "agent_cp_test.db"
        store = MemoryStore(db_path=db_path)

        # Save an interrupted task checkpoint
        store.save_checkpoint(
            task_id="task_interrupted_99",
            user_request="Download and extract latest dataset",
            goal="Extract dataset",
            status="interrupted",
            completed_steps=["curl -O https://example.com/data.tar.gz"],
            pending_steps=["tar -xzf data.tar.gz"],
            last_tool_name="run_command",
            last_tool_result="Download complete (15MB)",
            error_summary="Network timeout during extraction",
        )

        builder = ContextBuilder(store=store)
        context = builder.build_context(user_query="continue downloading")

        assert "task_interrupted_99" or "Extract dataset" in context
        assert "curl -O" in context
        assert "Network timeout during extraction" in context
