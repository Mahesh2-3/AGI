import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from config.settings import settings
from core.agent import JarvisAgent
from core.fast_path import FastPathRouter
from memory.store import MemoryEntry, MemoryStore
from tools.base import ToolRegistry, registry as global_registry


def test_memory_entry_dataclass():
    entry = MemoryEntry(
        id="mem_test1",
        content="User's favorite chess opening is the Sicilian Defense",
        category="preference",
        tags=["chess", "opening"],
        importance=4,
    )
    data = entry.to_dict()
    assert data["id"] == "mem_test1"
    assert data["importance"] == 4
    assert "chess" in data["tags"]

    restored = MemoryEntry.from_dict(data)
    assert restored.content == entry.content
    assert restored.category == "preference"


def test_memory_store_persistence_and_reload():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_path = Path(tmp_dir) / "test_memory.json"
        store = MemoryStore(file_path=store_path)

        # Store facts
        m1 = store.remember("My name is Mahesh", category="profile", importance=5)
        m2 = store.remember("Preferred browser is Google Chrome", category="preference")
        m3 = store.remember("Friend on chess.com is alex99", category="fact", tags=["chess", "friend"])

        assert store_path.exists()
        assert len(store.list_all()) == 3
        assert store.get_profile().get("name") == "Mahesh"

        # Reload from scratch in a fresh store instance to verify disk persistence
        store2 = MemoryStore(file_path=store_path)
        assert len(store2.list_all()) == 3
        assert store2.get_profile().get("name") == "Mahesh"

        # Recall by query
        results = store2.recall("chess")
        assert len(results) >= 1
        assert "alex99" in results[0].content
        assert results[0].access_count == 1

        # Near-duplicate update
        updated_m3 = store2.remember("Friend on chess.com is alex99", importance=5)
        assert len(store2.list_all()) == 3
        assert updated_m3.importance == 5


def test_memory_store_search_relevance_and_forget():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = MemoryStore(file_path=Path(tmp_dir) / "mem.json")
        store.remember("The project workspace is located at /home/mahesh/core/AGI", tags=["project", "path"])
        store.remember("User prefers Hyprland workspace 2 for coding", tags=["hyprland", "workspace"])
        store.remember("User likes playing classical chess on Sundays", tags=["chess"])

        # Query matches
        hits = store.recall("coding workspace")
        assert len(hits) >= 1
        assert "workspace 2" in hits[0].content

        # Category filter
        hits_proj = store.recall("workspace", category="fact")
        assert len(hits_proj) >= 1

        # Forget by content substring
        removed = store.forget("classical chess")
        assert removed is not None
        assert "classical chess" in removed.content
        assert len(store.list_all()) == 2

        # Context summary
        summary = store.get_context_summary()
        assert "workspace 2" in summary


def test_memory_tools_execution():
    from tools.memory_ops import (
        forget_memory,
        get_memory_summary,
        list_memories,
        recall_memory,
        remember_fact,
        update_user_preference,
    )

    # Verify tool execution via registry schema
    schemas = global_registry.get_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "remember_fact" in names
    assert "recall_memory" in names
    assert "list_memories" in names
    assert "forget_memory" in names
    assert "update_user_preference" in names
    assert "get_memory_summary" in names

    # Execute tools
    rem_res = remember_fact("Test API server runs on port 8080", category="fact", tags=["api", "port"])
    assert rem_res["success"] is True

    rec_res = recall_memory("port 8080")
    assert rec_res["success"] is True
    assert rec_res["count"] >= 1

    list_res = list_memories()
    assert list_res["success"] is True

    pref_res = update_user_preference("terminal", "kitty")
    assert pref_res["success"] is True

    summary_res = get_memory_summary()
    assert summary_res["success"] is True
    assert summary_res["profile"].get("terminal") == "kitty"

    forget_res = forget_memory("port 8080")
    assert forget_res["success"] is True


def test_agent_proactive_memory_context_injection():
    with tempfile.TemporaryDirectory() as tmp_dir:
        custom_store = MemoryStore(file_path=Path(tmp_dir) / "agent_mem.json")
        custom_store.remember("User's name is Mahesh", category="profile", importance=5)
        custom_store.remember("Challenger friend on Chess.com is alex99", category="fact", tags=["chess"])

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(tool_calls=None, content="Certainly, Sir."))]
        mock_llm.chat.return_value = mock_resp

        agent = JarvisAgent(
            llm_client=mock_llm,
            memory_store=custom_store,
        )

        # Trigger step with chess query
        agent.step("Challenge my friend on chess.com")

        # Verify proactive memory injection into system prompt
        system_content = agent.messages[0]["content"]
        assert "PERSISTENT LONG-TERM MEMORY" in system_content
        assert "Name: Mahesh" in system_content
        assert "alex99" in system_content


def test_fast_path_memory_commands():
    with tempfile.TemporaryDirectory() as tmp_dir:
        from memory.store import memory_store
        # Temporarily use isolated file
        orig_file = memory_store.file_path
        memory_store.file_path = Path(tmp_dir) / "fast_path_mem.json"
        memory_store.clear()
        memory_store.profile.clear()

        try:
            router = FastPathRouter()

            # Fast remember
            res_rem = router.route("remember that my chess username is blitz_king")
            assert res_rem is not None
            assert "committed that to long-term memory" in res_rem
            assert "blitz_king" in res_rem

            # Fast recall list
            res_list = router.route("what do you remember about me")
            assert res_list is not None
            assert "blitz_king" in res_list

            # Fast forget
            res_forget = router.route("forget that my chess username is blitz_king")
            assert res_forget is not None
            assert "purged that from my memory" in res_forget

        finally:
            memory_store.file_path = orig_file
            memory_store._load()


if __name__ == "__main__":
    test_memory_entry_dataclass()
    test_memory_store_persistence_and_reload()
    test_memory_store_search_relevance_and_forget()
    test_memory_tools_execution()
    test_agent_proactive_memory_context_injection()
    test_fast_path_memory_commands()
    print("✅ All memory tests passed successfully!")
