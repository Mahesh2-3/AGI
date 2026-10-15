import concurrent.futures
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import MagicMock
import pytest

from memory.extractor import MemoryExtractor
from memory.store import MemoryStore


def test_heuristic_extraction():
    extractor = MemoryExtractor(llm_client=None)

    # 1. Profile name
    items1 = extractor.extract_heuristically("My name is Mahesh and I like coding.")
    assert len(items1) == 1
    assert items1[0]["category"] == "profile"
    assert items1[0]["subject"] == "user"
    assert items1[0]["predicate"] == "name"
    assert "Mahesh" in items1[0]["content"]

    # 2. Key-value preference
    items2 = extractor.extract_heuristically("User preference terminal: kitty")
    assert len(items2) == 1
    assert items2[0]["category"] == "preference"
    assert items2[0]["predicate"] == "terminal"
    assert "kitty" in items2[0]["content"]

    # 3. Preference comparison
    items3 = extractor.extract_heuristically("I prefer VSCode over Neovim for Python development")
    assert len(items3) == 1
    assert items3[0]["category"] == "preference"
    assert "VSCode" in items3[0]["content"]

    # 4. Explicit memory directive
    items4 = extractor.extract_heuristically("Remember that PostgreSQL runs on port 5433 on this server.")
    assert len(items4) == 1
    assert items4[0]["category"] == "fact"
    assert "port 5433" in items4[0]["content"]


def test_json_parsing_and_robustness():
    extractor = MemoryExtractor(llm_client=None)

    # 1. Standard markdown code block
    raw_markdown = """Here is the extracted memory:
```json
[
  {
    "layer": "semantic",
    "category": "preference",
    "subject": "user",
    "predicate": "theme",
    "content": "User prefers dark mode themes",
    "importance": 3,
    "tags": ["theme", "ui"]
  }
]
```
Hope this helps!"""
    parsed = extractor._parse_json_items(raw_markdown)
    assert len(parsed) == 1
    assert parsed[0]["predicate"] == "theme"

    # 2. Raw JSON string
    raw_json = '[{"content": "Hyprland is using waybar", "category": "fact"}]'
    parsed2 = extractor._parse_json_items(raw_json)
    assert len(parsed2) == 1
    assert "waybar" in parsed2[0]["content"]

    # 3. Malformed input returns empty list without raising exception
    assert extractor._parse_json_items("Not a json at all {broken") == []


def test_process_and_store_semantic_and_supersession():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "store_test.db"
        store = MemoryStore(db_path=db_path)
        extractor = MemoryExtractor(store=store)

        # 1. Initial preference
        item1 = {
            "layer": "semantic",
            "category": "preference",
            "subject": "user",
            "predicate": "browser",
            "content": "User prefers Firefox",
            "importance": 4,
            "tags": ["browser", "web"],
        }
        res1 = extractor.process_and_store([item1])
        assert len(res1) == 1
        assert res1[0].status == "active"

        # 2. Conflicting updated preference (supersession)
        item2 = {
            "layer": "semantic",
            "category": "preference",
            "subject": "user",
            "predicate": "browser",
            "content": "User prefers Google Chrome",
            "importance": 4,
            "tags": ["browser", "chrome"],
        }
        res2 = extractor.process_and_store([item2])
        assert len(res2) == 1
        assert res2[0].id != res1[0].id
        assert res2[0].status == "active"

        # Check old record in SQLite is superseded
        with store.db_manager.connection() as conn:
            old_row = conn.execute("SELECT * FROM memories WHERE id = ?;", (res1[0].id,)).fetchone()
            assert old_row["status"] == "superseded"
            assert old_row["superseded_by"] == res2[0].id

        # Recalling only returns the active preference
        active = store.recall("browser")
        assert len(active) == 1
        assert "Google Chrome" in active[0].content


def test_procedural_workflow_extraction_and_persistence():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "proc_test.db"
        store = MemoryStore(db_path=db_path)
        extractor = MemoryExtractor(store=store)

        goal = "Fix audio crackling in PipeWire"
        steps = [
            {"tool": "run_command", "args": {"CommandLine": "mkdir -p ~/.config/pipewire/pipewire.conf.d"}},
            {"tool": "run_command", "args": {"CommandLine": "systemctl --user restart pipewire"}},
        ]

        extracted = extractor.extract_from_task(
            goal=goal,
            plan="Create config -> restart pipewire",
            steps=steps,
            success=True,
        )

        assert len(extracted) == 2  # 1 episodic + 1 procedural
        assert extracted[0]["layer"] == "episodic"
        assert extracted[1]["layer"] == "procedural"
        assert "procedure_details" in extracted[1]

        # Persist extracted items
        saved = extractor.process_and_store(extracted)
        assert len(saved) == 2

        # Verify procedure in procedures table
        procs = store.recall_procedures("PipeWire")
        assert len(procs) == 1
        assert procs[0]["trigger_intent"] == goal
        assert len(procs[0]["steps"]) == 2
        assert "pipewire.conf.d" in procs[0]["steps"][0]

        # Record procedure outcome
        proc_id = procs[0]["id"]
        store.record_procedure_result(proc_id=proc_id, success=True)
        updated_procs = store.recall_procedures("PipeWire")
        assert updated_procs[0]["success_count"] == 1
        assert updated_procs[0]["verified"] == 1


def test_async_extraction_non_blocking():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "async_test.db"
        store = MemoryStore(db_path=db_path)
        extractor = MemoryExtractor(llm_client=None, store=store)

        completed_records = []

        def callback(records):
            completed_records.extend(records)

        future = extractor.extract_and_store_async(
            "User preference editor: Sublime Text 4",
            on_complete=callback,
        )

        assert isinstance(future, concurrent.futures.Future)
        # Wait for future with reasonable timeout
        res = future.result(timeout=5.0)
        assert len(res) == 1
        assert len(completed_records) == 1
        assert "Sublime Text 4" in res[0].content

        # Verify saved in SQLite store
        recalled = store.recall("Sublime")
        assert len(recalled) == 1

        extractor.shutdown(wait=True)


def test_mocked_llm_extraction_pipeline():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "mock_llm.db"
        store = MemoryStore(db_path=db_path)

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps([
            {
                "layer": "semantic",
                "category": "preference",
                "subject": "git",
                "predicate": "default_branch",
                "content": "User prefers main as the default Git branch",
                "importance": 4,
                "tags": ["git", "branch"],
                "confidence": 0.98,
            }
        ])
        mock_response.choices = [mock_choice]
        mock_llm.is_configured = True
        mock_llm.classify_intent.return_value = mock_response

        extractor = MemoryExtractor(llm_client=mock_llm, store=store)
        extracted = extractor.extract_from_text("Always use main as the default git branch")

        assert len(extracted) == 1
        assert extracted[0]["predicate"] == "default_branch"

        saved = extractor.process_and_store(extracted)
        assert len(saved) == 1
        assert saved[0].status == "active"
        assert "main" in saved[0].content
