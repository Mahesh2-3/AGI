import concurrent.futures
import json
import logging
import re
import uuid
from typing import Any, Callable, Dict, List, Optional

from core.llm_client import LLMClient
from memory.store import MemoryEntry, MemoryStore, memory_store

logger = logging.getLogger("JARVIS.MemoryExtractor")

EXTRACTION_SYSTEM_PROMPT = """You are J.A.R.V.I.S.'s Cognitive Memory Extractor.
Analyze the provided user input, conversation, or completed task execution, and extract durable, long-term memories.

MEMORY LAYERS:
- "semantic": Stable facts, user preferences, habits, profile attributes, and corrections.
- "episodic": Notable past events, user decisions, task milestones, and error resolutions.
- "project": Codebase architecture, frameworks, file paths, active branch, blockers.
- "procedural": Reusable troubleshooting workflows, multi-step fixes, and command sequences.

CATEGORIES:
- "preference": User tool, styling, workflow, or configuration preferences (e.g. editor, theme, shell).
- "fact": Verified facts about user environment, hardware, project, or tools.
- "profile": Core user personal details (name, title, timezone, role).
- "event": Significant milestone or resolved problem from past task.
- "procedure": Step-by-step actionable procedure or fix that can be reused later.
- "correction": An explicit correction from the user superseding an earlier belief.

RULES:
1. Extract ONLY durable, useful knowledge. Do NOT extract ephemeral chatter (e.g., "hello", "thank you", "what is 2+2").
2. For preferences and facts, identify a clear `subject` (entity e.g. "user", "hyprland", "git") and `predicate` (attribute e.g. "text_editor", "terminal", "branch").
3. Rate importance from 1 (minor detail) to 5 (critical core fact/preference).
4. For procedural workflows, include `procedure_details` with `title`, `trigger_intent`, `steps` (list of strings), and `expected_outcome`.
5. Return a strict JSON array of objects. If no durable memories exist, return `[]`.

JSON OUTPUT FORMAT:
[
  {
    "layer": "semantic",
    "category": "preference",
    "subject": "user",
    "predicate": "text_editor",
    "content": "User prefers Visual Studio Code over Neovim",
    "importance": 4,
    "tags": ["user", "preference", "editor", "vscode"],
    "confidence": 0.95
  }
]
"""


class MemoryExtractor:
    """Extracts, categorizes, and deduplicates structured memories using Qwen3,
    with non-blocking asynchronous execution and robust heuristic fallbacks."""

    def __init__(
        self,
        llm_client: Any = "default",
        store: Optional[MemoryStore] = None,
        max_workers: int = 1,
    ):
        if llm_client == "default":
            self.llm = LLMClient()
        else:
            self.llm = llm_client
        self.store = store or memory_store
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="jarvis_memory_worker",
        )

    def extract_from_text(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Extracts structured memories from raw text or a user request."""
        clean_text = text.strip() if text else ""
        if not clean_text:
            return []

        # If LLM is not configured, fall back directly to heuristic extraction
        if not self.llm or not getattr(self.llm, "is_configured", False):
            return self.extract_heuristically(clean_text)

        user_content = f"Input Text:\n\"{clean_text}\"\n"
        if context:
            user_content += f"\nContext:\n{context}\n"
        user_content += "\nExtract long-term memories in strict JSON array format:"

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            response = self.llm.classify_intent(messages=messages, temperature=0.0)
            raw_content = response.choices[0].message.content or ""
            items = self._parse_json_items(raw_content)
            if items:
                return items
        except Exception as e:
            logger.warning(f"LLM memory extraction failed, falling back to heuristics: {e}")

        # Fallback to deterministic regex heuristics if LLM produced no items
        return self.extract_heuristically(clean_text)

    def extract_from_conversation(
        self,
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Extracts durable facts, preferences, and corrections from conversation turns."""
        if not messages:
            return []

        formatted_lines = []
        for msg in messages[-8:]:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "").strip()
            if content:
                formatted_lines.append(f"{role}: {content}")

        conv_text = "\n".join(formatted_lines)
        return self.extract_from_text(conv_text, context="Recent multi-turn conversation")

    def extract_from_task(
        self,
        goal: str,
        plan: str,
        steps: List[Dict[str, Any]],
        success: bool,
        error: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Extracts episodic event summaries and reusable procedural workflows from completed tasks."""
        memories: List[Dict[str, Any]] = []
        clean_goal = goal.strip() if goal else "Unknown task"

        # 1. Episodic task memory
        status_str = "Successfully completed" if success else "Failed to complete"
        step_summary = []
        for s in steps[:5]:
            tool = s.get("tool", "action")
            step_summary.append(tool)

        episodic_content = f"{status_str} task: '{clean_goal}' using actions: {', '.join(step_summary)}."
        if error:
            episodic_content += f" Encountered error: {error}."

        memories.append({
            "layer": "episodic",
            "category": "event",
            "subject": "task",
            "predicate": "execution",
            "content": episodic_content,
            "importance": 3 if success else 4,
            "tags": ["task", "execution", "history"],
            "confidence": 1.0,
        })

        # 2. If multi-step task succeeded, extract reusable procedural workflow
        if success and len(steps) >= 2:
            step_descriptions = []
            for s in steps:
                tool = s.get("tool", "")
                args = s.get("args", {})
                if tool == "run_command" and "CommandLine" in args:
                    step_descriptions.append(f"Execute `{args['CommandLine']}`")
                elif tool:
                    step_descriptions.append(f"Run {tool} with {args}")

            if step_descriptions:
                memories.append({
                    "layer": "procedural",
                    "category": "procedure",
                    "subject": clean_goal,
                    "predicate": "procedure",
                    "content": f"Procedure for '{clean_goal}': {'; '.join(step_descriptions)}",
                    "importance": 4,
                    "tags": ["procedure", "workflow", "verified"],
                    "confidence": 0.9,
                    "procedure_details": {
                        "title": f"Workflow: {clean_goal}",
                        "trigger_intent": clean_goal,
                        "steps": step_descriptions,
                        "expected_outcome": f"Successful completion of {clean_goal}",
                        "verified": True,
                    },
                })

        return memories

    def process_and_store(
        self,
        extracted_items: List[Dict[str, Any]],
    ) -> List[Any]:
        """Saves extracted memories to the SQLite store, handling procedures,
        deduplication, and fact supersession."""
        saved_records = []
        for item in extracted_items:
            try:
                layer = item.get("layer", "semantic")
                category = item.get("category", "fact")
                content = item.get("content", "").strip()
                if not content:
                    continue

                subject = item.get("subject")
                predicate = item.get("predicate")
                importance = int(item.get("importance", 3))
                tags = item.get("tags", [])
                confidence = float(item.get("confidence", 1.0))
                proc_details = item.get("procedure_details")

                # If this item contains full procedural workflow details, save to procedures table
                if category == "procedure" and proc_details and isinstance(proc_details, dict):
                    proc_res = self.store.save_procedure(
                        title=proc_details.get("title", content[:50]),
                        trigger_intent=proc_details.get("trigger_intent", subject or content[:30]),
                        steps=proc_details.get("steps", [content]),
                        expected_outcome=proc_details.get("expected_outcome", "Task completed"),
                        preconditions=proc_details.get("preconditions"),
                        verification_method=proc_details.get("verification_method"),
                        confidence=confidence,
                        verified=bool(proc_details.get("verified", False)),
                    )
                    saved_records.append(proc_res)
                    continue

                # Otherwise store into core memories table
                entry = self.store.remember(
                    content=content,
                    category=category,
                    layer=layer,
                    subject=subject,
                    predicate=predicate,
                    importance=importance,
                    tags=tags,
                    confidence=confidence,
                )
                saved_records.append(entry)
            except Exception as e:
                logger.error(f"Error persisting extracted memory item: {e}")

        return saved_records

    def extract_and_store_async(
        self,
        text_or_messages: Any,
        context: Optional[str] = None,
        on_complete: Optional[Callable[[List[Any]], None]] = None,
    ) -> concurrent.futures.Future:
        """Asynchronously extracts and persists memories in a background thread."""
        def _worker():
            try:
                if isinstance(text_or_messages, list):
                    items = self.extract_from_conversation(text_or_messages)
                else:
                    items = self.extract_from_text(str(text_or_messages), context=context)

                saved = self.process_and_store(items)
                if on_complete:
                    on_complete(saved)
                return saved
            except Exception as ex:
                logger.error(f"Background memory extraction failed: {ex}")
                return []

        return self._executor.submit(_worker)

    def extract_heuristically(self, text: str) -> List[Dict[str, Any]]:
        """Fast, rule-based extraction for common user declarations and preferences without LLM."""
        items: List[Dict[str, Any]] = []
        lower = text.lower()

        # 1. User name declarations
        name_match = re.search(r"\b(?:my name is|call me|i am|i'm)\s+([a-zA-Z]+)\b", lower)
        if name_match:
            name = name_match.group(1).capitalize()
            items.append({
                "layer": "semantic",
                "category": "profile",
                "subject": "user",
                "predicate": "name",
                "content": f"User's name is {name}",
                "importance": 5,
                "tags": ["profile", "name", "user"],
                "confidence": 1.0,
            })

        # 2. Key-value preferences ("User preference terminal: kitty" or "preference editor: vscode")
        pref_match = re.search(r"\b(?:user\s+)?preference\s+([a-zA-Z0-9_]+):\s*(.+)$", text, re.IGNORECASE)
        if pref_match:
            p_key = pref_match.group(1).strip().lower()
            p_val = pref_match.group(2).strip()
            items.append({
                "layer": "semantic",
                "category": "preference",
                "subject": "user",
                "predicate": p_key,
                "content": f"User preferred {p_key} is {p_val}",
                "importance": 4,
                "tags": ["preference", p_key],
                "confidence": 0.95,
            })

        # 3. Preference phrasing: "I prefer X over Y" or "I prefer X"
        prefer_match = re.search(r"\bi\s+prefer\s+([a-zA-Z0-9_\-\.\s]+?)(?:\s+over\s+([a-zA-Z0-9_\-\.\s]+))?(?:[,\.\n]|$)", text, re.IGNORECASE)
        if prefer_match and not pref_match:
            chosen = prefer_match.group(1).strip()
            alternative = prefer_match.group(2).strip() if prefer_match.group(2) else None
            content = f"User prefers {chosen}" + (f" over {alternative}" if alternative else "")
            items.append({
                "layer": "semantic",
                "category": "preference",
                "subject": "user",
                "predicate": "preference",
                "content": content,
                "importance": 4,
                "tags": ["preference", chosen.split()[0].lower()],
                "confidence": 0.9,
            })

        # 4. Explicit memory directives: "Remember that X"
        remember_match = re.search(r"\b(?:remember\s+that|note\s+that|don't\s+forget\s+that)\s+(.+)$", text, re.IGNORECASE)
        if remember_match:
            raw_fact = remember_match.group(1).strip().rstrip(".")
            items.append({
                "layer": "semantic",
                "category": "fact",
                "subject": "general",
                "predicate": "fact",
                "content": raw_fact,
                "importance": 4,
                "tags": ["explicit_memory"],
                "confidence": 1.0,
            })

        return items

    def _parse_json_items(self, raw_text: str) -> List[Dict[str, Any]]:
        """Safely parses JSON array of memory items from LLM response text."""
        candidates = []

        # 1. Markdown code blocks
        cb_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text, re.IGNORECASE)
        for m in cb_matches:
            candidates.append(m.strip())

        candidates.append(raw_text.strip())

        for candidate in candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    return [d for d in data if isinstance(d, dict) and "content" in d]
                elif isinstance(data, dict):
                    if "memories" in data and isinstance(data["memories"], list):
                        return [d for d in data["memories"] if isinstance(d, dict) and "content" in d]
                    if "content" in data:
                        return [data]
            except Exception:
                pass

        # Balanced array search fallback
        start = raw_text.find("[")
        end = raw_text.rfind("]")
        if start != -1 and end > start:
            snippet = raw_text[start : end + 1]
            try:
                data = json.loads(snippet)
                if isinstance(data, list):
                    return [d for d in data if isinstance(d, dict) and "content" in d]
            except Exception:
                pass

        return []

    def shutdown(self, wait: bool = False):
        """Cleans up the background executor thread pool."""
        self._executor.shutdown(wait=wait)
