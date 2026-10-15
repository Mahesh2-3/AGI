import json
import logging
import math
import re
from typing import Any, Dict, List, Optional

from memory.store import MemoryEntry, MemoryStore, memory_store

logger = logging.getLogger("JARVIS.ContextBuilder")


class ContextBuilder:
    """Token-bounded context builder that assembles multi-layer memories,
    active project metadata, and resumable task checkpoints into system prompt context."""

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        max_tokens: int = 1200,
    ):
        self.store = store or memory_store
        self.max_tokens = max_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Fast approximate token count for English and code (avg ~3.8 chars per token)."""
        if not text:
            return 0
        return math.ceil(len(text) / 3.8)

    def build_context(
        self,
        user_query: Optional[str] = None,
        project_id: Optional[str] = None,
        include_checkpoint: bool = True,
        custom_budget: Optional[int] = None,
    ) -> str:
        """Constructs a structured, token-bounded context block.
        
        Priority order:
        1. User Profile (Critical)
        2. Active Resumed Task Checkpoint (Critical when recovering)
        3. Active Project Metadata (High)
        4. Query-Relevant Recalled Facts & Procedures (High)
        5. General Top Long-Term Memories (Medium)
        """
        budget = custom_budget or self.max_tokens
        sections: List[str] = []
        current_tokens = 0

        # Base header/footer reserve (~40 tokens)
        header = "[PERSISTENT LONG-TERM MEMORY & CONTEXT]\n<recalled_context>"
        footer = "</recalled_context>"
        overhead_tokens = self.estimate_tokens(header) + self.estimate_tokens(footer)
        available_budget = max(50, budget - overhead_tokens)

        # -------------------------------------------------------------
        # Tier 1: User Profile Attributes (Highest Priority)
        # -------------------------------------------------------------
        profile = self.store.get_profile()
        if profile:
            prof_items = []
            if "name" in profile:
                prof_items.append(f"Name: {profile['name']}")
            for k, v in profile.items():
                if k != "name":
                    prof_items.append(f"{k.capitalize()}: {v}")
            if prof_items:
                prof_str = f"• User Profile: {', '.join(prof_items)}"
                p_tokens = self.estimate_tokens(prof_str)
                if current_tokens + p_tokens <= available_budget:
                    sections.append(prof_str)
                    current_tokens += p_tokens

        # -------------------------------------------------------------
        # Tier 2: Resumable Task Checkpoint (High Priority for Recovery)
        # -------------------------------------------------------------
        if include_checkpoint:
            checkpoint = self.store.get_latest_active_checkpoint(project_id=project_id)
            if checkpoint:
                cp_lines = [
                    f"• Active Task Checkpoint (Resumed / In-Progress):",
                    f"  - Goal: {checkpoint.get('goal', 'Unknown')}",
                    f"  - Status: {checkpoint.get('status', 'in_progress')}",
                ]
                comp = checkpoint.get("completed_steps", [])
                if comp:
                    cp_lines.append(f"  - Already Completed Steps: {json.dumps(comp)}")
                pend = checkpoint.get("pending_steps", [])
                if pend:
                    cp_lines.append(f"  - Pending Steps: {json.dumps(pend)}")
                if checkpoint.get("last_tool_result"):
                    res_snippet = str(checkpoint['last_tool_result'])[:150]
                    cp_lines.append(f"  - Last Action Result: {res_snippet}")
                if checkpoint.get("error_summary"):
                    cp_lines.append(f"  - Last Error: {checkpoint['error_summary']}")

                cp_str = "\n".join(cp_lines)
                cp_tokens = self.estimate_tokens(cp_str)
                if current_tokens + cp_tokens <= available_budget:
                    sections.append(cp_str)
                    current_tokens += cp_tokens

        # -------------------------------------------------------------
        # Tier 3: Active Project Context
        # -------------------------------------------------------------
        if project_id:
            try:
                with self.store.db_manager.connection() as conn:
                    p_row = conn.execute("SELECT * FROM projects WHERE id = ?;", (project_id,)).fetchone()
                    if p_row:
                        p_str = f"• Active Project: {p_row['name']} (Stack: {p_row['tech_stack']})"
                        pt_tokens = self.estimate_tokens(p_str)
                        if current_tokens + pt_tokens <= available_budget:
                            sections.append(p_str)
                            current_tokens += pt_tokens
            except Exception as e:
                logger.debug(f"Error fetching project context: {e}")

        # -------------------------------------------------------------
        # Tier 4: Query-Specific Relevant Memories & Procedures (Hybrid/FTS)
        # -------------------------------------------------------------
        if user_query and user_query.strip():
            # 4a. Relevant procedures
            procs = self.store.recall_procedures(user_query, limit=2)
            if procs:
                proc_lines = ["• Verified Reusable Procedures:"]
                for p in procs:
                    steps_preview = "; ".join(p.get("steps", []))[:120]
                    proc_lines.append(f"  - Workflow '{p['title']}': {steps_preview}")
                proc_str = "\n".join(proc_lines)
                pr_tokens = self.estimate_tokens(proc_str)
                if current_tokens + pr_tokens <= available_budget:
                    sections.append(proc_str)
                    current_tokens += pr_tokens

            # 4b. Relevant semantic facts
            relevant = self.store.recall(user_query, project_id=project_id, limit=4, hybrid=False)
            if relevant:
                rel_lines = ["• Relevant Recalled Memories for current instruction:"]
                for m in relevant:
                    tag_str = f" [{', '.join(m.tags)}]" if m.tags else ""
                    line = f"  - ({m.category.capitalize()}) {m.content}{tag_str}"
                    line_tokens = self.estimate_tokens(line)
                    if current_tokens + line_tokens <= available_budget:
                        rel_lines.append(line)
                        current_tokens += line_tokens

                if len(rel_lines) > 1:
                    sections.append("\n".join(rel_lines))

        # -------------------------------------------------------------
        # Tier 5: General Stored Long-Term Memories (Lower Priority)
        # -------------------------------------------------------------
        general_memories = self.store.list_all(project_id=project_id, limit=5)
        if general_memories:
            gen_lines = ["• Stored Long-Term Facts & Preferences:"]
            for m in general_memories:
                tag_str = f" [{', '.join(m.tags)}]" if m.tags else ""
                line = f"  - ({m.category.capitalize()}) {m.content}{tag_str}"
                line_tokens = self.estimate_tokens(line)
                if current_tokens + line_tokens <= available_budget:
                    gen_lines.append(line)
                    current_tokens += line_tokens

            if len(gen_lines) > 1:
                sections.append("\n".join(gen_lines))

        if not sections:
            return ""

        return f"{header}\n" + "\n\n".join(sections) + f"\n{footer}"
