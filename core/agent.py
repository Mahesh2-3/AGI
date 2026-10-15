import json
import re
import time
import uuid
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional
from core.fast_path import FastPathRouter
from core.llm_client import LLMClient
from config.settings import settings
from core.model_manager import model_manager
from core.prompts import get_system_prompt
from core.safety import SafetyEngine
from core.workspace import workspace_manager
from memory.store import MemoryStore, memory_store as default_memory_store
from tools.base import ToolRegistry, ToolResult, registry as default_registry
from vision.grounding import grounding_engine

CLASSIFIER_SYSTEM_PROMPT = """You are the semantic intent classifier for Jarvis, an autonomous desktop AI assistant.
Analyze the user request given the current desktop agent context and classify the intent into exactly one of the allowed categories.

ALLOWED INTENTS:
- "new_task": An independent request for Jarvis to operate desktop tools, launch applications, automate workflows, interact with websites or software (e.g., "open terminal", "launch chrome", "play music on spotify", "chat with chatgpt for 10 messages", "look at the screen and click submit", "search for weather"). Any compound request combining inspection and desktop actions (view+act) is "new_task".
- "continuation": Resumes, corrects, retries, modifies, or adds instructions to an existing active task or recent workflow (e.g., "do the same in Chrome", "click the other one", "that didn't work", "close it", "try again", "already opened one", "yes, proceed").
- "observational": Purely inspect, read, or describe the current screen WITHOUT interacting with desktop UI (e.g., "what do you see on the screen", "read my screen", "describe what is visible").
- "general": Conversational questions, chit-chat, explanations, or responses addressed directly to Jarvis that do NOT require desktop actions or tools (e.g., "What is Python?", "How are you?", "Tell me a joke", "thank you", "hello"). Even if an active goal exists, general knowledge or conversational questions remain "general".
- "cancel": Explicit request to abort, stop, cancel, or clear the current active task (e.g., "cancel this", "stop what you are doing", "abort task"). Do not classify desktop UI actions like closing an application, window, or tab ("close it", "close Chrome") as cancel; closing an application or tab is an action task ("continuation" or "new_task").

CONTEXT RESOLUTION:
Resolve context-dependent pronouns, references, and relative instructions (such as "do the same in Chrome", "click the other one", "that didn't work", "close it") into concrete, explicit instructions in "resolved_action" using the Active Goal, Previous Goal, Execution Status, and Recent Conversation History.

OUTPUT FORMAT:
Respond with ONLY valid JSON:
{
  "intent": "new_task" | "continuation" | "observational" | "general" | "cancel",
  "resolved_action": "<clarified/resolved action or request resolving pronouns like 'it', 'the other one', 'same'>",
  "reasoning": "<short explanation>"
}"""


class JarvisAgent:
    """The central ReAct (Reason + Act) autonomous orchestrator for J.A.R.V.I.S."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        safety_engine: Optional[SafetyEngine] = None,
        memory_store: Optional[MemoryStore] = None,
        max_iterations: int = 10,
        enable_fast_path: bool = True,
        on_thought: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_result: Optional[Callable[..., None]] = None,
        on_workflow_plan: Optional[Callable[[str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        enable_streaming: bool = False,
    ):
        self.llm = llm_client or LLMClient()
        self.registry = tool_registry or default_registry
        self.safety = safety_engine or SafetyEngine()
        self.memory = memory_store or default_memory_store
        self.max_iterations = max_iterations
        self.enable_fast_path = enable_fast_path
        self.enable_streaming = enable_streaming
        self.fast_path = FastPathRouter(registry=self.registry, safety=self.safety)

        # Context caching (A5)
        self._cached_base_prompt: Optional[str] = None
        self._cached_memory_context: Optional[str] = None

        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()}
        ]

        # Event callbacks for CLI / UI hooks
        self.on_thought = on_thought
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_workflow_plan = on_workflow_plan
        self.on_token = on_token

        self.failed_operations: List[Dict[str, Any]] = []
        self.last_step_duration: float = 0.0
        self.active_goal: Optional[str] = None
        self.last_completed_goal: Optional[str] = None

        # Per-stage performance measurements (A4)
        self.last_turn_metrics: Dict[str, Any] = {
            "planning_duration_s": 0.0,
            "model_prep_duration_s": 0.0,
            "llm_requests": [],
            "tool_executions": [],
            "total_duration_s": 0.0,
            "iterations": 0,
            "streamed": False,
        }

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Formats execution duration into human-readable latency."""
        if seconds < 1.0:
            return f"{seconds:.2f}s ({int(seconds * 1000)}ms)"
        elif seconds < 60.0:
            return f"{seconds:.2f}s"
        else:
            mins = int(seconds // 60)
            rem_sec = seconds % 60
            return f"{mins}m {rem_sec:.1f}s ({seconds:.2f}s)"

    def _append_timing(self, reply: str, start_time: float) -> str:
        """Records elapsed operation duration telemetry without appending to user-facing reply."""
        elapsed = time.perf_counter() - start_time
        self.last_step_duration = elapsed
        return reply.rstrip()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Returns the granular per-stage latency breakdown of the most recent turn (A4)."""
        return self.last_turn_metrics

    def _build_system_prompt(self, user_query: Optional[str] = None) -> str:
        """Constructs system prompt augmented with proactive long-term memory & relevant recalled facts.
        Caches prompt if context has not changed (A5).
        """
        core_mem = self.memory.get_context_summary()
        recalled_section = ""
        if user_query:
            relevant = self.memory.recall(user_query, limit=3)
            if relevant:
                recalled_lines = [f"  - [{m.category.capitalize()}] {m.content}" for m in relevant]
                recalled_section = "• Relevant Recalled Memories for this instruction:\n" + "\n".join(recalled_lines)

        full_mem_context = f"{core_mem}\n{recalled_section}".strip()
        if self._cached_base_prompt and self._cached_memory_context == full_mem_context:
            return self._cached_base_prompt

        prompt = get_system_prompt(memory_context=full_mem_context)
        self._cached_base_prompt = prompt
        self._cached_memory_context = full_mem_context
        return prompt

    def reset(self):
        """Clears working memory back to the initial system prompt."""
        self._cached_base_prompt = None
        self._cached_memory_context = None
        self.messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        self.failed_operations = []
        self.active_goal = None
        self.last_completed_goal = None

    def _prune_context(self, max_recent: int = 12) -> List[Dict[str, Any]]:
        """Maintains a slim context window to prevent memory bloat and model confusion (A5)."""
        if len(self.messages) <= max_recent + 2:
            return self.messages

        system_msg = self.messages[0]
        initial_user = self.messages[1] if len(self.messages) > 1 and self.messages[1].get("role") == "user" else None

        tail = self.messages[-max_recent:]
        while tail and tail[0].get("role") == "tool":
            tail = tail[1:]

        pruned = [system_msg]
        if initial_user and initial_user not in tail:
            pruned.append(initial_user)
        for m in tail:
            if m != system_msg and m != initial_user:
                pruned.append(m)
        return pruned

    def _extract_synthetic_tool_calls(self, content: str) -> List[Any]:
        """Extracts tool calls emitted as raw JSON or code blocks within message content."""
        if not content or not isinstance(content, str):
            return []

        candidates = []

        # 1. Extract markdown code blocks (e.g. ```json ... ``` or ```tool_call ... ```)
        code_block_pattern = re.compile(r"```(?:json|tool_call)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
        for match in code_block_pattern.finditer(content):
            candidates.append(match.group(1).strip())

        # 2. Extract <tool_call> tags
        tag_pattern = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.IGNORECASE)
        for match in tag_pattern.finditer(content):
            candidates.append(match.group(1).strip())

        # 3. Add entire content
        candidates.append(content.strip())

        def _extract_balanced_json(text: str) -> List[Dict[str, Any]]:
            results = []
            depth = 0
            start = -1
            in_str = False
            escape = False
            for idx, ch in enumerate(text):
                if ch == '"' and not escape:
                    in_str = not in_str
                elif ch == '\\' and in_str:
                    escape = not escape
                    continue
                elif not in_str:
                    if ch == '{':
                        if depth == 0:
                            start = idx
                        depth += 1
                    elif ch == '}':
                        if depth > 0:
                            depth -= 1
                            if depth == 0 and start != -1:
                                snippet = text[start : idx + 1]
                                try:
                                    parsed = json.loads(snippet)
                                    if isinstance(parsed, dict):
                                        results.append(parsed)
                                except Exception:
                                    pass
            return results

        tool_calls = []
        for candidate in candidates:
            # Check if directly valid JSON object or list
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            tool_calls.append(item)
                elif isinstance(data, dict):
                    tool_calls.append(data)
            except Exception:
                # Extract embedded balanced JSON structures
                tool_calls.extend(_extract_balanced_json(candidate))

        valid_calls = []
        for call_data in tool_calls:
            fn_name = None
            args = {}
            if "name" in call_data and ("arguments" in call_data or "parameters" in call_data):
                fn_name = call_data["name"]
                args = call_data.get("arguments", call_data.get("parameters", {}))
            elif "function" in call_data:
                fn_info = call_data["function"]
                if isinstance(fn_info, dict):
                    fn_name = fn_info.get("name")
                    args = fn_info.get("arguments", fn_info.get("parameters", {}))
                elif isinstance(fn_info, str):
                    fn_name = fn_info
                    args = call_data.get("arguments", call_data.get("parameters", {}))

            if not fn_name:
                continue

            call_id = call_data.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            valid_calls.append(
                SimpleNamespace(
                    id=call_id,
                    type="function",
                    function=SimpleNamespace(
                        name=fn_name,
                        arguments=json.dumps(args) if isinstance(args, dict) else str(args),
                    ),
                )
            )

        return valid_calls

    def _generate_workflow_via_llm(self, user_input: str) -> Optional[str]:
        """Optional dynamic workflow synthesis helper. Disabled by default to eliminate redundant pre-ReAct LLM calls (A1)."""
        return None

    def generate_workflow_plan(self, user_input: str) -> str:
        """Constructs a deterministic conditional arrow-chained workflow plan without extra LLM round-trips (A1)."""
        prompt_lower = user_input.lower().strip()

        # Check for explicit arrow chains e.g. "step 1 -> step 2 -> step 3"
        if "->" in user_input:
            parts = [p.strip() for p in user_input.split("->") if p.strip()]
            return " -> ".join(parts)

        # 1. Commits / GitHub / Repository inspection
        if any(w in prompt_lower for w in ["commit", "commits", "repository", "repo", "github"]):
            return (
                "see the screen -> if browser has github tab that contains a repo -> "
                "go to the commits -> analyze commits using DOM -> "
                "else tell the user there is no github tab is opened"
            )

        # 2. Visual Screen Capture & Perception
        if any(w in prompt_lower for w in [
            "what are you seeing", "what do you see", "see on the screen", "see the screen",
            "see screen", "inspect screen", "inspect the screen", "look at the screen",
            "look at screen", "read screen", "read the screen", "read my screen",
            "screen description", "what is on the screen", "what's on the screen",
            "what is on my screen", "what's on my screen", "describe screen", "describe the screen",
            "tell me what you see",
        ]):
            return (
                "capture screenshot -> inspect screen with context-aware VLM -> "
                "present visual observations to user"
            )

        # 3. Conversational / Web App Chat Interaction (ChatGPT, Claude, messaging)
        if any(w in prompt_lower for w in ["chatgpt", "chat with", "talk to", "send a message", "message on", "talk with"]):
            return (
                "check open windows -> if target app is open -> focus window -> "
                "locate message input field -> type message and submit -> "
                "observe response"
            )

        # 4. Existing Open Window / Interaction without opening duplicate sessions
        if any(phrase in prompt_lower for phrase in [
            "don't open new", "do not open new", "already opened", "already open",
            "currently open", "existing window", "existing browser"
        ]):
            return (
                "inspect active windows -> if target window is open -> "
                "focus window -> dispatch input actions directly -> "
                "else report target window is not open"
            )

        # 4. Workspace / isolation
        if "new workspace" in prompt_lower or "workspace" in prompt_lower:
            return (
                "check workspaces -> switch to isolated workspace -> "
                "launch target application -> verify workspace focus -> perform actions"
            )

        # 5. Browser / Web navigation
        if any(w in prompt_lower for w in [
            "browser", "chess", "website", "url", "http", "google", "youtube", "webpage", "web page"
        ]):
            return (
                "check open windows -> if browser is running -> switch to tab -> "
                "else launch browser in workspace -> navigate to target URL -> "
                "scan page DOM -> interact with elements -> verify outcome"
            )

        # 6. Memory operations
        if any(w in prompt_lower for w in [
            "remember", "recall", "memorize", "forget", "memory", "memories", "preference"
        ]):
            return (
                "query neural memory store -> if relevant record exists -> "
                "update or recall context -> persist memory state -> confirm to user"
            )

        # 7. File operations
        if any(w in prompt_lower for w in [
            "file", "folder", "directory", "move", "copy", "delete", "create"
        ]):
            return (
                "inspect filesystem path -> if path exists -> perform file operation -> "
                "verify disk changes -> else report path not found"
            )

        # 8. GUI Targeting & Interaction (click, type, press)
        if any(w in prompt_lower for w in ["click", "press", "button", "tap"]):
            return (
                "capture screenshot -> locate target element via vision grounding -> "
                "move mouse to coordinates -> if target confirmed -> execute click -> "
                "else report element not found"
            )

        if any(w in prompt_lower for w in ["type", "input text", "enter text"]):
            return (
                "locate target input field -> if field found -> focus input -> "
                "type text payload -> verify input state -> else report input field missing"
            )

        # 9. Audio / Volume / System telemetry
        if any(w in prompt_lower for w in ["volume", "mute", "unmute", "audio", "battery", "telemetry", "status"]):
            return (
                "query system status -> apply requested setting -> verify state -> report to user"
            )

        # 10. Process / Shell execution
        if any(w in prompt_lower for w in ["run", "execute", "shell", "bash", "command", "process", "kill", "terminate"]):
            return (
                "evaluate command safety tier -> execute action -> capture output -> report result to user"
            )

        # 11. Deterministic conversational fallback (eliminates pre-ReAct LLM call)
        return (
            "analyze user request -> identify target application -> "
            "dispatch actions -> if action succeeds -> verify outcome -> "
            "else report issue to user"
        )

    def _execute_llm_turn(
        self,
        active_context: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]],
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Any:
        """Executes one LLM inference step, streaming tokens when on_token is active (L1)."""
        effective_on_token = on_token or self.on_token
        use_stream = effective_on_token is not None or getattr(self, "enable_streaming", False)

        if use_stream:
            stream = self.llm.chat_stream(
                messages=active_context,
                tools=tools_schema if tools_schema else None,
            )
            content_chunks: List[str] = []
            tool_calls_dict: Dict[int, Dict[str, Any]] = {}
            reasoning_chunks: List[str] = []

            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta

                # Reasoning tokens
                reasoning = getattr(delta, "reasoning", None) or getattr(delta, "thinking", None)
                if reasoning and isinstance(reasoning, str):
                    reasoning_chunks.append(reasoning)
                    if self.on_thought:
                        self.on_thought(reasoning)

                # Content tokens
                content_chunk = getattr(delta, "content", None)
                if content_chunk and isinstance(content_chunk, str):
                    content_chunks.append(content_chunk)
                    if effective_on_token:
                        effective_on_token(content_chunk)

                # Tool call tokens
                tcs = getattr(delta, "tool_calls", None)
                if tcs:
                    for tc in tcs:
                        idx = getattr(tc, "index", 0)
                        if idx not in tool_calls_dict:
                            tool_calls_dict[idx] = {
                                "id": getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}",
                                "name": "",
                                "arguments": [],
                            }
                        if getattr(tc, "id", None) and isinstance(tc.id, str):
                            tool_calls_dict[idx]["id"] = tc.id
                        if getattr(tc, "function", None):
                            fn = tc.function
                            if getattr(fn, "name", None) and isinstance(fn.name, str):
                                tool_calls_dict[idx]["name"] += fn.name
                            if getattr(fn, "arguments", None) and isinstance(fn.arguments, str):
                                tool_calls_dict[idx]["arguments"].append(fn.arguments)

            content = "".join(content_chunks) if content_chunks else None
            assembled_tool_calls = []
            for idx in sorted(tool_calls_dict.keys()):
                entry = tool_calls_dict[idx]
                assembled_tool_calls.append(
                    SimpleNamespace(
                        id=entry["id"],
                        type="function",
                        function=SimpleNamespace(
                            name=entry["name"],
                            arguments="".join(entry["arguments"]),
                        ),
                    )
                )

            return SimpleNamespace(
                role="assistant",
                content=content,
                tool_calls=assembled_tool_calls if assembled_tool_calls else None,
                reasoning="".join(reasoning_chunks) if reasoning_chunks else None,
            )
        else:
            response = self.llm.chat(
                messages=active_context,
                tools=tools_schema if tools_schema else None,
            )
            return response.choices[0].message

    def _classify_request_intent(self, user_input: str) -> Dict[str, Any]:
        """Semantically classifies incoming user intent using the local Qwen3 model (self.llm).

        Distinguishes between new_task, continuation, observational, general, and cancel.
        Resolves context-dependent references, handles compound requests, and eliminates
        all hardcoded phrase and starter lists.
        """
        raw_text = user_input.strip() if user_input else ""
        lower = raw_text.lower()

        # 1. Input validation: Empty inputs default to general
        if not raw_text:
            return {
                "type": "general",
                "is_followup": False,
                "is_observational": False,
                "is_new_action": False,
                "resolved_action": "",
                "reasoning": "Empty user input",
            }

        # 2. Safety-critical emergency controls (stop, cancel, reset, abort)
        if lower in ("stop", "cancel", "reset", "clear goal", "abort"):
            return {
                "type": "cancel",
                "is_followup": False,
                "is_observational": False,
                "is_new_action": False,
                "resolved_action": "cancel",
                "reasoning": "Safety-critical emergency stop command",
            }

        # 3. Assemble lightweight agent context for semantic classification
        active_goal = self.active_goal
        last_goal = getattr(self, "last_completed_goal", None)

        if self.failed_operations:
            status = f"Failed operations: {', '.join(f.get('tool', 'unknown') + ': ' + str(f.get('error', '')) for f in self.failed_operations[-2:])}"
        elif active_goal:
            status = f"In progress (active goal: {active_goal})"
        elif last_goal:
            status = f"Completed previous goal: {last_goal}"
        else:
            status = "Ready / Idle"

        history_lines = []
        for msg in self.messages[-6:]:
            role = msg.get("role")
            if role in ("user", "assistant"):
                content = msg.get("content") or ""
                if len(content) > 120:
                    content = content[:120] + "..."
                history_lines.append(f"{role.capitalize()}: {content}")
        recent_history = "\n".join(history_lines) if history_lines else "None"

        user_prompt = (
            f"Context:\n"
            f"- Active Goal: {active_goal or 'None'}\n"
            f"- Previous Completed Goal: {last_goal or 'None'}\n"
            f"- Execution Status: {status}\n"
            f"- Recent Conversation History:\n{recent_history}\n\n"
            f"User Request: \"{raw_text}\"\n"
            f"Classify the intent into JSON:"
        )

        messages = [
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # 4. Invoke existing local Qwen3 model via self.llm
        intent_t0 = time.perf_counter()
        ALLOWED_INTENTS = {"new_task", "continuation", "observational", "general", "cancel"}
        try:
            # Check if self.llm has classify_intent method
            if hasattr(self.llm, "classify_intent") and callable(self.llm.classify_intent):
                response = self.llm.classify_intent(messages, temperature=0.0)
            else:
                response = self.llm.chat(messages=messages, tools=None, temperature=0.0)

            # Extract message content
            content = None
            if hasattr(response, "choices") and response.choices:
                msg_obj = response.choices[0].message
                content = getattr(msg_obj, "content", None)
            elif isinstance(response, dict):
                content = response.get("content")

            if not isinstance(content, str):
                raise ValueError("LLM response did not contain string content")

            # Strip thinking tags from Qwen3
            clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            # Parse JSON
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, flags=re.DOTALL)
            if not json_match:
                json_match = re.search(r"\{.*\}", clean, flags=re.DOTALL)

            if not json_match:
                raise ValueError(f"No JSON object found in output: {clean[:100]}")

            parsed = json.loads(json_match.group(1 if json_match.lastindex else 0))
            intent = parsed.get("intent", "").lower().strip()
            resolved = parsed.get("resolved_action", "").strip() or raw_text
            reasoning = parsed.get("reasoning", "").strip()

            if intent not in ALLOWED_INTENTS:
                raise ValueError(f"Unknown intent: {intent}")

            intent_dur = time.perf_counter() - intent_t0
            self.last_turn_metrics["intent_classification_duration_s"] = round(intent_dur, 4)

            return {
                "type": intent,
                "is_followup": (intent == "continuation"),
                "is_observational": (intent == "observational"),
                "is_new_action": (intent == "new_task"),
                "resolved_action": resolved,
                "reasoning": reasoning,
            }

        except Exception as e:
            intent_dur = time.perf_counter() - intent_t0
            self.last_turn_metrics["intent_classification_duration_s"] = round(intent_dur, 4)
            # Safe fallback handling for unparseable output or mock environments
            is_obs = ("screen" in lower) and any(w in lower for w in ("see", "inspect", "describe", "look", "read", "what", "show")) and not any(w in lower for w in ("click", "type", "press", "open", "launch", "submit", "run", "close", "delete"))
            if is_obs:
                fallback_type = "observational"
            elif active_goal:
                fallback_type = "continuation"
            else:
                fallback_type = "new_task"

            return {
                "type": fallback_type,
                "is_followup": (fallback_type == "continuation"),
                "is_observational": (fallback_type == "observational"),
                "is_new_action": (fallback_type == "new_task"),
                "resolved_action": raw_text,
                "reasoning": f"Safe fallback due to classification error or mock environment: {e}",
            }

    def step(self, user_input: str, on_token: Optional[Callable[[str], None]] = None) -> str:
        """Processes a user input through the autonomous ReAct cycle with latency tracking and streaming (L1, A4)."""
        start_time = time.perf_counter()
        lower_input = user_input.strip().lower()

        # Reset per-turn performance telemetry (A4)
        self.last_turn_metrics = {
            "planning_duration_s": 0.0,
            "intent_classification_duration_s": 0.0,
            "model_prep_duration_s": 0.0,
            "llm_requests": [],
            "tool_executions": [],
            "total_duration_s": 0.0,
            "iterations": 0,
            "streamed": bool(on_token or self.on_token or getattr(self, "enable_streaming", False)),
        }

        # A2: Ultra-fast reflex route for deterministic commands (<15ms)
        # Safe, independent commands (volume, time, mute, telemetry) execute immediately even if a goal is active!
        if self.enable_fast_path:
            fast_reply = self.fast_path.route(user_input)
            if fast_reply is not None:
                # Goal-dependent continuation directives stay on the reasoning path
                goal_continuation_keywords = ["continue", "submit", "try again", "already open", "keep going"]
                if not (self.active_goal and any(k in lower_input for k in goal_continuation_keywords)):
                    timed_reply = self._append_timing(fast_reply, start_time)
                    self.messages.append({"role": "user", "content": user_input})
                    self.messages.append({"role": "assistant", "content": timed_reply})
                    self.last_turn_metrics["total_duration_s"] = round(time.perf_counter() - start_time, 4)
                    return timed_reply

        # Determine task relation: new task vs. continuation vs. observational query
        intent_info = self._classify_request_intent(user_input)

        # Check for explicit goal cancellation or reset
        if intent_info["type"] == "cancel":
            self.active_goal = None
            self.last_completed_goal = None
            self.failed_operations.clear()
            self.messages.append({"role": "user", "content": user_input})
            reply = "Understood, Sir. I have cleared the active goal."
            self.messages.append({"role": "assistant", "content": reply})
            self.last_turn_metrics["total_duration_s"] = round(time.perf_counter() - start_time, 4)
            return self._append_timing(reply, start_time)

        is_observational_query = intent_info["is_observational"]
        is_followup_or_correction = intent_info["is_followup"]

        if is_observational_query:
            # Observational queries are self-contained and clear lingering active action goals
            self.active_goal = None
        elif intent_info["is_new_action"]:
            # Fresh action task: start a new goal
            self.active_goal = user_input.strip()

        # Build prompt payload preserving active goal context and resolved action
        target_active_goal = self.active_goal or getattr(self, "last_completed_goal", None)
        if is_followup_or_correction and target_active_goal:
            self.active_goal = target_active_goal
            resolved_directive = intent_info.get("resolved_action") or user_input
            effective_user_content = (
                f"[CURRENT ACTIVE OBJECTIVE]: {self.active_goal}\n"
                f"[USER CORRECTION / DIRECTION]: {user_input}\n"
                f"[RESOLVED ACTION]: {resolved_directive}\n\n"
                f"DIRECTIVE: Continue executing towards the ACTIVE OBJECTIVE above using the user's latest correction/guidance. "
                f"Do NOT reset, do NOT ask what task is meant, and do NOT claim you lack capabilities. "
                f"Execute the next required desktop tool action."
            )
        else:
            effective_user_content = user_input

        # Contextualize vision grounding and screen inspection with current user request
        grounding_engine.set_current_user_prompt(user_input)

        # A1: Deterministic workflow plan upfront without extra LLM round-trip
        plan_t0 = time.perf_counter()
        workflow_plan = self.generate_workflow_plan(user_input)
        self.last_turn_metrics["planning_duration_s"] = round(time.perf_counter() - plan_t0, 4)
        if self.on_workflow_plan:
            self.on_workflow_plan(workflow_plan)

        # Refresh system prompt with cached memory context (A5)
        self.messages[0]["content"] = self._build_system_prompt(user_input)
        self.messages.append({"role": "user", "content": effective_user_content})

        # Initialize working workspace if not already established
        if workspace_manager._working_workspace is None:
            init_ws = workspace_manager.get_active_workspace()
            workspace_manager.set_working_workspace(init_ws["id"])

        tools_schema = self.registry.get_schemas()
        iterations = 0

        # Action requests detected semantically (new_task or continuation)
        has_action_request = intent_info["type"] in ("new_task", "continuation")

        # State tracking to prevent redundant / duplicate tool executions
        perception_cache: Dict[str, ToolResult] = {}
        screen_modified_since_perception = False
        last_tool_signature = None

        state_changing_tools = {
            "click_element", "click_coordinate", "click_mouse", "move_mouse",
            "move_to_element", "type_into_element", "type_text_input",
            "press_shortcut", "scroll_page", "launch_application", "close_application",
            "open_browser_in_workspace", "open_browser_url", "click_webpage_element",
            "type_webpage_input", "execute_browser_chain", "focus_browser_address_bar",
            "focus_window", "make_working_workspace_active", "switch_workspace",
        }
        perception_tools = {
            "inspect_screen", "scan_webpage_dom", "list_open_windows",
        }

        while iterations < self.max_iterations:
            iterations += 1
            self.last_turn_metrics["iterations"] = iterations

            # Prune context to avoid memory bloat (A5)
            active_context = self._prune_context()

            try:
                message = self._execute_llm_turn(active_context, tools_schema, on_token=on_token)
            except Exception as e:
                err_msg = f"Neural processing error: {e}"
                if self.on_thought:
                    self.on_thought(err_msg)
                self.last_turn_metrics["total_duration_s"] = round(time.perf_counter() - start_time, 4)
                return self._append_timing(
                    f"I encountered an error communicating with my cognitive backend: {e}",
                    start_time,
                )

            # Record per-request LLM latency metrics (A4)
            llm_metrics = dict(getattr(self.llm, "last_metrics", {}))
            self.last_turn_metrics["llm_requests"].append(llm_metrics)
            prep_metrics = getattr(model_manager, "last_switch_metrics", {})
            self.last_turn_metrics["model_prep_duration_s"] += prep_metrics.get("duration_s", 0.0)

            # If the model produced reasoning tokens (e.g. Qwen3 thinking), surface to thought callback
            reasoning = getattr(message, "reasoning", None) or getattr(message, "thinking", None)
            if reasoning and self.on_thought:
                self.on_thought(str(reasoning).strip())

            # Check if the LLM invoked tools (via standard tool_calls or synthetic tool calls in content)
            tool_calls = list(message.tool_calls) if getattr(message, "tool_calls", None) else []
            is_synthetic = False
            if not tool_calls and getattr(message, "content", None):
                extracted = self._extract_synthetic_tool_calls(message.content)
                if extracted:
                    tool_calls = extracted
                    is_synthetic = True

            if tool_calls:
                # Add the assistant's tool-call request to the message history
                if is_synthetic:
                    self.messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments if isinstance(tc.function.arguments, str) else json.dumps(tc.function.arguments),
                                },
                            }
                            for tc in tool_calls
                        ],
                    })
                elif hasattr(message, "model_dump"):
                    self.messages.append(message.model_dump(exclude_none=True))
                else:
                    self.messages.append({
                        "role": "assistant",
                        "content": getattr(message, "content", None),
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments if isinstance(tc.function.arguments, str) else json.dumps(tc.function.arguments),
                                },
                            }
                            for tc in tool_calls
                        ],
                    })

                abort_subsequent = False
                for tool_call in tool_calls:
                    fn_name = tool_call.function.name
                    raw_args = tool_call.function.arguments

                    if abort_subsequent:
                        # If a previous tool in this batch failed, skip execution
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": fn_name,
                            "content": "Tool execution aborted because a previous tool in this batch failed. Please inspect the previous error.",
                        })
                        continue

                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {"raw_arguments": raw_args}

                    # Flatten nested arguments/parameters dictionaries from Qwen/Ollama models
                    if isinstance(args, dict):
                        if "arguments" in args and isinstance(args["arguments"], dict):
                            nested = args["arguments"]
                            extra = {k: v for k, v in args.items() if k not in ("arguments", "function", "name")}
                            args = {**nested, **extra}
                        elif "parameters" in args and isinstance(args["parameters"], dict):
                            nested = args["parameters"]
                            extra = {k: v for k, v in args.items() if k not in ("parameters", "function", "name")}
                            args = {**nested, **extra}

                    if self.on_tool_call:
                        self.on_tool_call(fn_name, args)

                    tool_sig = (fn_name, json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(args))
                    duration = 0.0
                    cached_perception_hit = False

                    # 1. Deduplicate consecutive identical tool call
                    if tool_sig == last_tool_signature and fn_name not in ("wait_seconds",):
                        result = ToolResult(
                            success=False,
                            output=None,
                            error=(
                                f"Repeated tool execution suppressed: '{fn_name}' was just executed with identical arguments. "
                                f"Do not repeat identical calls. Formulate your final response to {settings.USER_TITLE} or proceed with next action."
                            ),
                        )
                    # 2. Deduplicate repeated perception tool calls if screen state has not changed
                    elif fn_name in perception_tools and fn_name in perception_cache and not screen_modified_since_perception:
                        cached_result = perception_cache[fn_name]
                        result = ToolResult(
                            success=True,
                            output=cached_result.output,
                            error=None,
                        )
                        duration = 0.001
                        cached_perception_hit = True
                    else:
                        # Authorize action via SafetyEngine
                        tool_obj = self.registry.get_tool(fn_name)
                        req_conf = tool_obj.requires_confirmation if tool_obj else False
                        allowed, reject_reason = self.safety.authorize(fn_name, args, req_conf)

                        if not allowed:
                            result = ToolResult(success=False, output=None, error=reject_reason)
                        else:
                            # Ensure Jarvis's working workspace is active on the monitor before interacting
                            if fn_name in (
                                "click_element", "type_into_element", "click_coordinate", "move_mouse", "click_mouse",
                                "move_to_element", "locate_element_by_image",
                                "type_text_input", "press_shortcut", "scroll_page",
                                "take_screenshot", "inspect_screen", "locate_element", "capture_window_screenshot",
                                "open_browser_in_workspace", "scan_webpage_dom", "click_webpage_element",
                                "type_webpage_input", "execute_browser_chain", "focus_browser_address_bar"
                            ):
                                workspace_manager.ensure_working_workspace_active()

                            tool_start = time.perf_counter()
                            result = self.registry.execute(fn_name, args)
                            duration = time.perf_counter() - tool_start

                            # Track screen state modification and cache perceptions
                            if fn_name in state_changing_tools:
                                screen_modified_since_perception = True
                                perception_cache.clear()
                            elif fn_name in perception_tools and result.success:
                                perception_cache[fn_name] = result
                                screen_modified_since_perception = False

                            # Terminal gate for observational queries: close tool gate so LLM synthesizes final text (A3)
                            if fn_name in perception_tools and result.success and is_observational_query and not has_action_request:
                                tools_schema = None

                            # Terminal gate when speak_message is executed
                            if fn_name == "speak_message" and result.success:
                                tools_schema = None

                    last_tool_signature = tool_sig

                    # Record per-tool latency telemetry (A4)
                    self.last_turn_metrics["tool_executions"].append({
                        "tool": fn_name,
                        "duration_s": round(duration, 4),
                        "success": result.success,
                    })

                    if not result.success:
                        self.failed_operations.append({
                            "tool": fn_name,
                            "error": result.error,
                            "args": args,
                        })
                        abort_subsequent = True

                    if self.on_tool_result:
                        try:
                            self.on_tool_result(fn_name, result.output if result.success else result.error, result.success, duration)
                        except TypeError:
                            self.on_tool_result(fn_name, result.output if result.success else result.error, result.success)

                    # Preserve rich perception outputs while guarding against token overflow (A5)
                    clean_content = result.to_message_content()
                    if cached_perception_hit:
                        clean_content = (
                            "[SCREEN UNCHANGED - RE-USING PREVIOUS OBSERVATIONS]\n"
                            f"{clean_content}\n\n"
                            f"DIRECTIVE: Screen state has not changed since your previous {fn_name}. "
                            f"Do NOT repeat {fn_name}. Instead, proceed directly to your next action "
                            f"(such as focus_window, click_element, type_text_input, or press_shortcut) "
                            f"to make progress on your active objective."
                        )

                    if fn_name == "inspect_screen":
                        if is_observational_query or any(w in user_input.lower() for w in ["what do you see", "read the screen", "see the screen", "describe the screen", "what's on my screen", "what is on my screen", "tell me what you see"]):
                            clean_content += f"\n\nDIRECTIVE FOR JARVIS: Report what you observe directly to {settings.USER_TITLE} in your first-person Jarvis persona. Your observational objective is complete."
                        else:
                            clean_content += (
                                f"\n\nDIRECTIVE FOR JARVIS: Now that you have observed the screen, proceed immediately with the next action "
                                f"required for {settings.USER_TITLE}'s goal: '{user_input}'. "
                                f"Focus the target application window using focus_window, click using click_element, or type using type_text_input. "
                                f"Do NOT stop at observation or repeat inspect_screen."
                            )

                    max_chars = 2500
                    if len(clean_content) > max_chars:
                        clean_content = clean_content[:max_chars] + "... [truncated to prevent token overflow]"

                    # Append tool execution result
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": clean_content,
                    })

                # Loop continues so the LLM can inspect tool outputs
                continue

            # No tool calls: final answer produced (A3: Stop when goal is satisfied)
            content = message.content or ""

            # Guard against local 4B/8B model meta-evaluator leakage (e.g. "No function call required")
            lower_c = content.lower().strip()
            meta_evaluator_phrases = [
                "no function call required",
                "no function call needed",
                "no further function calls are needed",
                "no further tool calls are needed",
                "no tool call needed",
                "has already retrieved the screen description",
                "the assistant has already retrieved",
            ]
            if any(phrase in lower_c for phrase in meta_evaluator_phrases):
                if is_observational_query or not has_action_request:
                    last_tool_msg = next((m for m in reversed(self.messages) if m.get("role") == "tool" and m.get("name") in ("inspect_screen", "take_screenshot")), None)
                    if last_tool_msg and last_tool_msg.get("content"):
                        desc = last_tool_msg["content"]
                        desc_chunk = desc.split("\n\nDIRECTIVE")[0].strip()
                        try:
                            parsed = json.loads(desc_chunk)
                            if isinstance(parsed, dict) and "description" in parsed:
                                desc = parsed["description"]
                        except Exception:
                            pass
                        content = f"Sir, on your screen I observe the following:\n\n{desc}"
                else:
                    content = re.sub(
                        r"(?i)(?:the assistant (?:has )?)?(?:already retrieved [^.\n]*[.\n]?|no (?:further )?(?:function|tool) call[s]? (?:are |is )?(?:required|needed)[^.\n]*[.\n]?)",
                        "",
                        content,
                    ).strip()

            timed_reply = self._append_timing(content, start_time)
            self.messages.append({"role": "assistant", "content": content})
            self.last_turn_metrics["total_duration_s"] = round(time.perf_counter() - start_time, 4)

            # Observational queries are self-contained and clear active goal; completed goals record last_completed_goal
            if is_observational_query:
                self.active_goal = None
            elif self.active_goal:
                self.last_completed_goal = self.active_goal

            try:
                model_manager.unload_on_idle(force=False)
            except Exception:
                pass
            return timed_reply

        try:
            model_manager.unload_on_idle(force=False)
        except Exception:
            pass

        self.last_turn_metrics["total_duration_s"] = round(time.perf_counter() - start_time, 4)

        if self.failed_operations:
            failure_lines = "\n".join([f"- **{f['tool']}**: {f['error']}" for f in self.failed_operations[-3:]])
            fail_msg = (
                f"Sir, I was unable to complete the task because the following operations encountered issues:\n"
                f"{failure_lines}\n\n"
                f"Please let me know how you would like to proceed or if I should attempt an alternative method."
            )
            return self._append_timing(fail_msg, start_time)

        return self._append_timing(
            "I have reached the maximum reasoning iterations for this task without reaching a final conclusion, Sir.",
            start_time,
        )
