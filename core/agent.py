import json
import re
import time
import uuid
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional
from core.fast_path import FastPathRouter
from core.llm_client import LLMClient
from config.settings import settings
from core.prompts import get_system_prompt
from core.safety import SafetyEngine
from core.workspace import workspace_manager
from memory.store import MemoryStore, memory_store as default_memory_store
from tools.base import ToolRegistry, ToolResult, registry as default_registry
from vision.grounding import grounding_engine


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
    ):
        self.llm = llm_client or LLMClient()
        self.registry = tool_registry or default_registry
        self.safety = safety_engine or SafetyEngine()
        self.memory = memory_store or default_memory_store
        self.max_iterations = max_iterations
        self.enable_fast_path = enable_fast_path
        self.fast_path = FastPathRouter(registry=self.registry, safety=self.safety)
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        
        # Event callbacks for CLI / UI hooks
        self.on_thought = on_thought
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_workflow_plan = on_workflow_plan
        self.failed_operations: List[Dict[str, Any]] = []
        self.last_step_duration: float = 0.0
        self.active_goal: Optional[str] = None

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

    def _build_system_prompt(self, user_query: Optional[str] = None) -> str:
        """Constructs system prompt augmented with proactive long-term memory & relevant recalled facts."""
        core_mem = self.memory.get_context_summary()
        recalled_section = ""
        if user_query:
            relevant = self.memory.recall(user_query, limit=3)
            if relevant:
                recalled_lines = [f"  - [{m.category.capitalize()}] {m.content}" for m in relevant]
                recalled_section = "• Relevant Recalled Memories for this instruction:\n" + "\n".join(recalled_lines)

        full_mem_context = f"{core_mem}\n{recalled_section}".strip()
        return get_system_prompt(memory_context=full_mem_context)

    def reset(self):
        """Clears working memory back to the initial system prompt."""
        self.messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        self.failed_operations = []
        self.active_goal = None

    def _prune_context(self, max_recent: int = 14) -> List[Dict[str, Any]]:
        """Maintains a slim context window to prevent exceeding Groq TPM (Tokens Per Minute) limits."""
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
                                start = -1
                escape = False
            return results

        all_dicts = []
        for text in candidates:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    all_dicts.append(parsed)
                elif isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            all_dicts.append(item)
            except Exception:
                all_dicts.extend(_extract_balanced_json(text))

        valid_calls = []
        for item in all_dicts:
            fn_name = item.get("name") or item.get("function") or item.get("tool")
            if not fn_name or not isinstance(fn_name, str):
                continue

            tool_obj = self.registry.get_tool(fn_name)
            if not tool_obj:
                continue

            args = item.get("arguments", item.get("parameters", {}))
            if not args and any(k not in ("name", "function", "tool", "type") for k in item.keys()):
                args = {k: v for k, v in item.items() if k not in ("name", "function", "tool", "type")}

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass

            call_id = f"call_{uuid.uuid4().hex[:8]}"
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
        """Queries LLM for a fast one-line conditional arrow-chain workflow."""
        prompt = (
            "Generate a high-level execution workflow for desktop AI assistant Jarvis.\n"
            "Format strictly as a single line using ' -> ' between steps, including conditional 'if' and 'else'.\n"
            "Example: see the screen -> if browser has github tab that contains a repo -> go to the commits -> analyze commits using DOM -> else tell the user there is no github tab is opened\n\n"
            f"User request: {user_input}\n"
            "Workflow:"
        )
        try:
            resp = self.llm.chat(
                messages=[
                    {"role": "system", "content": "You are a desktop AI workflow planner. Output ONLY the single-line arrow chain."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=80,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.strip('"`\n ')
            if "->" in raw:
                return raw.split("\n")[0].strip()
        except Exception:
            return None
        return None

    def generate_workflow_plan(self, user_input: str) -> str:
        """Constructs a conditional arrow-chained workflow plan (step -> if ... -> else ...) for user inspection."""
        prompt_lower = user_input.lower().strip()

        # Check for explicit arrow chains e.g. "step 1 -> step 2 -> step 3"
        if "->" in user_input:
            parts = [p.strip() for p in user_input.split("->") if p.strip()]
            return " -> ".join(parts)

        # 1. Commits / GitHub / Repository inspection (e.g. user directive)
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
        ]):
            return (
                "capture screenshot -> inspect screen with context-aware VLM -> "
                "present visual observations to user"
            )

        # 3. Existing Open Window / Interaction without opening duplicate sessions
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

        # 9. Dynamic synthesis via LLM if configured, otherwise structured conditional fallback
        if getattr(self.llm, "is_configured", False):
            try:
                llm_plan = self._generate_workflow_via_llm(user_input)
                if llm_plan:
                    return llm_plan
            except Exception:
                pass

        return (
            "analyze user request -> identify target application -> "
            "dispatch actions -> if action succeeds -> verify outcome -> "
            "else report issue to user"
        )

    def step(self, user_input: str) -> str:
        """Processes a user input through the autonomous ReAct cycle."""
        start_time = time.perf_counter()
        lower_input = user_input.strip().lower()

        # Check for explicit goal cancellation or reset
        if lower_input in ("stop", "cancel", "reset", "clear goal", "abort"):
            self.active_goal = None
            self.messages.append({"role": "user", "content": user_input})
            reply = "Understood, Sir. I have cleared the active goal."
            self.messages.append({"role": "assistant", "content": reply})
            return self._append_timing(reply, start_time)

        # Detect if this input is a correction or continuation of an active goal
        is_followup_or_correction = False
        if self.active_goal:
            followup_indicators = [
                "already open", "opened one", "in the browser", "see the screen",
                "look at", "navigate", "clicking", "continue", "submit", "try",
                "man", "please", "yes", "no", "why", "do it", "focus", "click",
            ]
            if any(ind in lower_input for ind in followup_indicators) or len(user_input.split()) <= 8:
                is_followup_or_correction = True

        if not is_followup_or_correction:
            action_indicators = [
                "do ", "solve ", "complete ", "navigate ", "open ", "find ", "click ",
                "type ", "write ", "run ", "test ", "submit ", "fix ", "check ",
            ]
            if any(lower_input.startswith(ind) or ind in lower_input for ind in action_indicators):
                self.active_goal = user_input.strip()

        # Ultra-fast reflex route for common system tasks (<15ms)
        # Bypassed if an active goal is underway so perception/control is handled by the agent
        if self.enable_fast_path and not self.active_goal:
            fast_reply = self.fast_path.route(user_input)
            if fast_reply is not None:
                timed_reply = self._append_timing(fast_reply, start_time)
                self.messages.append({"role": "user", "content": user_input})
                self.messages.append({"role": "assistant", "content": timed_reply})
                return timed_reply

        # Build prompt payload preserving active goal context
        if is_followup_or_correction and self.active_goal:
            effective_user_content = (
                f"[CURRENT ACTIVE OBJECTIVE]: {self.active_goal}\n"
                f"[USER CORRECTION / DIRECTION]: {user_input}\n\n"
                f"DIRECTIVE: Continue executing towards the ACTIVE OBJECTIVE above using the user's latest correction/guidance. "
                f"Do NOT reset, do NOT ask what task is meant, and do NOT claim you lack capabilities. "
                f"Execute the next required desktop tool action."
            )
        else:
            effective_user_content = user_input

        # Contextualize vision grounding and screen inspection with current user request
        grounding_engine.set_current_user_prompt(user_input)

        # Display planned execution workflow upfront
        workflow_plan = self.generate_workflow_plan(user_input)
        if self.on_workflow_plan:
            self.on_workflow_plan(workflow_plan)

        # Refresh system prompt with latest memory context & relevant recalled facts
        self.messages[0]["content"] = self._build_system_prompt(user_input)
        self.messages.append({"role": "user", "content": effective_user_content})

        # Initialize working workspace if not already established
        if workspace_manager._working_workspace is None:
            init_ws = workspace_manager.get_active_workspace()
            workspace_manager.set_working_workspace(init_ws["id"])

        tools_schema = self.registry.get_schemas()
        iterations = 0

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

            # Prune context to avoid TPM limit blowouts
            active_context = self._prune_context()

            try:
                response = self.llm.chat(
                    messages=active_context,
                    tools=tools_schema if tools_schema else None,
                )
            except Exception as e:
                err_msg = f"Neural processing error: {e}"
                if self.on_thought:
                    self.on_thought(err_msg)
                return self._append_timing(
                    f"I encountered an error communicating with my cognitive backend: {e}",
                    start_time,
                )

            choice = response.choices[0]
            message = choice.message

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
                else:
                    self.messages.append(message.model_dump(exclude_none=True))

                abort_subsequent = False
                for tool_call in tool_calls:
                    fn_name = tool_call.function.name
                    raw_args = tool_call.function.arguments

                    if abort_subsequent:
                        # If a previous tool in this batch failed, skip execution
                        # and append a dummy error result to satisfy LLM API requirements.
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

                    last_tool_signature = tool_sig

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

                    # Preserve rich perception outputs (inspect_screen, DOM, windows) while guarding against massive outputs
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

                    max_chars = 3500 if fn_name in ("inspect_screen", "scan_webpage_dom", "list_open_windows", "read_file") else 2500
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

            # No tool calls: final answer produced
            content = message.content or ""
            timed_reply = self._append_timing(content, start_time)
            self.messages.append({"role": "assistant", "content": content})
            try:
                from core.model_manager import model_manager
                model_manager.unload_on_idle()
            except Exception:
                pass
            return timed_reply

        try:
            from core.model_manager import model_manager
            model_manager.unload_on_idle()
        except Exception:
            pass

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
