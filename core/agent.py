import json
import time
from typing import Any, Callable, Dict, List, Optional
from core.fast_path import FastPathRouter
from core.llm_client import LLMClient
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

    def _prune_context(self, max_recent: int = 8) -> List[Dict[str, Any]]:
        """Maintains a slim context window to prevent exceeding Groq TPM (Tokens Per Minute) limits."""
        if len(self.messages) <= max_recent + 2:
            return self.messages

        system_msg = self.messages[0]
        initial_user = self.messages[1] if len(self.messages) > 1 and self.messages[1].get("role") == "user" else None

        tail = self.messages[-max_recent:]
        pruned = [system_msg]
        if initial_user and initial_user not in tail:
            pruned.append(initial_user)
        for m in tail:
            if m != system_msg and m != initial_user:
                pruned.append(m)
        return pruned

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
            "what are you seeing", "what do you see", "see on the screen", "inspect screen",
            "look at the screen", "read screen", "screen description", "what is on the screen"
        ]):
            return (
                "capture screenshot -> inspect screen with context-aware VLM -> "
                "extract UI elements and state -> present visual observations to user"
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

        # Contextualize vision grounding and screen inspection with current user request
        grounding_engine.set_current_user_prompt(user_input)

        # Ultra-fast reflex route for common system tasks (<15ms)
        if self.enable_fast_path:
            fast_reply = self.fast_path.route(user_input)
            if fast_reply is not None:
                timed_reply = self._append_timing(fast_reply, start_time)
                self.messages.append({"role": "user", "content": user_input})
                self.messages.append({"role": "assistant", "content": timed_reply})
                return timed_reply

        # Display planned execution workflow upfront
        workflow_plan = self.generate_workflow_plan(user_input)
        if self.on_workflow_plan:
            self.on_workflow_plan(workflow_plan)

        # Refresh system prompt with latest memory context & relevant recalled facts
        self.messages[0]["content"] = self._build_system_prompt(user_input)
        self.messages.append({"role": "user", "content": user_input})

        # Initialize working workspace if not already established
        if workspace_manager._working_workspace is None:
            init_ws = workspace_manager.get_active_workspace()
            workspace_manager.set_working_workspace(init_ws["id"])

        tools_schema = self.registry.get_schemas()
        iterations = 0

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

            # Check if the LLM invoked tools
            if hasattr(message, "tool_calls") and message.tool_calls:
                # Add the assistant's tool-call request to the message history
                self.messages.append(message.model_dump(exclude_none=True))

                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    raw_args = tool_call.function.arguments

                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {"raw_arguments": raw_args}

                    if self.on_tool_call:
                        self.on_tool_call(fn_name, args)

                    # Authorize action via SafetyEngine
                    tool_obj = self.registry.get_tool(fn_name)
                    req_conf = tool_obj.requires_confirmation if tool_obj else False
                    allowed, reject_reason = self.safety.authorize(fn_name, args, req_conf)

                    duration = 0.0
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

                    if not result.success:
                        self.failed_operations.append({
                            "tool": fn_name,
                            "error": result.error,
                            "args": args,
                        })

                    if self.on_tool_result:
                        try:
                            self.on_tool_result(fn_name, result.output if result.success else result.error, result.success, duration)
                        except TypeError:
                            self.on_tool_result(fn_name, result.output if result.success else result.error, result.success)

                    # Preserve rich perception outputs (inspect_screen, DOM, windows) while guarding against massive outputs
                    clean_content = result.to_message_content()
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
            return timed_reply

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
