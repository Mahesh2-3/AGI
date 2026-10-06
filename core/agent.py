import json
from typing import Any, Callable, Dict, List, Optional
from core.fast_path import FastPathRouter
from core.llm_client import LLMClient
from core.prompts import get_system_prompt
from core.safety import SafetyEngine
from core.workspace import workspace_manager
from memory.store import MemoryStore, memory_store as default_memory_store
from tools.base import ToolRegistry, ToolResult, registry as default_registry


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
        on_tool_result: Optional[Callable[[str, Any, bool], None]] = None,
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

    def generate_workflow_plan(self, user_input: str) -> str:
        """Constructs a clear, concise step-by-step workflow plan for user inspection."""
        prompt_lower = user_input.lower()
        steps = []

        # Check for explicit arrow chains e.g. "step 1 -> step 2 -> step 3"
        if "->" in user_input:
            chain_parts = [p.strip() for p in user_input.split("->") if p.strip()]
            for idx, part in enumerate(chain_parts, 1):
                steps.append(f"{idx}. **{part.capitalize()}**")
            return "\n".join(steps)

        # Workspace / isolation
        if "new workspace" in prompt_lower or "workspace" in prompt_lower:
            steps.append("1. **Workspace Routing**: Switch to a designated desktop workspace for isolation.")

        # Web / Browser
        if any(w in prompt_lower for w in ["browser", "chess", "website", "url", "http", "google", "youtube", "github"]):
            if not steps:
                steps.append("1. **Workspace Routing**: Verify active workspace and window focus.")
            steps.append("2. **High-Speed Navigation**: Launch browser and navigate directly to target URL via DOM driver.")
            steps.append("3. **DOM Scanning**: Query page DOM structure for interactive buttons, links, and forms (<10ms).")
            steps.append("4. **Direct Interaction**: Execute DOM clicks and form inputs with pixel-perfect precision.")
            steps.append("5. **State Verification**: Confirm action results and report outcome.")
        # Memory operations
        elif any(w in prompt_lower for w in ["remember", "recall", "memorize", "forget", "memory", "memories", "preference"]):
            steps.append("1. **Memory Ingestion/Retrieval**: Access persistent neural memory banks.")
            steps.append("2. **State Synchronization**: Persist updates to disk and update user profile.")
            steps.append("3. **Response Delivery**: Confirm memory state to user.")
        # File operations
        elif any(w in prompt_lower for w in ["file", "folder", "directory", "move", "copy", "delete", "create"]):
            steps.append("1. **Path Resolution**: Inspect source and target filesystem locations.")
            steps.append("2. **Filesystem Execution**: Perform file and folder manipulations safely.")
            steps.append("3. **Verification**: Confirm changes on disk and report status.")
        # General GUI / desktop
        elif any(w in prompt_lower for w in ["click", "open", "launch", "press", "type"]):
            steps.append("1. **Window / UI Targeting**: Locate application windows or screen coordinates.")
            steps.append("2. **Input Dispatch**: Execute native Wayland hardware pointer/keyboard events.")
            steps.append("3. **Visual Verification**: Confirm UI response via screen diffing.")
        else:
            steps.append("1. **Intent Analysis**: Determine required system capabilities and tools.")
            steps.append("2. **Autonomous Execution**: Invoke designated tools in sequence.")
            steps.append("3. **Outcome Delivery**: Present completed work and status.")

        return "\n".join(steps)

    def step(self, user_input: str) -> str:
        """Processes a user input through the autonomous ReAct cycle."""
        # Ultra-fast reflex route for common system tasks (<15ms)
        if self.enable_fast_path:
            fast_reply = self.fast_path.route(user_input)
            if fast_reply is not None:
                self.messages.append({"role": "user", "content": user_input})
                self.messages.append({"role": "assistant", "content": fast_reply})
                return fast_reply

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
                return f"I encountered an error communicating with my cognitive backend: {e}"

            choice = response.choices[0]
            message = choice.message

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

                    if not allowed:
                        result = ToolResult(success=False, output=None, error=reject_reason)
                    else:
                        # Ensure Jarvis's working workspace is active on the monitor before interacting
                        if fn_name in (
                            "click_element", "type_into_element", "click_coordinate", "move_mouse", "click_mouse",
                            "type_text_input", "press_shortcut", "scroll_page",
                            "take_screenshot", "inspect_screen", "locate_element", "capture_window_screenshot",
                            "open_browser_in_workspace", "scan_webpage_dom", "click_webpage_element",
                            "type_webpage_input", "execute_browser_chain", "focus_browser_address_bar"
                        ):
                            workspace_manager.ensure_working_workspace_active()

                        result = self.registry.execute(fn_name, args)

                    if not result.success:
                        self.failed_operations.append({
                            "tool": fn_name,
                            "error": result.error,
                            "args": args,
                        })

                    if self.on_tool_result:
                        self.on_tool_result(fn_name, result.output if result.success else result.error, result.success)

                    # Truncate overly long tool outputs to 600 chars to conserve Groq TPM
                    clean_content = result.to_message_content()
                    if len(clean_content) > 600:
                        clean_content = clean_content[:600] + "... [truncated to prevent rate limit]"

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
            self.messages.append({"role": "assistant", "content": content})
            return content

        if self.failed_operations:
            failure_lines = "\n".join([f"- **{f['tool']}**: {f['error']}" for f in self.failed_operations[-3:]])
            return (
                f"Sir, I was unable to complete the task because the following operations encountered issues:\n"
                f"{failure_lines}\n\n"
                f"Please let me know how you would like to proceed or if I should attempt an alternative method."
            )

        return "I have reached the maximum reasoning iterations for this task without reaching a final conclusion, Sir."
