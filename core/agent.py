import json
from typing import Any, Callable, Dict, List, Optional
from core.fast_path import FastPathRouter
from core.llm_client import LLMClient
from core.prompts import get_system_prompt
from core.safety import SafetyEngine
from core.workspace import workspace_manager
from tools.base import ToolRegistry, ToolResult, registry as default_registry


class JarvisAgent:
    """The central ReAct (Reason + Act) autonomous orchestrator for J.A.R.V.I.S."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        safety_engine: Optional[SafetyEngine] = None,
        max_iterations: int = 10,
        enable_fast_path: bool = True,
        on_thought: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_result: Optional[Callable[[str, Any, bool], None]] = None,
    ):
        self.llm = llm_client or LLMClient()
        self.registry = tool_registry or default_registry
        self.safety = safety_engine or SafetyEngine()
        self.max_iterations = max_iterations
        self.enable_fast_path = enable_fast_path
        self.fast_path = FastPathRouter(registry=self.registry, safety=self.safety)
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": get_system_prompt()}
        ]
        
        # Event callbacks for CLI / UI hooks
        self.on_thought = on_thought
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result

    def reset(self):
        """Clears working memory back to the initial system prompt."""
        self.messages = [
            {"role": "system", "content": get_system_prompt()}
        ]

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

    def step(self, user_input: str) -> str:
        """Processes a user input through the autonomous ReAct cycle."""
        # Ultra-fast reflex route for common system tasks (<15ms)
        if self.enable_fast_path:
            fast_reply = self.fast_path.route(user_input)
            if fast_reply is not None:
                self.messages.append({"role": "user", "content": user_input})
                self.messages.append({"role": "assistant", "content": fast_reply})
                return fast_reply

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
                            "click_element", "type_into_element", "move_mouse", "click_mouse",
                            "type_text_input", "press_shortcut", "scroll_page",
                            "take_screenshot", "inspect_screen", "locate_element", "capture_window_screenshot"
                        ):
                            workspace_manager.ensure_working_workspace_active()

                        result = self.registry.execute(fn_name, args)

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

        return "I have reached the maximum reasoning iterations for this task without reaching a final conclusion, Sir."
