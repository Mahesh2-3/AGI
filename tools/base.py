import inspect
import json
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, create_model


class ToolResult(BaseModel):
    """Encapsulates the result of a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None

    def to_message_content(self) -> str:
        """Formats the result into a clean string for the LLM context."""
        if self.success:
            if isinstance(self.output, (dict, list)):
                return json.dumps(self.output, indent=2)
            return str(self.output)
        details = ""
        if self.output is not None:
            details = f"\nDetails: {json.dumps(self.output, indent=2) if isinstance(self.output, (dict, list)) else self.output}"
        return (
            f"⚠️ OPERATION FAILED: {self.error}{details}\n"
            f"[SYSTEM DIRECTIVE]: You MUST explicitly inform the user in your chat response about this failed operation and its reason."
        )



class Tool:
    """Represents a callable tool with JSON schema generation."""

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        requires_confirmation: bool = False,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip()
        self.requires_confirmation = requires_confirmation
        self.schema = self._build_schema()

    def _build_schema(self) -> Dict[str, Any]:
        """Generates an OpenAI-compatible function tool schema from type hints."""
        sig = inspect.signature(self.func)
        fields = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            annotation = param.default if param.annotation == inspect.Parameter.empty else param.annotation
            default_val = ... if param.default == inspect.Parameter.empty else param.default
            fields[param_name] = (annotation if annotation != inspect.Parameter.empty else Any, default_val)

        pydantic_model = create_model(f"{self.name}Model", **fields)
        json_schema = pydantic_model.model_json_schema()

        properties = json_schema.get("properties", {})
        required = json_schema.get("required", [])

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def execute(self, **kwargs) -> ToolResult:
        """Executes the tool with parameter filtering and error handling."""
        try:
            sig = inspect.signature(self.func)
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if has_var_keyword:
                call_args = kwargs
            else:
                call_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
            result = self.func(**call_args)
            if isinstance(result, dict) and result.get("success") is False:
                err_msg = result.get("error") or "Operation returned success=False"
                return ToolResult(success=False, output=result, error=err_msg)
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class ToolRegistry:
    """Central registry for all tools available to the Jarvis agent."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        requires_confirmation: bool = False,
    ):
        """Decorator or function to register a tool."""
        def decorator(f: Callable) -> Callable:
            tool = Tool(
                func=f,
                name=name,
                description=description,
                requires_confirmation=requires_confirmation,
            )
            self._tools[tool.name] = tool
            return f

        if func is None:
            return decorator
        return decorator(func)

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_schemas(self) -> List[Dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]

    def execute(self, name: str, arguments: Dict[str, Any] | str) -> ToolResult:
        if name not in self._tools:
            return ToolResult(success=False, output=None, error=f"Tool '{name}' not found.")

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as err:
                return ToolResult(success=False, output=None, error=f"Invalid JSON arguments: {err}")

        tool = self._tools[name]
        return tool.execute(**arguments)

    @property
    def tools(self) -> Dict[str, Tool]:
        return self._tools


# Global default registry
registry = ToolRegistry()
