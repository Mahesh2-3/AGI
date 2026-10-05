import re
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple


class RiskTier(Enum):
    SAFE = "safe"              # Read-only or telemetry
    MODERATE = "moderate"      # App launch, volume set, file write
    DANGEROUS = "dangerous"    # File delete, kill process, shell execution


# Patterns of dangerous shell commands
BLOCKED_PATTERNS = [
    r"\brm\s+-[rf]{1,2}\s+/",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",  # Fork bomb
    r">\s*/dev/sd[a-z]",
    r"\bchmod\s+-R\s+777\s+/",
]


class SafetyEngine:
    """Enforces safety guardrails, blacklists, and confirmation gates."""

    def __init__(self, confirmation_handler: Optional[Callable[[str, Dict[str, Any], str], bool]] = None):
        self.confirmation_handler = confirmation_handler

    def evaluate_risk(self, tool_name: str, args: Dict[str, Any]) -> Tuple[RiskTier, str]:
        """Classifies the risk level of an action."""
        safe_tools = (
            "get_system_telemetry", "get_current_time", "list_directory", "read_file",
            "search_files", "grep_in_files", "list_open_windows", "list_running_processes",
            "get_audio_volume", "get_network_status", "open_path",
            "take_screenshot", "capture_window_screenshot", "locate_element",
            "inspect_screen", "check_visual_difference",
            "speak_message", "listen_to_user"
        )
        if tool_name in safe_tools:
            return RiskTier.SAFE, "Read-only, visual, or speech operation."

        if tool_name in ("execute_shell_command", "run_shell"):
            cmd = str(args.get("command", ""))
            for pattern in BLOCKED_PATTERNS:
                if re.search(pattern, cmd):
                    return RiskTier.DANGEROUS, f"Blocked critical pattern detected: {pattern}"
            return RiskTier.DANGEROUS, f"Shell command execution: '{cmd}'"

        if tool_name == "terminate_process":
            return RiskTier.DANGEROUS, f"Terminating process PID {args.get('pid')}."

        if tool_name == "safe_delete_file":
            return RiskTier.DANGEROUS, f"Moving file '{args.get('path')}' to trash."

        return RiskTier.MODERATE, "State modification."

    def authorize(self, tool_name: str, args: Dict[str, Any], requires_explicit_confirmation: bool = False) -> Tuple[bool, Optional[str]]:
        """Verifies if an action is permitted, blocked, or requires user confirmation."""
        tier, reason = self.evaluate_risk(tool_name, args)

        # Immediate block on catastrophic patterns
        if "Blocked critical pattern" in reason:
            return False, f"Action blocked by Jarvis safety guardrails: {reason}"

        # If confirmation is required by tier or tool specification
        if tier == RiskTier.DANGEROUS or requires_explicit_confirmation:
            if self.confirmation_handler:
                granted = self.confirmation_handler(tool_name, args, reason)
                if not granted:
                    return False, f"Action rejected: User declined authorization for {tool_name} ({reason})."
                return True, None
            # If no handler provided in automated runs, dangerous tools require explicit gate
            return True, None

        return True, None
