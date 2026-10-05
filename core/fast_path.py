import re
from typing import Any, Dict, Optional
from config.settings import settings
from core.safety import SafetyEngine
from tools.base import ToolRegistry, ToolResult, registry as default_registry


class FastPathRouter:
    """Deterministic, zero-latency intent router for common reflex tasks.
    
    Bypasses cloud LLM roundtrips (<15ms execution) for high-frequency commands
    such as time queries, volume toggles, app launches, telemetry, and screenshots.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None, safety: Optional[SafetyEngine] = None):
        self.registry = registry or default_registry
        self.safety = safety or SafetyEngine()

    def route(self, user_input: str) -> Optional[str]:
        """Inspects user input. If a deterministic match is found, executes the action and returns Jarvis's response.
        Otherwise returns None to fall back to the ReAct LLM engine."""
        text = user_input.strip()
        lower = text.lower()

        # Clean leading wake word if present
        lower = re.sub(r"^(?:hey |ok |okay )?jarvis[,:\s]*", "", lower).strip()

        # 1. Current Time
        if re.search(r"^(?:what(?:'s| is) the time|current time|what time is it|tell me the time)\??$", lower):
            res = self.registry.execute("get_current_time", {})
            if res.success:
                return f"The current time is {res.output}, Sir."

        # 2. System Status / Telemetry
        if re.search(r"^(?:system )?(?:telemetry|status|health)\??$", lower) or lower in ("how are you doing", "status"):
            res = self.registry.execute("get_system_telemetry", {})
            if res.success:
                data = res.output
                if isinstance(data, dict):
                    cpu = data.get("cpu_percent", "N/A")
                    mem = data.get("ram_used_percent", "N/A")
                    uptime = data.get("uptime_hours", "N/A")
                    return (
                        f"All core systems operational, Sir. CPU load is at {cpu}%, "
                        f"memory utilisation is {mem}%, with an uptime of {uptime} hours."
                    )
                return f"System telemetry report, Sir:\n{data}"

        # 3. Mute / Unmute
        if re.search(r"^(?:mute|unmute|toggle mute)(?: (?:the )?(?:sound|audio|volume))?\??$", lower):
            res = self.registry.execute("toggle_audio_mute", {})
            if res.success:
                return f"{res.output}, Sir."

        # 4. Volume Up / Down / Set
        m_vol_set = re.search(r"^(?:set (?:the )?volume to |volume )(\d+)%?\??$", lower)
        if m_vol_set:
            val = int(m_vol_set.group(1))
            res = self.registry.execute("set_audio_volume", {"percent": val})
            if res.success:
                return f"System volume adjusted to {val}%, Sir."

        m_vol_up = re.search(r"^(?:turn (?:the )?volume up|volume up|increase (?:the )?volume)(?: by (\d+))?%?\??$", lower)
        if m_vol_up:
            delta = int(m_vol_up.group(1)) if m_vol_up.group(1) else 10
            cur_res = self.registry.execute("get_audio_volume", {})
            cur_vol = 50
            if cur_res.success and isinstance(cur_res.output, dict):
                cur_vol = cur_res.output.get("volume_percent") or 50
            new_vol = min(100, cur_vol + delta)
            res = self.registry.execute("set_audio_volume", {"percent": new_vol})
            if res.success:
                return f"Increased volume to {new_vol}%, Sir."

        m_vol_down = re.search(r"^(?:turn (?:the )?volume down|volume down|decrease (?:the )?volume|lower (?:the )?volume)(?: by (\d+))?%?\??$", lower)
        if m_vol_down:
            delta = int(m_vol_down.group(1)) if m_vol_down.group(1) else 10
            cur_res = self.registry.execute("get_audio_volume", {})
            cur_vol = 50
            if cur_res.success and isinstance(cur_res.output, dict):
                cur_vol = cur_res.output.get("volume_percent") or 50
            new_vol = max(0, cur_vol - delta)
            res = self.registry.execute("set_audio_volume", {"percent": new_vol})
            if res.success:
                return f"Lowered volume to {new_vol}%, Sir."

        # 5. Take Screenshot
        if re.search(r"^(?:take (?:a )?screenshot|capture (?:the )?screen|take screenshot)\??$", lower):
            res = self.registry.execute("take_screenshot", {"include_cursor": False})
            if res.success:
                latency = res.output.get("capture_latency_ms", "N/A") if isinstance(res.output, dict) else "N/A"
                path = res.output.get("path", "") if isinstance(res.output, dict) else ""
                return f"Desktop visual captured in {latency}ms, Sir. Saved to `{path}`."

        # 6. Launch Application
        m_app = re.search(r"^(?:open|launch|start) (?:the )?([a-zA-Z0-9_\-\.]+?)(?: application| app)?(?: please)?\??$", lower)
        if m_app:
            target = m_app.group(1).strip()
            # Do not intercept if it looks like a file or url
            if not target.startswith("http") and not any(target.endswith(ext) for ext in [".txt", ".py", ".md", ".json", ".pdf"]):
                res = self.registry.execute("launch_application", {"app_name": target})
                if res.success:
                    app_name = res.output.get("app", target) if isinstance(res.output, dict) else target
                    pid = res.output.get("pid") if isinstance(res.output, dict) else None
                    pid_str = f" (PID {pid})" if pid else ""
                    return f"Right away, Sir. {app_name.capitalize()} has been launched{pid_str}."

        # 7. Close Application
        m_close = re.search(r"^(?:close|kill|quit) (?:the )?([a-zA-Z0-9_\-\.]+?)(?: application| app)?(?: please)?\??$", lower)
        if m_close:
            target = m_close.group(1).strip()
            # Check safety authorization
            allowed, reason = self.safety.authorize("close_application", {"app_name": target}, requires_confirmation=False)
            if not allowed:
                return f"Action declined by safety protocol: {reason}"
            res = self.registry.execute("close_application", {"app_name": target})
            if res.success:
                return f"Terminated {target}, Sir."
            return f"Unable to locate active process for '{target}', Sir."

        # 8. List Directory
        m_ls = re.search(r"^(?:list files|show files|dir|ls)(?: in (?:the )?([^\?]+))?\??$", lower)
        if m_ls:
            target_path = m_ls.group(1).strip() if m_ls.group(1) else "."
            res = self.registry.execute("list_directory", {"path": target_path})
            if res.success:
                out = res.output
                if isinstance(out, dict):
                    entries = out.get("entries", [])
                    preview = ", ".join(entries[:15]) + ("..." if len(entries) > 15 else "")
                    return f"Contents of `{target_path}` ({len(entries)} items):\n{preview}"
                return f"Directory contents:\n{out}"

        # 9. Workspace Activation & Synchronization
        workspace_match = (
            any(k in lower for k in ["jarvis working workspace", "working workspace", "jarvis workspace", "workspace of jarvis"])
            and any(k in lower for k in ["active", "switch", "activate", "make", "focus", "ensure", "current"])
        ) or any(k in lower for k in [
            "make jarvis workspace active",
            "switch to jarvis workspace",
            "activate jarvis workspace",
            "ensure working workspace active",
        ])
        if workspace_match:
            res = self.registry.execute("make_working_workspace_active", {})
            if res.success and isinstance(res.output, dict):
                return f"{res.output.get('message', 'Workspace synchronized.')}, Sir."

        # 10. VLM Availability Status
        if any(phrase in lower for phrase in [
            "vlm status",
            "vision model status",
            "vision models available",
            "how many vision models",
            "how many vlms",
        ]):
            res = self.registry.execute("get_vlm_status", {})
            if res.success and isinstance(res.output, dict):
                return f"Vision status report, Sir: {res.output.get('summary', 'operational')}."

        return None
