import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional
import psutil

logger = logging.getLogger("jarvis.workspace")


class WorkspaceManager:
    """Tracks and enforces the active workspace for Jarvis desktop operations."""

    def __init__(self):
        self._working_workspace: Optional[int] = None
        self._jarvis_process_workspace: Optional[int] = None

    def _is_hyprland(self) -> bool:
        return bool(shutil.which("hyprctl")) and os.environ.get("XDG_CURRENT_DESKTOP") == "Hyprland"

    def get_active_workspace(self) -> Dict[str, Any]:
        """Queries the currently active workspace displayed on the monitor."""
        if self._is_hyprland():
            try:
                res = subprocess.run(["hyprctl", "activeworkspace", "-j"], capture_output=True, text=True, check=True)
                data = json.loads(res.stdout)
                return {
                    "id": int(data.get("id", 1)),
                    "name": str(data.get("name", "1")),
                }
            except Exception as e:
                logger.debug(f"Failed to query active workspace: {e}")
        return {"id": 1, "name": "1"}

    def get_jarvis_process_workspace(self) -> Optional[int]:
        """Detects the workspace where the Jarvis terminal / process window lives."""
        if not self._is_hyprland():
            return None

        try:
            cur_pid = os.getpid()
            pids = set()
            p = psutil.Process(cur_pid)
            while p:
                pids.add(p.pid)
                try:
                    p = p.parent()
                except Exception:
                    break

            res = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True)
            clients = json.loads(res.stdout)
            for c in clients:
                if c.get("pid") in pids:
                    ws_id = c.get("workspace", {}).get("id")
                    if ws_id is not None:
                        self._jarvis_process_workspace = int(ws_id)
                        return self._jarvis_process_workspace
        except Exception as e:
            logger.debug(f"Failed to find Jarvis process workspace: {e}")

        return self._jarvis_process_workspace

    def set_working_workspace(self, workspace_id: int | str):
        """Sets the designated workspace where Jarvis is currently performing tasks."""
        try:
            self._working_workspace = int(workspace_id)
            logger.info(f"Jarvis working workspace set to: {self._working_workspace}")
        except (ValueError, TypeError):
            pass

    def get_working_workspace(self) -> int:
        """Returns the designated working workspace (or falls back to active/process workspace)."""
        if self._working_workspace is not None:
            return self._working_workspace

        proc_ws = self.get_jarvis_process_workspace()
        if proc_ws is not None:
            self._working_workspace = proc_ws
            return proc_ws

        active = self.get_active_workspace()
        self._working_workspace = active["id"]
        return self._working_workspace

    def switch_to_workspace(self, workspace_id: int | str) -> bool:
        """Switches the monitor display viewport to the given workspace."""
        if not self._is_hyprland():
            return False

        try:
            ws_int = int(workspace_id)
            lua = f'return hl.dispatch(hl.dsp.focus({{ workspace = {ws_int} }}))'
            res = subprocess.run(["hyprctl", "repl", lua], capture_output=True, text=True)
            if res.returncode == 0:
                time.sleep(0.15)
                return True
        except Exception as e:
            logger.error(f"Failed to switch workspace: {e}")
        return False

    def ensure_working_workspace_active(self) -> Dict[str, Any]:
        """Checks if the working workspace is active. If not, automatically switches to make it active."""
        target_ws = self.get_working_workspace()
        active = self.get_active_workspace()
        active_id = active.get("id")

        if active_id != target_ws:
            logger.info(f"Jarvis working workspace ({target_ws}) is not active ({active_id}). Activating...")
            switched = self.switch_to_workspace(target_ws)
            new_active = self.get_active_workspace()
            return {
                "was_active": False,
                "switched": switched,
                "target_workspace": target_ws,
                "active_workspace": new_active["id"],
                "message": f"Switched display from workspace {active_id} to Jarvis working workspace {target_ws}.",
            }

        return {
            "was_active": True,
            "switched": False,
            "target_workspace": target_ws,
            "active_workspace": active_id,
            "message": f"Jarvis working workspace {target_ws} is already active.",
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns complete workspace status: active, working, process."""
        active = self.get_active_workspace()
        working = self.get_working_workspace()
        proc = self.get_jarvis_process_workspace()
        return {
            "active_workspace": active["id"],
            "working_workspace": working,
            "process_workspace": proc,
            "is_working_workspace_active": active["id"] == working,
        }


workspace_manager = WorkspaceManager()
