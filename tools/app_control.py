import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional
import psutil

from core.workspace import workspace_manager
from tools.base import registry


def _is_hyprland() -> bool:
    """Checks if Hyprland is active and hyprctl is available."""
    return workspace_manager._is_hyprland()


@registry.register(description="Returns the currently active desktop workspace ID and name (e.g. {'id': 1, 'name': '1'}).")
def get_current_workspace() -> Dict[str, Any]:
    """Returns the ID and name of the currently active Hyprland workspace."""
    return workspace_manager.get_active_workspace()


def get_active_workspace() -> Dict[str, Any]:
    return workspace_manager.get_active_workspace()


@registry.register(description="If Jarvis's working workspace is not the active one on the screen, automatically switches to make it active.")
def make_working_workspace_active() -> Dict[str, Any]:
    return workspace_manager.ensure_working_workspace_active()


@registry.register(description="Gets the workspace ID where Jarvis is currently working versus the monitor's active workspace.")
def get_jarvis_workspace_status() -> Dict[str, Any]:
    return workspace_manager.get_status()


@registry.register(description="Sets the workspace ID where Jarvis should perform its tasks.")
def set_jarvis_working_workspace(workspace_id: int) -> str:
    workspace_manager.set_working_workspace(workspace_id)
    return f"Jarvis working workspace set to {workspace_id}."


@registry.register(description="Lists all currently open desktop windows with their titles, application classes, workspaces, and PIDs.")
def list_open_windows() -> List[Dict[str, Any]]:
    windows = []
    if _is_hyprland():
        try:
            res = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True, check=True)
            clients = json.loads(res.stdout)
            for c in clients:
                windows.append({
                    "title": c.get("title", ""),
                    "class": c.get("class", ""),
                    "pid": c.get("pid"),
                    "workspace": c.get("workspace", {}).get("name"),
                    "workspace_id": c.get("workspace", {}).get("id"),
                    "focused": c.get("focusHistoryID") == 0,
                    "address": c.get("address"),
                })
            return windows
        except Exception:
            pass

    # Generic fallback using psutil scanning UI processes
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            info = proc.info
            name = info.get("name", "").lower()
            if any(term in name for term in ["chrome", "code", "firefox", "kitty", "terminal", "gedit", "nautilus", "dolphin"]):
                windows.append({
                    "title": info.get("name"),
                    "class": info.get("name"),
                    "pid": info.get("pid"),
                    "workspace": "unknown",
                    "focused": False,
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return windows


COMMON_APP_ALIASES = {
    "fileexplorer": ["dolphin", "nautilus", "thunar", "nemo", "pcmanfm"],
    "files": ["dolphin", "nautilus", "thunar", "nemo", "pcmanfm"],
    "filemanager": ["dolphin", "nautilus", "thunar", "nemo", "pcmanfm"],
    "browser": ["google-chrome", "firefox", "chromium", "brave", "zen-browser"],
    "terminal": ["kitty", "alacritty", "wezterm", "foot", "gnome-terminal", "konsole", "xterm"],
    "editor": ["code", "cursor", "subl", "gedit", "kate", "nvim"],
}


def resolve_application(name: str) -> str:
    """Resolves generic application names to installed system binaries."""
    cleaned = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    for alias_key, candidates in COMMON_APP_ALIASES.items():
        if cleaned == alias_key:
            for candidate in candidates:
                if shutil.which(candidate):
                    return candidate
    return name


@registry.register(description="Launches an application by name or common alias (e.g., 'fileExplorer', 'google-chrome', 'code', 'kitty', 'browser'). Opens on the current active workspace.")
def launch_application(app_name: str, arguments: str = "") -> str:
    app_name = app_name.strip()
    target_app = resolve_application(app_name)
    active_ws = get_active_workspace()

    # If launching browser without explicit args, pass --new-window to guarantee
    # it appears on the current active workspace rather than attaching to a background window
    if "chrome" in target_app.lower() and not arguments:
        arguments = "--new-window"

    # 1. Direct binary execution in PATH
    exe = shutil.which(target_app)
    if exe:
        args_list = [exe] + (arguments.split() if arguments else [])
        proc = subprocess.Popen(args_list, start_new_session=True)
        time.sleep(0.5)
        return f"Successfully launched '{target_app}' (PID: {proc.pid}) on workspace {active_ws['name']}."

    # 2. Try desktop launcher if .desktop file exists
    desktop_file = f"{target_app}.desktop"
    desktop_paths = [
        Path("/usr/share/applications") / desktop_file,
        Path.home() / ".local/share/applications" / desktop_file,
    ]
    if any(p.exists() for p in desktop_paths) and shutil.which("gtk-launch"):
        try:
            subprocess.Popen(["gtk-launch", target_app], start_new_session=True)
            time.sleep(0.5)
            return f"Successfully launched '{target_app}' via desktop launcher on workspace {active_ws['name']}."
        except Exception:
            pass

    # 3. Fallback to xdg-open if URL or file path
    if shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", target_app], start_new_session=True)
        time.sleep(0.5)
        return f"Dispatched '{target_app}' via xdg-open."

    raise RuntimeError(f"Could not locate or launch application '{app_name}'.")


@registry.register(description="Opens a file, directory folder, or URL using the system's default GUI application.")
def open_path(path: str = ".") -> str:
    target = os.path.expanduser(path.strip())
    if shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", target], start_new_session=True)
        return f"Successfully opened '{target}' with default system application."
    elif shutil.which("gio"):
        subprocess.Popen(["gio", "open", target], start_new_session=True)
        return f"Dispatched '{target}' via gio open."
    else:
        raise RuntimeError("Neither 'xdg-open' nor 'gio' is available to open paths.")


@registry.register(description="Navigates directly to any website URL in the browser (e.g. 'https://www.chess.com/play/computer', 'https://github.com'). Guarantees the browser window opens on the current active workspace.")
def open_browser_url(url: str, new_window: bool = True) -> str:
    clean_url = url.strip()
    if not clean_url.startswith(("http://", "https://")):
        clean_url = f"https://{clean_url}"

    active_ws = get_active_workspace()
    chrome_bin = (
        shutil.which("google-chrome-stable")
        or shutil.which("google-chrome")
        or shutil.which("chromium")
    )

    # In multi-workspace environments (like Hyprland), --new-window guarantees
    # the browser window opens on the USER'S CURRENT WORKSPACE instead of silently
    # appending a tab to a window on an occluded or background workspace.
    if chrome_bin:
        cmd = [chrome_bin]
        if new_window:
            cmd.append("--new-window")
        cmd.append(clean_url)
        proc = subprocess.Popen(cmd, start_new_session=True)
        time.sleep(0.8)
        focus_window("chrome")
        return f"Navigated to '{clean_url}' in Google Chrome on workspace {active_ws['name']} (PID: {proc.pid})."
    elif shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", clean_url], start_new_session=True)
        time.sleep(0.8)
        return f"Opened '{clean_url}' in default browser on workspace {active_ws['name']}."
    raise RuntimeError("No supported web browser found.")


@registry.register(description="Focuses the top address bar/URL bar of the active web browser (Google Chrome or Firefox) using the universal accelerator shortcut (Ctrl+L). Prepares the browser for typing a URL or search.")
def focus_browser_address_bar() -> Dict[str, Any]:
    active_ws = get_active_workspace()
    focus_res = focus_window("chrome")
    time.sleep(0.15)
    from gui_driver.mouse_keyboard import controller
    controller.hotkey("ctrl+l")
    time.sleep(0.1)
    return {
        "success": True,
        "action": "focus_browser_address_bar",
        "message": "Successfully focused browser top address bar and highlighted URL field.",
        "workspace": active_ws.get("name"),
    }


@registry.register(description="Focuses an active desktop window matching the given application name or title keyword. Automatically switches to the window's workspace if located on a different workspace.")
def focus_window(query: str, bring_to_current_workspace: bool = False) -> Dict[str, Any]:
    query_clean = query.strip().lower()
    if _is_hyprland():
        try:
            active_ws = get_active_workspace()
            active_id = active_ws.get("id", 1)
            res = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True, check=True)
            clients = json.loads(res.stdout)
            matching = None
            for c in clients:
                title = (c.get("title") or "").lower()
                cls = (c.get("class") or "").lower()
                if query_clean in title or query_clean in cls:
                    matching = c
                    break

            if matching:
                win_ws = matching.get("workspace", {}).get("id", active_id)
                win_title = matching.get("title", "")
                win_class = matching.get("class", "")
                addr = matching.get("address")

                if bring_to_current_workspace and win_ws != active_id:
                    lua = f'for _, w in ipairs(hl.get_windows()) do if w.address == "{addr}" then hl.dispatch(hl.dsp.window.move({{ workspace = {active_id} }})); hl.dispatch(hl.dsp.focus({{ window = w }})); return "MOVED" end end return "FAIL"'
                    subprocess.run(["hyprctl", "repl", lua], capture_output=True, text=True)
                    time.sleep(0.15)
                    workspace_manager.set_working_workspace(active_id)
                    return {
                        "success": True,
                        "message": f"Moved '{win_title}' ({win_class}) from workspace {win_ws} to current workspace {active_id} and focused it.",
                        "window": win_title,
                        "class": win_class,
                        "workspace": active_id,
                    }
                else:
                    lua = f'for _, w in ipairs(hl.get_windows()) do if w.address == "{addr}" then if w.workspace and w.workspace.id then hl.dispatch(hl.dsp.focus({{ workspace = w.workspace.id }})) end; hl.dispatch(hl.dsp.focus({{ window = w }})); return "FOCUSED" end end return "FAIL"'
                    subprocess.run(["hyprctl", "repl", lua], capture_output=True, text=True)
                    time.sleep(0.15)
                    workspace_manager.set_working_workspace(win_ws)
                    ws_msg = f"switched to workspace {win_ws} and " if win_ws != active_id else ""
                    return {
                        "success": True,
                        "message": f"Successfully {ws_msg}focused '{win_title}' ({win_class}).",
                        "window": win_title,
                        "class": win_class,
                        "workspace": win_ws,
                    }
        except Exception as e:
            return {"success": False, "error": f"Failed while focusing window query '{query}': {e}"}

    return {"success": False, "error": f"No open window found matching query: '{query}'."}


@registry.register(description="Closes an active window matching the application name, title keyword, or window address.")
def close_window(query: str) -> Dict[str, Any]:
    query_clean = query.strip().lower()
    if _is_hyprland():
        try:
            lua = f'for _, w in ipairs(hl.get_windows()) do local t = string.lower(w.title or ""); local c = string.lower(w.class or ""); local q = "{query_clean}"; if string.find(t, q, 1, true) or string.find(c, q, 1, true) then hl.dispatch(hl.dsp.window.close({{ window = w }})); return string.format("OK|%s", w.title) end end; return "NOT_FOUND"'
            res = subprocess.run(["hyprctl", "repl", lua], capture_output=True, text=True)
            output = res.stdout.strip()
            if output.startswith("OK|"):
                title = output.split("|", 1)[1]
                return {"success": True, "message": f"Successfully closed window '{title}'."}
        except Exception as e:
            return {"success": False, "error": f"Failed while closing window '{query}': {e}"}

    return {"success": False, "error": f"No open window found matching: '{query}'."}


@registry.register(description="Switches the active desktop display viewport to a specific workspace number or name.")
def switch_to_workspace(workspace_id: int) -> str:
    if workspace_manager._is_hyprland():
        try:
            ws_int = int(workspace_id)
            if workspace_manager.switch_to_workspace(ws_int):
                workspace_manager.set_working_workspace(ws_int)
                return f"Successfully switched to workspace {workspace_id}."
            return f"Failed to switch to workspace {workspace_id}."
        except Exception as e:
            return f"Failed to switch to workspace {workspace_id}: {e}"
    return "Workspace switching is not supported in this environment."


@registry.register(description="Moves an open application window matching the query to a target workspace (or current workspace if omitted).")
def move_window_to_workspace(query: str, target_workspace: Optional[int] = None) -> str:
    query_clean = query.strip().lower()
    if _is_hyprland():
        try:
            active_ws = get_active_workspace()
            dest_ws = active_ws.get("id", 1) if target_workspace is None else int(target_workspace)
            res = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True, check=True)
            clients = json.loads(res.stdout)
            for c in clients:
                title = (c.get("title") or "").lower()
                cls = (c.get("class") or "").lower()
                if query_clean in title or query_clean in cls:
                    addr = c.get("address")
                    lua = f'for _, w in ipairs(hl.get_windows()) do if w.address == "{addr}" then hl.dispatch(hl.dsp.window.move({{ workspace = {dest_ws} }})); return "OK" end end return "FAIL"'
                    subprocess.run(["hyprctl", "repl", lua], capture_output=True, text=True)
                    time.sleep(0.1)
                    return f"Successfully moved '{c.get('title')}' to workspace {dest_ws}."
            return f"No open window matching '{query}' found."
        except Exception as e:
            return f"Failed to move window: {e}"
    return "Window workspace management is not supported in this environment."


@registry.register(description="Lists top running processes with optional name filtering and sorting by memory/CPU.")
def list_running_processes(filter_name: str = "", limit: int = 15) -> List[Dict[str, Any]]:
    filter_name = filter_name.strip().lower()
    procs = []

    for p in psutil.process_iter(["pid", "name", "username", "cpu_percent", "memory_percent"]):
        try:
            info = p.info
            name = (info.get("name") or "").lower()
            if filter_name and filter_name not in name:
                continue
            procs.append({
                "pid": info.get("pid"),
                "name": info.get("name"),
                "user": info.get("username"),
                "cpu_percent": round(info.get("cpu_percent") or 0.0, 1),
                "memory_percent": round(info.get("memory_percent") or 0.0, 1),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    procs.sort(key=lambda x: x["memory_percent"], reverse=True)
    return procs[:limit]


@registry.register(description="Terminates a process safely by PID with protection for system-critical processes.")
def terminate_process(pid: int) -> str:
    if pid in (0, 1) or pid == os.getpid():
        raise PermissionError(f"Refusing to terminate critical process PID {pid}.")

    try:
        proc = psutil.Process(pid)
        proc_name = proc.name()
        proc.terminate()
        proc.wait(timeout=3)
        return f"Process '{proc_name}' (PID: {pid}) was cleanly terminated."
    except psutil.TimeoutExpired:
        proc.kill()
        return f"Process (PID: {pid}) was forcefully killed."
    except psutil.NoSuchProcess:
        return f"No process found with PID {pid}."
    except psutil.AccessDenied as e:
        return f"Permission denied to terminate PID {pid}: {e}"
