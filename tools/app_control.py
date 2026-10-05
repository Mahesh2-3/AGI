import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
import psutil
from tools.base import registry


def _is_hyprland() -> bool:
    """Checks if Hyprland is active and hyprctl is available."""
    return bool(shutil.which("hyprctl")) and os.environ.get("XDG_CURRENT_DESKTOP") == "Hyprland"


@registry.register(description="Lists all currently open desktop windows with their titles, application classes, and PIDs.")
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
            if any(term in name for term in ["chrome", "code", "firefox", "kitty", "terminal", "gedit", "nautilus"]):
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


@registry.register(description="Launches an application by name or common alias (e.g., 'fileExplorer', 'google-chrome', 'code', 'kitty', 'browser').")
def launch_application(app_name: str, arguments: str = "") -> str:
    app_name = app_name.strip()
    target_app = resolve_application(app_name)

    # 1. Direct binary execution in PATH
    exe = shutil.which(target_app)
    if exe:
        args_list = [exe] + (arguments.split() if arguments else [])
        proc = subprocess.Popen(args_list, start_new_session=True)
        return f"Successfully launched '{target_app}' (PID: {proc.pid})."

    # 2. Try desktop launcher if .desktop file exists
    desktop_file = f"{target_app}.desktop"
    desktop_paths = [
        Path("/usr/share/applications") / desktop_file,
        Path.home() / ".local/share/applications" / desktop_file,
    ]
    if any(p.exists() for p in desktop_paths) and shutil.which("gtk-launch"):
        try:
            subprocess.Popen(["gtk-launch", target_app], start_new_session=True)
            return f"Successfully launched '{target_app}' via desktop launcher."
        except Exception:
            pass

    # 3. Fallback to xdg-open if URL or file path
    if shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", target_app], start_new_session=True)
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


@registry.register(description="Navigates directly to any website URL in the browser (e.g. 'https://www.chess.com/play/computer', 'https://github.com').")
def open_browser_url(url: str) -> str:
    clean_url = url.strip()
    if not clean_url.startswith(("http://", "https://")):
        clean_url = f"https://{clean_url}"

    # Try launching directly with google-chrome or default browser
    if shutil.which("google-chrome"):
        subprocess.Popen(["google-chrome", clean_url], start_new_session=True)
        return f"Navigated to '{clean_url}' in Google Chrome."
    elif shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", clean_url], start_new_session=True)
        return f"Opened '{clean_url}' in web browser."
    raise RuntimeError("No supported web browser found.")


@registry.register(description="Focuses an active desktop window matching the given application name or title keyword.")
def focus_window(query: str) -> str:
    query = query.strip().lower()
    if _is_hyprland():
        try:
            lua = f'for _, w in ipairs(hl.get_windows()) do if string.find(string.lower(w.title), "{query}", 1, true) or string.find(string.lower(w.class), "{query}", 1, true) then hl.dispatch(hl.dsp.focus({{ window = w }})) return w.title end end'
            res = subprocess.run(["hyprctl", "eval", lua], capture_output=True, text=True)
            if res.returncode == 0:
                return f"Successfully focused window matching '{query}'."
        except Exception:
            pass

    return f"No open window found matching query: '{query}'."


@registry.register(description="Closes an active window matching the application name, title keyword, or window address.")
def close_window(query: str) -> str:
    query = query.strip().lower()
    if _is_hyprland():
        try:
            lua = f'for _, w in ipairs(hl.get_windows()) do if string.find(string.lower(w.title), "{query}", 1, true) or string.find(string.lower(w.class), "{query}", 1, true) then hl.dispatch(hl.dsp.window.close({{ window = w }})) return w.title end end'
            res = subprocess.run(["hyprctl", "eval", lua], capture_output=True, text=True)
            if res.returncode == 0:
                return f"Successfully closed window matching '{query}'."
        except Exception:
            pass

    return f"No open window found matching: '{query}'."


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
