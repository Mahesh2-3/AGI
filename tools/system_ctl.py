import re
import shutil
import subprocess
from typing import Any, Dict
from tools.base import registry


@registry.register(description="Sends a native Linux desktop notification via notify-send.")
def send_desktop_notification(title: str, message: str, urgency: str = "normal") -> str:
    if not shutil.which("notify-send"):
        return "Notification tool 'notify-send' is not installed."

    if urgency not in ("low", "normal", "critical"):
        urgency = "normal"

    subprocess.run(["notify-send", "-u", urgency, "-a", "JARVIS", title, message], check=True)
    return f"Notification sent: [{title}] {message}"


@registry.register(description="Gets the current audio playback volume percentage and mute status.")
def get_audio_volume() -> Dict[str, Any]:
    if not shutil.which("pactl"):
        return {"error": "pactl audio controller is not available."}

    vol_res = subprocess.run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"], capture_output=True, text=True)
    mute_res = subprocess.run(["pactl", "get-sink-mute", "@DEFAULT_SINK@"], capture_output=True, text=True)

    match = re.search(r"(\d+)%", vol_res.stdout)
    volume_percent = int(match.group(1)) if match else None
    is_muted = "yes" in mute_res.stdout.lower()

    return {
        "volume_percent": volume_percent,
        "is_muted": is_muted,
    }


@registry.register(description="Sets the master audio playback volume percentage (0 to 100).")
def set_audio_volume(percent: int) -> str:
    if not shutil.which("pactl"):
        raise RuntimeError("pactl is not installed on this system.")

    clamped = max(0, min(100, percent))
    subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{clamped}%"], check=True)
    return f"Master audio volume set to {clamped}%."


@registry.register(description="Toggles audio mute on or off.")
def toggle_audio_mute() -> str:
    if not shutil.which("pactl"):
        raise RuntimeError("pactl is not installed on this system.")

    subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"], check=True)
    curr = get_audio_volume()
    status = "muted" if curr.get("is_muted") else "unmuted"
    return f"Audio is now {status}."


@registry.register(description="Retrieves active network status and internet connectivity.")
def get_network_status() -> Dict[str, Any]:
    status = {"connected": False, "state": "unknown"}

    if shutil.which("nmcli"):
        try:
            res = subprocess.run(
                ["nmcli", "-t", "-f", "RUNNING,STATE,CONNECTIVITY", "general", "status"],
                capture_output=True,
                text=True,
                check=True,
            )
            parts = res.stdout.strip().split(":")
            if len(parts) >= 3:
                status["running"] = parts[0]
                status["state"] = parts[1]
                status["connectivity"] = parts[2]
                status["connected"] = parts[2] in ("full", "limited")
                return status
        except Exception:
            pass

    return status
