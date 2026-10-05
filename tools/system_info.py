import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import psutil
from tools.base import registry


@registry.register(description="Retrieves live system telemetry including CPU, Memory, Disk usage, and Battery status.")
def get_system_telemetry() -> Dict[str, Any]:
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    
    battery = psutil.sensors_battery()
    battery_info = None
    if battery:
        battery_info = {
            "percent": battery.percent,
            "plugged_in": battery.power_plugged,
            "seconds_left": battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else "unlimited",
        }

    boot_time = datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "cpu_usage_percent": cpu_percent,
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_used_gb": round(memory.used / (1024 ** 3), 2),
        "memory_total_gb": round(memory.total / (1024 ** 3), 2),
        "memory_percent": memory.percent,
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
        "disk_total_gb": round(disk.total / (1024 ** 3), 2),
        "battery": battery_info,
        "boot_time": boot_time,
    }


@registry.register(description="Returns the current local date, time, and timezone.")
def get_current_time() -> Dict[str, str]:
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": time.tzname[0],
        "iso": now.isoformat(),
    }


@registry.register(description="Lists files and subdirectories within a given folder path.")
def list_directory(path: str = ".") -> List[Dict[str, Any]]:
    target = Path(path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not target.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    entries = []
    for item in sorted(target.iterdir()):
        try:
            stat = item.stat()
            entries.append({
                "name": item.name,
                "is_dir": item.is_dir(),
                "size_bytes": stat.st_size if not item.is_dir() else None,
            })
        except PermissionError:
            entries.append({
                "name": item.name,
                "is_dir": item.is_dir(),
                "error": "Permission denied",
            })
    return entries
