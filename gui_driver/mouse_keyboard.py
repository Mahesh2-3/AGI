import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

# Mock mouseinfo to prevent tkinter requirement on headless/Linux setups
if "mouseinfo" not in sys.modules:
    sys.modules["mouseinfo"] = type("MockMouseInfo", (), {})()

import pyautogui

# Safety fail-safe
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02

from core.workspace import workspace_manager
from tools.base import registry


class MouseKeyboardController:
    """Controls physical mouse and keyboard inputs on Linux desktops (Hyprland Wayland and X11)."""

    @staticmethod
    def _is_hyprland() -> bool:
        return bool(shutil.which("hyprctl")) and os.environ.get("XDG_CURRENT_DESKTOP") == "Hyprland"

    @classmethod
    def get_position(cls) -> Dict[str, int]:
        """Returns the true current desktop cursor coordinates."""
        if cls._is_hyprland():
            try:
                res = subprocess.run(["hyprctl", "cursorpos"], capture_output=True, text=True)
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) == 2:
                    return {"x": int(parts[0]), "y": int(parts[1])}
            except Exception:
                pass

        try:
            pos = pyautogui.position()
            return {"x": pos.x, "y": pos.y}
        except Exception:
            return {"x": 0, "y": 0}

    @classmethod
    def move_to(cls, x: int, y: int, duration: float = 0.15) -> Dict[str, int]:
        """Moves cursor to target pixel coordinates with visual glide animation."""
        workspace_manager.ensure_working_workspace_active()
        cur = cls.get_position()
        cur_x, cur_y = cur.get("x", x), cur.get("y", y)

        if cls._is_hyprland():
            try:
                steps = 5
                for i in range(1, steps + 1):
                    inter_x = int(cur_x + (x - cur_x) * (i / steps))
                    inter_y = int(cur_y + (y - cur_y) * (i / steps))
                    subprocess.run(
                        ["hyprctl", "dispatch", f"hl.dsp.cursor.move({{x={inter_x}, y={inter_y}}})"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    time.sleep(duration / steps)
            except Exception:
                pass

        try:
            import pynput
            pynput.mouse.Controller().position = (x, y)
        except Exception:
            pass

        try:
            pyautogui.moveTo(x, y, duration=0)
        except Exception:
            pass

        return {"x": x, "y": y}

    @classmethod
    def click(cls, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", clicks: int = 1) -> Dict[str, Any]:
        """Moves to coordinates and executes native mouse click."""
        if x is not None and y is not None:
            cls.move_to(x, y)

        try:
            import pynput
            mouse_btn = (
                pynput.mouse.Button.left
                if button == "left"
                else (pynput.mouse.Button.right if button == "right" else pynput.mouse.Button.middle)
            )
            mouse = pynput.mouse.Controller()
            for _ in range(clicks):
                mouse.click(mouse_btn)
                time.sleep(0.04)
        except Exception:
            pass

        try:
            pyautogui.click(button=button, clicks=clicks)
        except Exception:
            pass

        pos = cls.get_position()
        return {"clicked_at": {"x": pos.get("x", x or 0), "y": pos.get("y", y or 0)}, "button": button, "clicks": clicks}

    @classmethod
    def double_click(cls, x: Optional[int] = None, y: Optional[int] = None) -> Dict[str, Any]:
        return cls.click(x=x, y=y, button="left", clicks=2)

    @classmethod
    def right_click(cls, x: Optional[int] = None, y: Optional[int] = None) -> Dict[str, Any]:
        return cls.click(x=x, y=y, button="right", clicks=1)

    @classmethod
    def drag_to(cls, x: int, y: int, duration: float = 0.4) -> Dict[str, Any]:
        pyautogui.dragTo(x, y, duration=duration, button="left")
        return {"dragged_to": {"x": x, "y": y}}

    @classmethod
    def scroll(cls, amount: int) -> str:
        try:
            import pynput
            pynput.mouse.Controller().scroll(0, amount)
        except Exception:
            pyautogui.scroll(amount)
        direction = "up" if amount > 0 else "down"
        return f"Scrolled {direction} by {abs(amount)} clicks."

    @classmethod
    def type_text(cls, text: str, press_enter: bool = False, interval: float = 0.01) -> str:
        workspace_manager.ensure_working_workspace_active()
        try:
            import pynput
            kb = pynput.keyboard.Controller()
            for char in text:
                kb.type(char)
                time.sleep(interval)
            if press_enter:
                kb.tap(pynput.keyboard.Key.enter)
        except Exception:
            pyautogui.write(text, interval=interval)
            if press_enter:
                pyautogui.press("enter")
        return f"Typed {len(text)} characters{' and pressed Enter' if press_enter else ''}."

    @classmethod
    def hotkey(cls, keys: List[str] | str) -> str:
        workspace_manager.ensure_working_workspace_active()
        if isinstance(keys, str):
            key_list = [k.strip().lower() for k in keys.split("+")]
        else:
            key_list = [k.strip().lower() for k in keys]

        pyautogui.hotkey(*key_list)
        return f"Pressed key combination: {'+'.join(key_list)}"


controller = MouseKeyboardController()


@registry.register(description="Moves mouse cursor smoothly to specific screen pixel coordinates (x, y).")
def move_mouse(x: int, y: int) -> Dict[str, Any]:
    return controller.move_to(x, y)


@registry.register(description="Clicks the mouse at current position or at target coordinates (x, y). Supports 'left', 'right', 'double'.")
def click_mouse(x: Optional[int] = None, y: Optional[int] = None, action: str = "left") -> Dict[str, Any]:
    if action == "double":
        return controller.double_click(x, y)
    elif action == "right":
        return controller.right_click(x, y)
    else:
        return controller.click(x, y, button="left")


@registry.register(description="Types text via simulated keyboard with optional Enter keypress.")
def type_text_input(text: str, press_enter: bool = False) -> str:
    return controller.type_text(text, press_enter=press_enter)


@registry.register(description="Triggers a keyboard shortcut hotkey (e.g. 'ctrl+c', 'ctrl+v', 'alt+tab', 'ctrl+w', 'super+d').")
def press_shortcut(hotkey: str) -> str:
    return controller.hotkey(hotkey)


@registry.register(description="Scrolls the active window page up or down (positive = up, negative = down).")
def scroll_page(clicks: int = -5) -> str:
    return controller.scroll(clicks)
