import sys
import time
from typing import Any, Dict, List, Optional

# Mock mouseinfo to prevent tkinter requirement on headless/Linux setups
if "mouseinfo" not in sys.modules:
    sys.modules["mouseinfo"] = type("MockMouseInfo", (), {})()

import pyautogui

# Safety fail-safe (moving mouse to corner raises an exception if enabled, set True for safety)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

from tools.base import registry


class MouseKeyboardController:
    """Controls physical mouse and keyboard inputs on the desktop."""

    @staticmethod
    def get_position() -> Dict[str, int]:
        pos = pyautogui.position()
        return {"x": pos.x, "y": pos.y}

    @staticmethod
    def move_to(x: int, y: int, duration: float = 0.2) -> Dict[str, int]:
        pyautogui.moveTo(x, y, duration=duration)
        return {"x": x, "y": y}

    @staticmethod
    def click(x: Optional[int] = None, y: Optional[int] = None, button: str = "left", clicks: int = 1) -> Dict[str, Any]:
        if x is not None and y is not None:
            pyautogui.moveTo(x, y, duration=0.15)
        pyautogui.click(button=button, clicks=clicks)
        pos = pyautogui.position()
        return {"clicked_at": {"x": pos.x, "y": pos.y}, "button": button, "clicks": clicks}

    @staticmethod
    def double_click(x: Optional[int] = None, y: Optional[int] = None) -> Dict[str, Any]:
        return MouseKeyboardController.click(x=x, y=y, button="left", clicks=2)

    @staticmethod
    def right_click(x: Optional[int] = None, y: Optional[int] = None) -> Dict[str, Any]:
        return MouseKeyboardController.click(x=x, y=y, button="right", clicks=1)

    @staticmethod
    def drag_to(x: int, y: int, duration: float = 0.5) -> Dict[str, Any]:
        pyautogui.dragTo(x, y, duration=duration, button="left")
        return {"dragged_to": {"x": x, "y": y}}

    @staticmethod
    def scroll(amount: int) -> str:
        pyautogui.scroll(amount)
        direction = "up" if amount > 0 else "down"
        return f"Scrolled {direction} by {abs(amount)} clicks."

    @staticmethod
    def type_text(text: str, press_enter: bool = False, interval: float = 0.02) -> str:
        pyautogui.write(text, interval=interval)
        if press_enter:
            pyautogui.press("enter")
        return f"Typed {len(text)} characters{' and pressed Enter' if press_enter else ''}."

    @staticmethod
    def hotkey(keys: List[str] | str) -> str:
        if isinstance(keys, str):
            # Parse 'ctrl+c' or 'ctrl+alt+t'
            key_list = [k.strip().lower() for k in keys.split("+")]
        else:
            key_list = [k.strip().lower() for k in keys]

        pyautogui.hotkey(*key_list)
        return f"Pressed key combination: {'+'.join(key_list)}"


controller = MouseKeyboardController()


@registry.register(description="Moves mouse cursor to specific screen pixel coordinates (x, y).")
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
