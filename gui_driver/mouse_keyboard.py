import logging
import os
from pathlib import Path
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

logger = logging.getLogger("jarvis.gui_driver")

WAYLAND_POINTER_BIN = Path(__file__).parent / "bin" / "wayland_pointer"


def ensure_wayland_pointer_binary() -> Optional[Path]:
    """Ensures the compiled wayland_pointer binary is available for native Wayland clicks."""
    if WAYLAND_POINTER_BIN.exists() and os.access(WAYLAND_POINTER_BIN, os.X_OK):
        return WAYLAND_POINTER_BIN

    src_dir = Path(__file__).parent / "src"
    src_file = src_dir / "wayland_pointer.c"
    proto_file = src_dir / "wlr-virtual-pointer-protocol.c"

    if src_file.exists() and proto_file.exists() and shutil.which("gcc"):
        try:
            WAYLAND_POINTER_BIN.parent.mkdir(parents=True, exist_ok=True)
            res = subprocess.run(
                [
                    "gcc",
                    str(src_file),
                    str(proto_file),
                    "-o",
                    str(WAYLAND_POINTER_BIN),
                    "-lwayland-client",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0:
                WAYLAND_POINTER_BIN.chmod(0o755)
                return WAYLAND_POINTER_BIN
        except Exception as e:
            logger.debug(f"Failed to compile wayland_pointer binary: {e}")

    return None


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
    def move_to(cls, x: int, y: int, duration: float = 0.0) -> Dict[str, int]:
        """Moves cursor to target pixel coordinates with zero latency (<2ms)."""
        workspace_manager.ensure_working_workspace_active()
        if cls._is_hyprland():
            try:
                # Instant pixel-perfect placement in Hyprland global coordinates
                subprocess.run(
                    ["hyprctl", "dispatch", f"hl.dsp.cursor.move({{x={x}, y={y}}})"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
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
        """Moves to coordinates and executes native mouse click across Wayland & X11."""
        workspace_manager.ensure_working_workspace_active()
        if x is not None and y is not None:
            cls.move_to(x, y)

        wp_bin = ensure_wayland_pointer_binary()
        wayland_sent = False

        # Native Wayland click via virtual pointer protocol
        if cls._is_hyprland() and wp_bin:
            try:
                cmd = [str(wp_bin), "click", button, str(clicks)]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                if res.returncode == 0:
                    wayland_sent = True
            except Exception as e:
                logger.debug(f"Wayland virtual pointer click failed: {e}")

        # Fallback / Dual-dispatch via X11 (pynput + pyautogui) for XWayland windows
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
                time.sleep(0.02)
        except Exception:
            pass

        try:
            pyautogui.click(button=button, clicks=clicks)
        except Exception:
            pass

        pos = cls.get_position()
        return {
            "success": True,
            "clicked_at": {"x": pos.get("x", x or 0), "y": pos.get("y", y or 0)},
            "button": button,
            "clicks": clicks,
            "native_wayland": wayland_sent,
        }

    @classmethod
    def double_click(cls, x: Optional[int] = None, y: Optional[int] = None) -> Dict[str, Any]:
        return cls.click(x=x, y=y, button="left", clicks=2)

    @classmethod
    def right_click(cls, x: Optional[int] = None, y: Optional[int] = None) -> Dict[str, Any]:
        return cls.click(x=x, y=y, button="right", clicks=1)

    @classmethod
    def drag_to(cls, x: int, y: int, duration: float = 0.4) -> Dict[str, Any]:
        workspace_manager.ensure_working_workspace_active()
        wp_bin = ensure_wayland_pointer_binary()
        if cls._is_hyprland() and wp_bin:
            try:
                subprocess.run([str(wp_bin), "down", "left"], timeout=2)
                cls.move_to(x, y, duration=duration)
                subprocess.run([str(wp_bin), "up", "left"], timeout=2)
                return {"dragged_to": {"x": x, "y": y}}
            except Exception:
                pass

        pyautogui.dragTo(x, y, duration=duration, button="left")
        return {"dragged_to": {"x": x, "y": y}}

    @classmethod
    def scroll(cls, amount: int) -> str:
        workspace_manager.ensure_working_workspace_active()
        wp_bin = ensure_wayland_pointer_binary()
        if cls._is_hyprland() and wp_bin:
            try:
                delta = -amount * 15
                subprocess.run([str(wp_bin), "scroll", str(delta)], timeout=2)
            except Exception:
                pass

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
        typed_via_hypr = False
        if cls._is_hyprland():
            try:
                for ch in text:
                    if ch == "\n":
                        lua = 'return hl.dispatch(hl.dsp.send_shortcut({ mods = "", key = "Return" }))'
                    elif ch == " ":
                        lua = 'return hl.dispatch(hl.dsp.send_shortcut({ mods = "", key = "space" }))'
                    elif ch.isupper():
                        lua = f'return hl.dispatch(hl.dsp.send_shortcut({{ mods = "shift", key = "{ch.lower()}" }}))'
                    elif ch in ('"', "'", "\\"):
                        escaped = f"\\{ch}" if ch == '"' else ch
                        lua = f'return hl.dispatch(hl.dsp.send_shortcut({{ mods = "", key = "{escaped}" }}))'
                    else:
                        lua = f'return hl.dispatch(hl.dsp.send_shortcut({{ mods = "", key = "{ch}" }}))'
                    subprocess.run(["hyprctl", "repl", lua], capture_output=True, timeout=1)
                    time.sleep(interval)

                if press_enter:
                    lua_enter = 'return hl.dispatch(hl.dsp.send_shortcut({ mods = "", key = "Return" }))'
                    subprocess.run(["hyprctl", "repl", lua_enter], capture_output=True, timeout=1)
                typed_via_hypr = True
            except Exception as e:
                logger.debug(f"Hyprland native typing error: {e}")

        # Also send via X11 (pynput + pyautogui)
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

        if cls._is_hyprland() and key_list:
            try:
                key = key_list[-1]
                mods = " ".join(key_list[:-1])
                lua = f'return hl.dispatch(hl.dsp.send_shortcut({{ mods = "{mods}", key = "{key}" }}))'
                subprocess.run(["hyprctl", "repl", lua], capture_output=True, text=True, timeout=2)
            except Exception as e:
                logger.debug(f"Hyprland send_shortcut hotkey error: {e}")

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
