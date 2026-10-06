import time
from typing import Any, Dict, Optional
from tools.base import registry
from vision.capture import capture_engine
from vision.grounding import grounding_engine
from vision.visual_diff import diff_engine
from gui_driver.mouse_keyboard import controller


class ClosedLoopNavigator:
    """Orchestrates closed-loop desktop interaction: Observe -> Locate -> Act -> Verify."""

    def click_element(
        self,
        element_description: str,
        click_type: str = "left",
        verify: bool = True,
    ) -> Dict[str, Any]:
        """Locates an element on screen, executes mouse click, and verifies visual update."""
        # 0. Fast-path: If browser DOM driver has an active page and matches
        try:
            from tools.browser_dom import browser_driver
            if browser_driver._page and not browser_driver._page.is_closed():
                dom_res = browser_driver.click_element_by_text(element_description)
                if dom_res.get("success"):
                    return {
                        "success": True,
                        "element": element_description,
                        "method": "browser_dom",
                        "verified": True,
                        "visual_change_detected": True,
                        "details": dom_res,
                    }
        except Exception:
            pass

        # 1. Observe: Capture baseline frame
        baseline = capture_engine.capture_full_screen()

        # 2. Decide: Locate coordinates using Vision Model
        loc = grounding_engine.locate_element(element_description, screenshot_path=baseline["path"])
        if not loc.get("found"):
            return {
                "success": False,
                "element": element_description,
                "error": f"Visual grounding could not locate '{element_description}' on the screen ({loc.get('description')}).",
                "details": loc.get("description"),
            }

        target_x = loc["x"]
        target_y = loc["y"]

        # 3. Act: Move and click
        if click_type == "double":
            controller.double_click(target_x, target_y)
        elif click_type == "right":
            controller.right_click(target_x, target_y)
        else:
            controller.click(target_x, target_y, button="left")

        if not verify:
            return {
                "success": True,
                "element": element_description,
                "clicked_at": {"x": target_x, "y": target_y},
                "verified": False,
            }

        # 4. Verify: Wait briefly for UI animation and capture result frame
        time.sleep(0.3)
        after_frame = capture_engine.capture_full_screen()
        diff = diff_engine.compare_images(baseline["path"], after_frame["path"])

        if not diff["changed"]:
            return {
                "success": False,
                "element": element_description,
                "clicked_at": {"x": target_x, "y": target_y},
                "verified": True,
                "visual_change_detected": False,
                "error": f"Clicked at ({target_x}, {target_y}) for '{element_description}', but no visual update was detected on screen. The interface did not react or the element was unclickable.",
                "ui_update_summary": diff["summary"],
                "confidence": loc.get("confidence", 1.0),
            }

        return {
            "success": True,
            "element": element_description,
            "clicked_at": {"x": target_x, "y": target_y},
            "verified": True,
            "visual_change_detected": True,
            "ui_update_summary": diff["summary"],
            "confidence": loc.get("confidence", 1.0),
        }

    def click_coordinate(
        self,
        x: int,
        y: int,
        button: str = "left",
        verify: bool = True,
    ) -> Dict[str, Any]:
        """Directly clicks at given pixel coordinates (x, y) and verifies visual reaction."""
        baseline = capture_engine.capture_full_screen() if verify else None

        controller.click(x, y, button=button)

        if not verify:
            return {
                "success": True,
                "clicked_at": {"x": x, "y": y},
                "verified": False,
            }

        time.sleep(0.3)
        after_frame = capture_engine.capture_full_screen()
        diff = diff_engine.compare_images(baseline["path"], after_frame["path"])

        if not diff["changed"]:
            return {
                "success": False,
                "clicked_at": {"x": x, "y": y},
                "verified": True,
                "visual_change_detected": False,
                "error": f"Mouse clicked at ({x}, {y}) with '{button}' button, but no visual update was detected on screen. The interface may be unreactive at this spot.",
                "ui_update_summary": diff["summary"],
            }

        return {
            "success": True,
            "clicked_at": {"x": x, "y": y},
            "verified": True,
            "visual_change_detected": True,
            "ui_update_summary": diff["summary"],
        }

    def type_into_element(
        self,
        element_description: str,
        text: str,
        press_enter: bool = False,
    ) -> Dict[str, Any]:
        """Locates a text input or search bar, clicks to focus, and types text."""
        # 1. Click target to focus input field
        click_res = self.click_element(element_description, verify=False)
        if not click_res.get("success"):
            return click_res

        time.sleep(0.15)
        # 2. Type text
        type_summary = controller.type_text(text, press_enter=press_enter)

        # 3. Verify screen change
        time.sleep(0.2)
        after_frame = capture_engine.capture_full_screen()

        return {
            "success": True,
            "element": element_description,
            "text_typed": text,
            "enter_pressed": press_enter,
            "summary": type_summary,
        }


navigator = ClosedLoopNavigator()


@registry.register(description="Visually finds a button, link, tab, or icon on the screen, clicks it, and verifies that the screen updated.")
def click_element(element_description: str, click_type: str = "left") -> Dict[str, Any]:
    return navigator.click_element(element_description, click_type=click_type)


@registry.register(description="Visually finds a search bar or text input on the screen, clicks to focus it, and types text.")
def type_into_element(element_description: str, text: str, press_enter: bool = False) -> Dict[str, Any]:
    return navigator.type_into_element(element_description, text=text, press_enter=press_enter)


@registry.register(description="Clicks directly at exact screen pixel coordinates (x, y) with visual verification. Ideal for clicking specific spots like browser address bars, tabs, or known coordinate targets.")
def click_coordinate(x: int, y: int, button: str = "left", verify: bool = True) -> Dict[str, Any]:
    return navigator.click_coordinate(x, y, button=button, verify=verify)

