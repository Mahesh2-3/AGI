import shutil
from typing import Any, Dict, List, Optional
from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from tools.base import registry

CHROME_BIN = shutil.which("google-chrome-stable") or shutil.which("google-chrome") or shutil.which("chromium") or "/usr/bin/google-chrome-stable"


class BrowserDOMDriver:
    """High-speed DOM-aware browser automation driver using Chrome and Playwright.
    Bypasses slow cloud VLM screenshot calls with sub-15ms local DOM inspection."""

    def __init__(self):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    def _ensure_browser(self, headless: bool = False) -> Page:
        if self._page and not self._page.is_closed():
            return self._page

        if not self._playwright:
            self._playwright = sync_playwright().start()

        # Launch headed or headless Chromium using installed system Chrome
        self._browser = self._playwright.chromium.launch(
            headless=headless,
            executable_path=CHROME_BIN if shutil.which(CHROME_BIN) else None,
            args=["--no-default-browser-check", "--no-first-run"],
        )
        self._context = self._browser.new_context(viewport={"width": 1920, "height": 1080})
        self._page = self._context.new_page()
        return self._page

    def navigate(self, url: str) -> Dict[str, Any]:
        """Navigates to URL and waits for DOM ready state."""
        page = self._ensure_browser(headless=False)
        clean_url = url.strip()
        if not clean_url.startswith(("http://", "https://")):
            clean_url = f"https://{clean_url}"

        page.goto(clean_url, wait_until="domcontentloaded", timeout=20000)
        page.wait_for_timeout(1000)

        return {
            "url": page.url,
            "title": page.title(),
            "status": "ready",
        }

    def get_interactive_elements(self, max_items: int = 40) -> List[Dict[str, Any]]:
        """Extracts visible buttons, links, inputs, and controls with their labels and coordinates."""
        if not self._page or self._page.is_closed():
            return []

        js_script = """() => {
            const elements = document.querySelectorAll('button, a, input, select, textarea, [role="button"], [role="link"]');
            return Array.from(elements).map((el, idx) => {
                const r = el.getBoundingClientRect();
                const text = (el.innerText || el.value || el.getAttribute('aria-label') || el.placeholder || '').trim();
                return {
                    id: el.id || `elem_${idx}`,
                    tag: el.tagName.toLowerCase(),
                    text: text.slice(0, 80),
                    role: el.getAttribute('role') || el.tagName.toLowerCase(),
                    type: el.getAttribute('type') || null,
                    visible: r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < window.innerHeight,
                    x: Math.round(r.left + r.width / 2),
                    y: Math.round(r.top + r.height / 2),
                    width: Math.round(r.width),
                    height: Math.round(r.height)
                };
            }).filter(e => e.visible && e.text.length > 0);
        }"""

        items = self._page.evaluate(js_script)
        return items[:max_items]

    def scan_summary(self, max_items: int = 35) -> Dict[str, Any]:
        """Scans DOM and returns page info plus a clear human-readable overview of interactive elements."""
        if not self._page or self._page.is_closed():
            return {
                "success": False,
                "error": "No active browser page open. Call browser_navigate or open_browser_in_workspace first.",
            }

        elements = self.get_interactive_elements(max_items=max_items)
        lines = []
        for el in elements:
            role = el.get("role") or el.get("tag", "element")
            text = el.get("text", "").strip()
            if text:
                lines.append(f"- [{role.capitalize()}] \"{text}\"")

        summary_text = "\n".join(lines) if lines else "No interactive elements detected."
        return {
            "success": True,
            "page_title": self._page.title(),
            "page_url": self._page.url,
            "element_count": len(elements),
            "interactive_summary": summary_text,
            "elements": elements,
        }

    def open_in_workspace(self, url: str, workspace_id: Optional[int | str] = None) -> Dict[str, Any]:
        """Switches to designated workspace and navigates directly to URL in browser."""
        import time
        from core.workspace import workspace_manager

        current_ws = workspace_manager.get_active_workspace().get("id", 1)
        if workspace_id is None:
            target_ws = 2 if current_ws == 1 else (current_ws + 1)
        else:
            target_ws = int(workspace_id)

        workspace_manager.switch_to_workspace(target_ws)
        workspace_manager.set_working_workspace(target_ws)
        time.sleep(0.2)

        nav_result = self.navigate(url)
        nav_result["workspace"] = target_ws
        return nav_result

    def click_element_by_text(self, text: str) -> Dict[str, Any]:
        """Locates an element by visible text, label, or selector and clicks it directly via DOM."""
        if not self._page or self._page.is_closed():
            raise RuntimeError("No active browser page open. Call navigate(url) first.")

        clean = text.strip()

        # If it looks like a CSS selector (.btn, #id, etc.)
        if clean.startswith((".", "#", "[", "xpath=")):
            try:
                loc = self._page.locator(clean)
                if loc.count() > 0 and loc.first.is_visible():
                    box = loc.first.bounding_box()
                    loc.first.click(timeout=3000, no_wait_after=True)
                    return {
                        "success": True,
                        "method": "CSS selector locator",
                        "target": clean,
                        "bounding_box": box,
                    }
            except Exception:
                pass

        # Try semantic locators
        locators = [
            self._page.get_by_role("button", name=clean, exact=False),
            self._page.get_by_role("link", name=clean, exact=False),
            self._page.locator(f"button:has-text('{clean}')"),
            self._page.locator(f"a:has-text('{clean}')"),
            self._page.locator(f"[role='button']:has-text('{clean}')"),
            self._page.locator(f"[role='link']:has-text('{clean}')"),
            self._page.get_by_text(clean, exact=False),
        ]

        for loc in locators:
            try:
                if loc.count() > 0 and loc.first.is_visible():
                    box = loc.first.bounding_box()
                    loc.first.scroll_into_view_if_needed(timeout=1000)
                    loc.first.click(timeout=3000, no_wait_after=True)
                    return {
                        "success": True,
                        "method": "DOM semantic locator",
                        "target": clean,
                        "bounding_box": box,
                    }
            except Exception:
                continue

        # Fallback to evaluating element coordinate from interactive elements
        elements = self.get_interactive_elements(max_items=80)
        clean_lower = clean.lower()
        for el in elements:
            if clean_lower in el["text"].lower():
                self._page.mouse.click(el["x"], el["y"])
                return {
                    "success": True,
                    "method": "DOM coordinates click",
                    "target": clean,
                    "matched_element": el,
                }

        return {
            "success": False,
            "error": f"Could not find interactive element matching text '{clean}'.",
        }

    def type_into_input(self, selector_or_label: str, text: str, press_enter: bool = False) -> Dict[str, Any]:
        """Finds input field by placeholder, label, or selector and types text."""
        if not self._page or self._page.is_closed():
            raise RuntimeError("No active browser page open. Call navigate(url) first.")

        locators = [
            self._page.get_by_placeholder(selector_or_label, exact=False),
            self._page.get_by_label(selector_or_label, exact=False),
            self._page.locator(f"input[name='{selector_or_label}']"),
            self._page.locator(selector_or_label),
        ]

        for loc in locators:
            try:
                if loc.count() > 0 and loc.first.is_visible():
                    loc.first.fill(text, timeout=3000)
                    if press_enter:
                        loc.first.press("Enter")
                    return {
                        "success": True,
                        "field": selector_or_label,
                        "typed_text": text,
                    }
            except Exception:
                continue

        return {
            "success": False,
            "error": f"Could not locate input field matching '{selector_or_label}'.",
        }

    def execute_chain(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Executes a compound chain of web actions in rapid succession."""
        import time
        history = []
        for i, step in enumerate(steps):
            action = step.get("action", "").lower()
            if action in ("open", "navigate", "goto"):
                url = step.get("url", "")
                ws = step.get("workspace")
                if ws:
                    res = self.open_in_workspace(url, ws)
                else:
                    res = self.navigate(url)
                history.append({"step": i + 1, "action": action, "result": res})
            elif action in ("scan", "inspect"):
                res = self.scan_summary()
                history.append({"step": i + 1, "action": action, "result": res.get("interactive_summary")})
            elif action in ("click", "click_element"):
                target = step.get("target") or step.get("text", "")
                res = self.click_element_by_text(target)
                history.append({"step": i + 1, "action": action, "target": target, "result": res})
            elif action in ("type", "fill"):
                field = step.get("field") or step.get("target", "")
                text = step.get("text", "")
                press_enter = step.get("press_enter", True)
                res = self.type_into_input(field, text, press_enter=press_enter)
                history.append({"step": i + 1, "action": action, "field": field, "result": res})
            elif action == "wait":
                secs = float(step.get("seconds", 1.0))
                time.sleep(secs)
                history.append({"step": i + 1, "action": "wait", "seconds": secs})
            else:
                history.append({"step": i + 1, "error": f"Unknown action '{action}'"})

        return {
            "success": True,
            "steps_executed": len(history),
            "history": history,
            "current_title": self._page.title() if self._page else None,
            "current_url": self._page.url if self._page else None,
        }

    def close(self):
        """Closes browser session."""
        try:
            if self._page and not self._page.is_closed():
                self._page.close()
            if self._browser and self._browser.is_connected():
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._page = None
        self._browser = None
        self._playwright = None


browser_driver = BrowserDOMDriver()


@registry.register(description="Switches to a designated or new desktop workspace, launches Google Chrome, and navigates directly to the target URL.")
def open_browser_in_workspace(url: str, workspace_id: Optional[int] = None) -> Dict[str, Any]:
    return browser_driver.open_in_workspace(url, workspace_id=workspace_id)


@registry.register(description="Navigates to a website and returns page title and load status using the high-speed DOM driver.")
def browser_navigate(url: str) -> Dict[str, Any]:
    return browser_driver.navigate(url)


@registry.register(description="Scans the current web page DOM in milliseconds and returns an instant summary of all clickable buttons, links, and text inputs without consuming vision tokens.")
def scan_webpage_dom(max_elements: int = 35) -> Dict[str, Any]:
    return browser_driver.scan_summary(max_items=max_elements)


@registry.register(description="Extracts all visible clickable buttons, links, and inputs on the current web page directly from the DOM in milliseconds.")
def get_webpage_interactive_elements() -> List[Dict[str, Any]]:
    return browser_driver.get_interactive_elements()


@registry.register(description="Clicks a button or link on the active web page by matching its text, label, or selector directly in the DOM (sub-15ms, 0 vision tokens).")
def click_webpage_element(element_text: str) -> Dict[str, Any]:
    return browser_driver.click_element_by_text(element_text)


@registry.register(description="Types text into a form or search input field on the current web page using semantic DOM locators.")
def type_webpage_input(field_name_or_placeholder: str, text_to_type: str, press_enter: bool = False) -> Dict[str, Any]:
    return browser_driver.type_into_input(field_name_or_placeholder, text_to_type, press_enter=press_enter)


@registry.register(description="Executes a compound chain of web actions (e.g. open, scan, click, type) in a single fast call without multiple reasoning turns.")
def execute_browser_chain(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    return browser_driver.execute_chain(steps)

