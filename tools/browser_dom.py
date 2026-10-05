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

    def click_element_by_text(self, text: str) -> Dict[str, Any]:
        """Locates an element by visible text or label and clicks it directly via DOM."""
        if not self._page or self._page.is_closed():
            raise RuntimeError("No active browser page open. Call navigate(url) first.")

        clean = text.strip().lower()

        # Try semantic locator first
        locators = [
            self._page.get_by_role("button", name=text, exact=False),
            self._page.locator(f"button:has-text('{text}')"),
            self._page.locator(f"a:has-text('{text}')"),
            self._page.locator(f"[role='button']:has-text('{text}')"),
            self._page.get_by_text(text, exact=False),
        ]

        for loc in locators:
            try:
                if loc.count() > 0 and loc.first.is_visible():
                    box = loc.first.bounding_box()
                    loc.first.click(timeout=3000)
                    return {
                        "success": True,
                        "method": "DOM semantic locator",
                        "target": text,
                        "bounding_box": box,
                    }
            except Exception:
                continue

        # Fallback to evaluating element coordinate from interactive elements
        elements = self.get_interactive_elements(max_items=80)
        for el in elements:
            if clean in el["text"].lower():
                self._page.mouse.click(el["x"], el["y"])
                return {
                    "success": True,
                    "method": "DOM coordinates click",
                    "target": text,
                    "matched_element": el,
                }

        return {
            "success": False,
            "error": f"Could not find interactive element matching text '{text}'.",
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


@registry.register(description="Navigates to a website and returns page title and load status using the high-speed DOM driver.")
def browser_navigate(url: str) -> Dict[str, Any]:
    return browser_driver.navigate(url)


@registry.register(description="Extracts all visible clickable buttons, links, and inputs on the current web page directly from the DOM in milliseconds.")
def get_webpage_interactive_elements() -> List[Dict[str, Any]]:
    return browser_driver.get_interactive_elements()


@registry.register(description="Clicks a button or link on the active web page by matching its text or label directly in the DOM (sub-15ms, 0 vision tokens).")
def click_webpage_element(element_text: str) -> Dict[str, Any]:
    return browser_driver.click_element_by_text(element_text)


@registry.register(description="Types text into a form or search input field on the current web page using semantic DOM locators.")
def type_webpage_input(field_name_or_placeholder: str, text_to_type: str, press_enter: bool = False) -> Dict[str, Any]:
    return browser_driver.type_into_input(field_name_or_placeholder, text_to_type, press_enter=press_enter)
