from tools.base import registry
from tools.browser_dom import browser_driver


def test_browser_dom_tools_registered():
    schemas = registry.get_schemas()
    names = [s["function"]["name"] for s in schemas]

    assert "browser_navigate" in names
    assert "get_webpage_interactive_elements" in names
    assert "click_webpage_element" in names
    assert "type_webpage_input" in names
    assert "open_browser_in_workspace" in names
    assert "scan_webpage_dom" in names
    assert "execute_browser_chain" in names


def test_browser_dom_navigation_and_extraction():
    try:
        # Test navigation to local/public fast page
        nav = browser_driver.navigate("https://example.com")
        assert "Example Domain" in nav["title"]

        # Extract DOM elements
        elements = browser_driver.get_interactive_elements()
        assert len(elements) > 0

        # Verify element structure
        first = elements[0]
        assert "tag" in first
        assert "text" in first
        assert "x" in first
        assert "y" in first

        # Test scan_summary
        summary = browser_driver.scan_summary()
        assert summary["success"] is True
        assert summary["element_count"] > 0
        assert "Example Domain" in summary["page_title"]

        # Test click by text using the actual extracted element's text
        target_text = first["text"]
        click_res = browser_driver.click_element_by_text(target_text)
        assert click_res["success"] is True

        # Test click failure reporting for nonexistent element
        fail_res = browser_driver.click_element_by_text("nonexistent_element_999xyz")
        assert fail_res["success"] is False
        assert "Could not find interactive element" in fail_res["error"]

        # Test execute_chain
        chain_res = browser_driver.execute_chain([
            {"action": "scan"},
            {"action": "wait", "seconds": 0.05},
        ])
        assert chain_res["success"] is True
        assert chain_res["steps_executed"] == 2

    finally:
        browser_driver.close()


if __name__ == "__main__":
    test_browser_dom_tools_registered()
    test_browser_dom_navigation_and_extraction()
    print("✅ All browser DOM driver tests passed successfully!")
