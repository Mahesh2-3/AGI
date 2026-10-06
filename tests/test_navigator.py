from unittest.mock import patch, MagicMock
from tools.base import registry
from gui_driver.navigator import navigator


def test_navigator_tools_registered():
    schemas = registry.get_schemas()
    names = [s["function"]["name"] for s in schemas]

    assert "click_element" in names
    assert "type_into_element" in names
    assert "click_coordinate" in names


def test_click_element_not_found():
    with patch("gui_driver.navigator.grounding_engine.locate_element") as mock_locate:
        mock_locate.return_value = {"found": False, "description": "Element missing"}
        res = navigator.click_element("Non-existent button")
        assert res["success"] is False
        assert "could not locate" in res["error"]


def test_click_element_closed_loop_success():
    with patch("gui_driver.navigator.grounding_engine.locate_element") as mock_locate, \
         patch("gui_driver.navigator.controller.click") as mock_click, \
         patch("gui_driver.navigator.diff_engine.compare_images") as mock_diff:

        mock_locate.return_value = {"found": True, "x": 250, "y": 300, "confidence": 0.9}
        mock_diff.return_value = {"changed": True, "summary": "UI updated in button region."}

        res = navigator.click_element("Submit button")
        assert res["success"] is True
        assert res["clicked_at"] == {"x": 250, "y": 300}
        assert res["visual_change_detected"] is True
        mock_click.assert_called_once_with(250, 300, button="left")


def test_click_element_no_visual_change_fails():
    with patch("gui_driver.navigator.grounding_engine.locate_element") as mock_locate, \
         patch("gui_driver.navigator.controller.click") as mock_click, \
         patch("gui_driver.navigator.diff_engine.compare_images") as mock_diff:

        mock_locate.return_value = {"found": True, "x": 250, "y": 300, "confidence": 0.9}
        mock_diff.return_value = {"changed": False, "summary": "No UI change detected."}

        res = navigator.click_element("Inactive button")
        assert res["success"] is False
        assert "no visual update was detected" in res["error"]
        assert res["visual_change_detected"] is False


def test_click_coordinate_closed_loop():
    with patch("gui_driver.navigator.controller.click") as mock_click, \
         patch("gui_driver.navigator.diff_engine.compare_images") as mock_diff:

        mock_diff.return_value = {"changed": True, "summary": "Address bar highlighted."}

        res = navigator.click_coordinate(500, 82, button="left")
        assert res["success"] is True
        assert res["clicked_at"] == {"x": 500, "y": 82}
        assert res["visual_change_detected"] is True
        mock_click.assert_called_once_with(500, 82, button="left")


if __name__ == "__main__":
    test_navigator_tools_registered()
    test_click_element_not_found()
    test_click_element_closed_loop_success()
    test_click_element_no_visual_change_fails()
    test_click_coordinate_closed_loop()
    print("✅ All closed-loop navigator tests passed successfully!")
