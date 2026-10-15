from unittest.mock import patch, MagicMock
from tools.base import registry
from gui_driver.navigator import navigator


def test_navigator_tools_registered():
    schemas = registry.get_schemas()
    names = [s["function"]["name"] for s in schemas]

    assert "click_element" in names
    assert "type_into_element" in names
    assert "click_coordinate" in names
    assert "move_to_element" in names


def test_click_element_not_found():
    with patch("gui_driver.navigator.grounding_engine.locate_element") as mock_locate:
        mock_locate.return_value = {"found": False, "description": "Element missing"}
        res = navigator.click_element("Non-existent button")
        assert res["success"] is False
        assert "could not locate" in res["error"]


def test_click_element_closed_loop_success():
    with patch("gui_driver.navigator.grounding_engine.locate_element") as mock_locate, \
         patch("gui_driver.navigator.controller.move_to") as mock_move, \
         patch("gui_driver.navigator.controller.click") as mock_click, \
         patch("gui_driver.navigator.diff_engine.compare_images") as mock_diff:

        mock_locate.return_value = {
            "found": True,
            "x": 250,
            "y": 300,
            "bbox": [0.13, 0.28, 0.15, 0.32],
            "bbox_pixels": [240, 290, 260, 310],
            "confidence": 0.9,
        }
        mock_diff.return_value = {"changed": True, "summary": "UI updated in button region."}

        res = navigator.click_element("Submit button")
        assert res["success"] is True
        assert res["clicked_at"] == {"x": 250, "y": 300}
        assert res["bbox"] == [0.13, 0.28, 0.15, 0.32]
        assert res["visual_change_detected"] is True
        mock_move.assert_called_once_with(250, 300)
        mock_click.assert_called_once_with(250, 300, button="left")


def test_move_to_element():
    with patch("gui_driver.navigator.grounding_engine.locate_element") as mock_locate, \
         patch("gui_driver.navigator.controller.move_to") as mock_move:

        mock_locate.return_value = {
            "found": True,
            "x": 400,
            "y": 500,
            "bbox": [0.2, 0.4, 0.25, 0.45],
            "confidence": 0.95,
        }

        res = navigator.move_to_element("Settings tab")
        assert res["success"] is True
        assert res["moved_to"] == {"x": 400, "y": 500}
        mock_move.assert_called_once_with(400, 500)


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


def test_click_element_none_coordinates_suppressed():
    with patch("gui_driver.navigator.grounding_engine.locate_element") as mock_locate, \
         patch("gui_driver.navigator.controller.click") as mock_click:
        mock_locate.return_value = {"found": True, "x": None, "y": None}
        res = navigator.click_element("Ghost button")
        assert res["success"] is False
        assert "no actionable coordinates" in res["error"]
        assert mock_click.call_count == 0


def test_click_coordinate_out_of_bounds_suppressed():
    with patch("gui_driver.navigator.controller.click") as mock_click:
        res = navigator.click_coordinate(5000, 9999)
        assert res["success"] is False
        assert "exceed screen boundaries" in res["error"]
        assert mock_click.call_count == 0


if __name__ == "__main__":
    test_navigator_tools_registered()
    test_click_element_not_found()
    test_click_element_closed_loop_success()
    test_click_element_no_visual_change_fails()
    test_click_coordinate_closed_loop()
    print("✅ All closed-loop navigator tests passed successfully!")
