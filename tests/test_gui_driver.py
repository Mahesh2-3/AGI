from unittest.mock import MagicMock, patch
from tools.base import registry
from gui_driver.mouse_keyboard import controller


def test_gui_tools_registered():
    schemas = registry.get_schemas()
    names = [s["function"]["name"] for s in schemas]

    assert "move_mouse" in names
    assert "click_mouse" in names
    assert "type_text_input" in names
    assert "press_shortcut" in names
    assert "scroll_page" in names


def test_cursor_position_query():
    with patch("gui_driver.mouse_keyboard.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="800, 600\n")
        pos = controller.get_position()
        assert "x" in pos
        assert "y" in pos
        assert isinstance(pos["x"], int)
        assert isinstance(pos["y"], int)


def test_type_and_hotkey_simulation():
    # Mock subprocess.run and pyautogui so tests never emit physical keyboard/mouse events
    with patch("gui_driver.mouse_keyboard.subprocess.run") as mock_run, \
         patch("gui_driver.mouse_keyboard.pyautogui") as mock_pya:
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n")

        res = registry.execute("type_text_input", {"text": "hello", "press_enter": False})
        assert res.success is True
        assert "Typed 5 characters" in res.output

        hotkey_res = registry.execute("press_shortcut", {"hotkey": "ctrl+c"})
        assert hotkey_res.success is True
        assert "ctrl+c" in hotkey_res.output


def test_click_mouse_execution():
    with patch("gui_driver.mouse_keyboard.subprocess.run") as mock_run, \
         patch("gui_driver.mouse_keyboard.pyautogui") as mock_pya:
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n")

        res = registry.execute("click_mouse", {"x": 500, "y": 500, "action": "left"})
        assert res.success is True
        assert isinstance(res.output, dict)
        assert res.output["button"] == "left"
        assert "clicked_at" in res.output


def test_special_characters_typing():
    from gui_driver.mouse_keyboard import char_to_xkb
    assert char_to_xkb(":") == ("shift", "semicolon")
    assert char_to_xkb("/") == ("", "slash")
    assert char_to_xkb("?") == ("shift", "slash")
    assert char_to_xkb(".") == ("", "period")

    with patch("gui_driver.mouse_keyboard.subprocess.run") as mock_run, \
         patch("gui_driver.mouse_keyboard.pyautogui") as mock_pya:
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n")

        res = registry.execute("type_text_input", {"text": "https://chatgpt.com/?q=hello", "press_enter": False})
        assert res.success is True
        assert "Typed 28 characters" in res.output


if __name__ == "__main__":
    test_gui_tools_registered()
    test_cursor_position_query()
    test_type_and_hotkey_simulation()
    test_click_mouse_execution()
    test_special_characters_typing()
    print("✅ All GUI driver tests passed successfully!")
