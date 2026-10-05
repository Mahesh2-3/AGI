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
    pos = controller.get_position()
    assert "x" in pos
    assert "y" in pos
    assert isinstance(pos["x"], int)
    assert isinstance(pos["y"], int)


def test_type_and_hotkey_simulation():
    res = registry.execute("type_text_input", {"text": "hello", "press_enter": False})
    assert res.success is True
    assert "Typed 5 characters" in res.output

    hotkey_res = registry.execute("press_shortcut", {"hotkey": "ctrl+c"})
    assert hotkey_res.success is True
    assert "ctrl+c" in hotkey_res.output


if __name__ == "__main__":
    test_gui_tools_registered()
    test_cursor_position_query()
    test_type_and_hotkey_simulation()
    print("✅ All GUI driver tests passed successfully!")
