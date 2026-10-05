from tools.base import registry
import tools.app_control
from tools.app_control import get_active_workspace, switch_to_workspace, move_window_to_workspace


def test_app_control_tools_registered():
    schemas = registry.get_schemas()
    names = [s["function"]["name"] for s in schemas]

    assert "list_open_windows" in names
    assert "launch_application" in names
    assert "focus_window" in names
    assert "close_window" in names
    assert "get_current_workspace" in names
    assert "switch_to_workspace" in names
    assert "move_window_to_workspace" in names
    assert "list_running_processes" in names
    assert "open_path" in names
    assert "open_browser_url" in names
    assert "terminate_process" in names


def test_alias_resolution():
    from tools.app_control import resolve_application
    assert resolve_application("fileExplorer") == "dolphin"
    assert resolve_application("browser") in ("google-chrome", "firefox", "chromium", "brave", "zen-browser", "browser")


def test_workspace_helpers():
    active_ws = get_active_workspace()
    assert isinstance(active_ws, dict)
    assert "id" in active_ws
    assert "name" in active_ws


def test_list_open_windows():
    res = registry.execute("list_open_windows", {})
    assert res.success is True
    assert isinstance(res.output, list)


def test_list_running_processes():
    res = registry.execute("list_running_processes", {"limit": 5})
    assert res.success is True
    assert isinstance(res.output, list)
    assert len(res.output) <= 5


if __name__ == "__main__":
    test_app_control_tools_registered()
    test_alias_resolution()
    test_workspace_helpers()
    test_list_open_windows()
    test_list_running_processes()
    print("✅ All app control tests passed successfully!")
