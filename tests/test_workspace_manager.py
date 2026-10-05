from core.workspace import workspace_manager, WorkspaceManager
from tools.base import registry
import tools.app_control


def test_workspace_manager_basic():
    mgr = WorkspaceManager()
    active = mgr.get_active_workspace()
    assert "id" in active
    assert "name" in active
    assert isinstance(active["id"], int)

    # Initial working workspace defaults to active
    working = mgr.get_working_workspace()
    assert working == active["id"]

    # Explicit set
    mgr.set_working_workspace(5)
    assert mgr.get_working_workspace() == 5

    status = mgr.get_status()
    assert status["working_workspace"] == 5
    assert status["active_workspace"] == active["id"]
    assert status["is_working_workspace_active"] == (active["id"] == 5)


def test_workspace_manager_ensure_active():
    active = workspace_manager.get_active_workspace()["id"]

    # Setting working workspace to current active workspace
    workspace_manager.set_working_workspace(active)
    res = workspace_manager.ensure_working_workspace_active()
    assert res["was_active"] is True
    assert res["target_workspace"] == active
    assert "already active" in res["message"]


def test_workspace_tools_registered():
    schemas = registry.get_schemas()
    names = [s["function"]["name"] for s in schemas]

    assert "make_working_workspace_active" in names
    assert "get_jarvis_workspace_status" in names
    assert "set_jarvis_working_workspace" in names
    assert "get_current_workspace" in names
    assert "switch_to_workspace" in names
    assert "move_window_to_workspace" in names


def test_workspace_tools_execution():
    status_res = registry.execute("get_jarvis_workspace_status", {})
    assert status_res.success is True
    assert "active_workspace" in status_res.output
    assert "working_workspace" in status_res.output

    ensure_res = registry.execute("make_working_workspace_active", {})
    assert ensure_res.success is True
    assert "was_active" in ensure_res.output
    assert "target_workspace" in ensure_res.output


if __name__ == "__main__":
    test_workspace_manager_basic()
    test_workspace_manager_ensure_active()
    test_workspace_tools_registered()
    test_workspace_tools_execution()
    print("✅ All workspace manager tests passed successfully!")
