from tools.base import registry
import tools.system_ctl


def test_system_ctl_tools_registered():
    schemas = registry.get_schemas()
    names = [s["function"]["name"] for s in schemas]

    assert "send_desktop_notification" in names
    assert "get_audio_volume" in names
    assert "set_audio_volume" in names
    assert "toggle_audio_mute" in names
    assert "get_network_status" in names


def test_get_audio_volume():
    res = registry.execute("get_audio_volume", {})
    assert res.success is True
    assert "volume_percent" in res.output


def test_get_network_status():
    res = registry.execute("get_network_status", {})
    assert res.success is True
    assert "connected" in res.output


if __name__ == "__main__":
    test_system_ctl_tools_registered()
    test_get_audio_volume()
    test_get_network_status()
    print("✅ All system control tests passed successfully!")
