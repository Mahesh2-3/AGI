from tools.base import registry
from voice.wake_word import wake_detector
from voice.speaker import speaker


def test_voice_tools_registered():
    schemas = registry.get_schemas()
    names = [s["function"]["name"] for s in schemas]

    assert "speak_message" in names
    assert "listen_to_user" in names


def test_wake_word_parsing():
    # Wake word with immediate command
    triggered, cmd = wake_detector.check_audio_for_wake_word("Jarvis, check the current system volume.")
    assert triggered is True
    assert "check the current system volume" in cmd

    # Wake word alone
    triggered_alone, cmd_alone = wake_detector.check_audio_for_wake_word("Hey Jarvis!")
    assert triggered_alone is True
    assert cmd_alone == ""

    # Non-wake word
    triggered_none, cmd_none = wake_detector.check_audio_for_wake_word("Good morning everyone.")
    assert triggered_none is False
    assert cmd_none == ""


def test_speaker_interruption():
    # Test that stop() safely handles inactive processes without throwing
    speaker.stop()
    assert speaker._current_process is None


if __name__ == "__main__":
    test_voice_tools_registered()
    test_wake_word_parsing()
    test_speaker_interruption()
    print("✅ All voice subsystem tests passed successfully!")
