from core.safety import SafetyEngine, RiskTier


def test_safety_tier_classification():
    safety = SafetyEngine()

    tier, _ = safety.evaluate_risk("get_system_telemetry", {})
    assert tier == RiskTier.SAFE

    tier, _ = safety.evaluate_risk("read_file", {"path": "README.md"})
    assert tier == RiskTier.SAFE

    tier, _ = safety.evaluate_risk("launch_application", {"app_name": "firefox"})
    assert tier == RiskTier.MODERATE

    tier, _ = safety.evaluate_risk("terminate_process", {"pid": 1234})
    assert tier == RiskTier.DANGEROUS

    tier, _ = safety.evaluate_risk("execute_shell_command", {"command": "ls -la"})
    assert tier == RiskTier.DANGEROUS


def test_safety_blocked_critical_command():
    safety = SafetyEngine()

    allowed, reason = safety.authorize("execute_shell_command", {"command": "rm -rf / --no-preserve-root"})
    assert allowed is False
    assert "Blocked critical pattern" in reason


def test_safety_confirmation_handler():
    # Simulate user saying "no"
    denying_safety = SafetyEngine(confirmation_handler=lambda name, args, reason: False)
    allowed, reason = denying_safety.authorize("terminate_process", {"pid": 5678})
    assert allowed is False
    assert "User declined authorization" in reason

    # Simulate user saying "yes"
    approving_safety = SafetyEngine(confirmation_handler=lambda name, args, reason: True)
    allowed, reason = approving_safety.authorize("terminate_process", {"pid": 5678})
    assert allowed is True
    assert reason is None


if __name__ == "__main__":
    test_safety_tier_classification()
    test_safety_blocked_critical_command()
    test_safety_confirmation_handler()
    print("✅ All safety engine tests passed successfully!")
