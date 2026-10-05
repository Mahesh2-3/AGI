import subprocess
from typing import Dict, Any
from tools.base import registry


@registry.register(
    description="Executes a shell command in the system terminal with timeout protection. (Requires confirmation for safety).",
    requires_confirmation=True,
)
def execute_shell_command(command: str, timeout_seconds: int = 20) -> Dict[str, Any]:
    try:
        res = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "exit_code": res.returncode,
            "stdout": res.stdout.strip()[:2000],
            "stderr": res.stderr.strip()[:1000],
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout_seconds} seconds.",
        }
    except Exception as e:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Execution error: {e}",
        }
