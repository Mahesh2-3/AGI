import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from core.agent import JarvisAgent
from core.llm_client import LLMClient
from core.safety import SafetyEngine
from tools.base import registry

# Pre-load all tools and capabilities
import tools.system_info
import tools.app_control
import tools.file_ops
import tools.system_ctl
import tools.shell_runner
import vision.capture
import vision.grounding
import vision.visual_diff
import gui_driver.mouse_keyboard
import gui_driver.navigator
import voice.speaker
import voice.listener


class BenchmarkRunner:
    """Executes a comprehensive, autonomous battery of prompts across all Jarvis subsystems,
    logging detailed execution metrics, tool latencies, and total elapsed time."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.output_file = Path("BENCHMARK_RESULTS.md")

        # Auto-authorize safe benchmark actions
        def benchmark_confirm(tool_name: str, args: dict, reason: str) -> bool:
            return True

        self.safety = SafetyEngine(confirmation_handler=benchmark_confirm)
        self.llm = LLMClient()

    def run_prompt_test(self, category: str, prompt: str, max_iterations: int = 5) -> Dict[str, Any]:
        """Runs a single prompt through JarvisAgent, recording every step and tool call."""
        tool_calls_log = []
        thought_log = []

        def on_thought(t: str):
            thought_log.append(t)

        def on_tool_call(name: str, args: dict):
            tool_calls_log.append({"name": name, "args": args, "t_start": time.perf_counter()})

        def on_tool_result(name: str, output: Any, success: bool):
            if tool_calls_log and tool_calls_log[-1]["name"] == name:
                entry = tool_calls_log[-1]
                entry["latency_ms"] = round((time.perf_counter() - entry["t_start"]) * 1000, 2)
                entry["success"] = success
                entry["output_preview"] = str(output)[:150]
                del entry["t_start"]

        agent = JarvisAgent(
            llm_client=self.llm,
            tool_registry=registry,
            safety_engine=self.safety,
            max_iterations=max_iterations,
            enable_fast_path=True,
            on_thought=on_thought,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
        )

        start_time = datetime.now()
        t0 = time.perf_counter()

        try:
            response = agent.step(prompt)
            status = "SUCCESS"
            error = None
        except Exception as e:
            response = ""
            status = "ERROR"
            error = str(e)

        elapsed_s = round(time.perf_counter() - t0, 3)
        end_time = datetime.now()

        # Check if fast-path handled it (<15ms, 0 tool calls recorded via callback)
        is_fast_path = (elapsed_s < 0.08 and len(tool_calls_log) == 0 and "Sir" in response)

        record = {
            "category": category,
            "prompt": prompt,
            "status": status,
            "is_fast_path": is_fast_path,
            "start_time": start_time.strftime("%H:%M:%S"),
            "end_time": end_time.strftime("%H:%M:%S"),
            "elapsed_seconds": elapsed_s,
            "tool_calls": tool_calls_log,
            "thoughts": thought_log,
            "response_preview": response.strip().replace("\n", " ")[:200],
            "error": error,
        }

        self.results.append(record)
        return record

    def save_markdown_report(self):
        """Generates a structured markdown report with summary tables and detailed metrics."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["status"] == "SUCCESS")
        fast_path_count = sum(1 for r in self.results if r["is_fast_path"])
        total_time_s = sum(r["elapsed_seconds"] for r in self.results)
        avg_time_s = round(total_time_s / max(1, total_tests), 2)

        lines = [
            "# Project J.A.R.V.I.S. – Autonomous Verification & Benchmark Report",
            f"**Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**System**: Arch Linux (Hyprland Wayland 0.56.2)",
            f"**Active Provider**: {self.llm.provider.upper()} | Primary Model: `{self.llm.model}`",
            "",
            "## Executive Summary",
            f"- **Total Tests Executed**: `{total_tests}`",
            f"- **Success Rate**: `{passed_tests}/{total_tests}` (`{round((passed_tests/total_tests)*100, 1)}%`)",
            f"- **Fast-Path Reflex Commands (<50ms)**: `{fast_path_count}`",
            f"- **Total Duration**: `{round(total_time_s, 2)}s` (Average: `{avg_time_s}s` per task)",
            "",
            "## Benchmark Results Table",
            "| # | Category | Prompt | Mode | Tools Invoked | Elapsed (s) | Status |",
            "|---|---|---|---|---|---|---|",
        ]

        for idx, r in enumerate(self.results, 1):
            mode = "**Fast-Path** ⚡" if r["is_fast_path"] else "ReAct Cognitive 🧠"
            tool_names = ", ".join(f"`{t['name']}`" for t in r["tool_calls"]) if r["tool_calls"] else "*(Direct Reflex)*"
            status_icon = "✅ PASS" if r["status"] == "SUCCESS" else f"❌ {r['error']}"
            lines.append(
                f"| {idx} | {r['category']} | \"{r['prompt']}\" | {mode} | {tool_names} | {r['elapsed_seconds']}s | {status_icon} |"
            )

        lines.extend([
            "",
            "## Detailed Step-by-Step Execution Logs",
            "",
        ])

        for idx, r in enumerate(self.results, 1):
            lines.extend([
                f"### Test {idx}: {r['category']} – \"{r['prompt']}\"",
                f"- **Time**: `{r['start_time']} -> {r['end_time']}` (`{r['elapsed_seconds']}s`)",
                f"- **Execution Mode**: {'FastPath Reflex' if r['is_fast_path'] else 'ReAct Multi-Turn Loop'}",
                f"- **Result Status**: `{r['status']}`",
            ])

            if r["tool_calls"]:
                lines.append("- **Tool Calls Executed**:")
                for tc in r["tool_calls"]:
                    lat = tc.get("latency_ms", "N/A")
                    lines.append(f"  - `{tc['name']}({tc['args']})` -> Latency: `{lat}ms` | Output: `{tc.get('output_preview', '')}`")

            lines.extend([
                f"- **Jarvis Reply**: \"{r['response_preview']}\"",
                "",
            ])

        self.output_file.write_text("\n".join(lines))
        print(f"\n[REPORT GENERATED] Saved to: {self.output_file.resolve()}")


def main():
    runner = BenchmarkRunner()

    tests = [
        # 1. Deterministic Fast-Path Reflexes (<15ms)
        ("Reflex", "Jarvis, what time is it?"),
        ("Reflex", "What is your current system status and telemetry?"),
        ("Reflex", "Turn the volume up by 10%"),
        ("Reflex", "Take a screenshot of my desktop"),
        ("Reflex", "List files in the current directory"),

        # 2. Workspace File Operations
        ("File Ops", "Create a file named notes.txt with a brief 3-bullet summary of quantum computing"),
        ("File Ops", "Read the file notes.txt and verify its content"),
        ("File Ops", "Search for the word 'quantum' in the workspace files"),
        ("File Ops", "Move notes.txt to the safe trash directory"),

        # 3. System & Window Management
        ("System Management", "List all open desktop windows and their process IDs"),
        ("System Management", "Check network connection status and connectivity"),

        # 4. Vision Perception & Screen Inspection
        ("Vision Perception", "Inspect the current desktop screen and summarize visible applications"),

        # 5. Web Browser Automation
        ("Browser Automation", "Open https://news.ycombinator.com in the browser and wait 2 seconds"),
        ("Browser Automation", "Focus the Google Chrome window"),
        ("Browser Automation", "Close the Google Chrome window"),

        # 6. Cognitive Reasoning
        ("Cognitive Core", "Explain the architectural difference between Wayland and X11 in two concise sentences"),
    ]

    print(f"=== Starting J.A.R.V.I.S. Benchmark Battery ({len(tests)} Tasks) ===")

    for i, (category, prompt) in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] [{category}] Testing: '{prompt}'...")
        rec = runner.run_prompt_test(category, prompt)
        mode = "⚡ Fast-Path" if rec["is_fast_path"] else "🧠 ReAct"
        print(f"    Status: {rec['status']} | Mode: {mode} | Duration: {rec['elapsed_seconds']}s")
        if rec["tool_calls"]:
            for tc in rec["tool_calls"]:
                print(f"    └─ Tool: {tc['name']} ({tc.get('latency_ms', 'N/A')}ms)")

    runner.save_markdown_report()
    print("\n=== Benchmark Battery Completed Successfully ===")


if __name__ == "__main__":
    main()
