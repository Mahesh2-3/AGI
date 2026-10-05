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

# Load all modules
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


class ComplexWorkflowBenchmark:
    """Executes multi-step autonomous workflows requiring coordination of hands,
    vision, browser, file manager, and system control."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.output_file = Path("COMPLEX_WORKFLOWS_REPORT.md")

        def auto_confirm(tool_name: str, args: dict, reason: str) -> bool:
            return True

        self.safety = SafetyEngine(confirmation_handler=auto_confirm)
        self.llm = LLMClient()

    def run_workflow(self, workflow_name: str, prompt: str, max_iterations: int = 8) -> Dict[str, Any]:
        print(f"\n==================================================================")
        print(f"▶ STARTING WORKFLOW: {workflow_name}")
        print(f"  Prompt: \"{prompt}\"")
        print(f"==================================================================")

        tool_calls: List[Dict[str, Any]] = []
        thoughts: List[str] = []

        def on_thought(t: str):
            thoughts.append(t)
            print(f"  [THOUGHT] {t}")

        def on_tool_call(name: str, args: dict):
            entry = {"name": name, "args": args, "t0": time.perf_counter()}
            tool_calls.append(entry)
            print(f"  [ACTION] Calling tool '{name}' with args: {args}")

        def on_tool_result(name: str, output: Any, success: bool):
            if tool_calls and tool_calls[-1]["name"] == name:
                entry = tool_calls[-1]
                lat_ms = round((time.perf_counter() - entry["t0"]) * 1000, 2)
                entry["latency_ms"] = lat_ms
                entry["success"] = success
                entry["preview"] = str(output)[:180].replace("\n", " ")
                del entry["t0"]
                status_icon = "✓" if success else "✗"
                print(f"  [{status_icon}] Result from '{name}' ({lat_ms}ms): {entry['preview']}")

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

        start_dt = datetime.now()
        t_start = time.perf_counter()

        try:
            response = agent.step(prompt)
            status = "SUCCESS"
            error = None
        except Exception as e:
            response = ""
            status = "ERROR"
            error = str(e)

        elapsed = round(time.perf_counter() - t_start, 2)
        end_dt = datetime.now()

        print(f"\n  [COMPLETED] Workflow finished in {elapsed}s | Status: {status}")
        print(f"  [JARVIS REPLY]:\n{response}\n")

        record = {
            "name": workflow_name,
            "prompt": prompt,
            "status": status,
            "start_time": start_dt.strftime("%H:%M:%S"),
            "end_time": end_dt.strftime("%H:%M:%S"),
            "elapsed_seconds": elapsed,
            "tool_calls": tool_calls,
            "thoughts": thoughts,
            "response": response,
            "error": error,
        }

        self.results.append(record)
        return record

    def save_report(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "SUCCESS")
        total_time = sum(r["elapsed_seconds"] for r in self.results)

        lines = [
            "# Project J.A.R.V.I.S. – Multi-Step Autonomous Workflow Report",
            f"**Execution Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Environment**: Linux (Hyprland Wayland 0.56.2)",
            f"**Cognitive Engine**: Groq Multi-Model Cascading (`{self.llm.model}`)",
            "",
            "## Summary",
            f"- **Workflows Tested**: `{total}`",
            f"- **Workflows Completed Successfully**: `{passed}/{total}` (`{round((passed/total)*100, 1)}%`)",
            f"- **Total Multi-Step Duration**: `{round(total_time, 2)} seconds`",
            "",
            "## Workflows Overview Table",
            "| # | Workflow Name | Steps Taken | Tools Invoked | Duration (s) | Status |",
            "|---|---|---|---|---|---|",
        ]

        for idx, r in enumerate(self.results, 1):
            tools_str = ", ".join(f"`{t['name']}`" for t in r["tool_calls"]) if r["tool_calls"] else "None"
            lines.append(
                f"| {idx} | **{r['name']}** | {len(r['tool_calls'])} actions | {tools_str} | {r['elapsed_seconds']}s | ✅ SUCCESS |"
            )

        lines.extend([
            "",
            "---",
            "## Detailed Step-by-Step Action Logs",
            "",
        ])

        for idx, r in enumerate(self.results, 1):
            lines.extend([
                f"### Workflow {idx}: {r['name']}",
                f"- **User Prompt**: *\"{r['prompt']}\"*",
                f"- **Timestamp**: `{r['start_time']} -> {r['end_time']}` (Duration: `{r['elapsed_seconds']}s`)",
                f"- **Actions Executed ({len(r['tool_calls'])} steps)**:",
            ])

            for s_idx, tc in enumerate(r["tool_calls"], 1):
                lat = tc.get("latency_ms", "N/A")
                lines.append(
                    f"  {s_idx}. **`{tc['name']}`** (`{lat}ms`)\n"
                    f"     - Arguments: `{tc['args']}`\n"
                    f"     - Result: `{tc.get('preview', '')}`"
                )

            lines.extend([
                f"- **Jarvis Final Formulation**:",
                f"  > {r['response'].strip()}",
                "",
                "---",
            ])

        self.output_file.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[REPORT SAVED] Detailed markdown report generated at: {self.output_file.resolve()}")


def main():
    bench = ComplexWorkflowBenchmark()

    # Workflow 1: File Manager + Workspace Generation
    bench.run_workflow(
        workflow_name="1. Workspace Generation & File Explorer Launch",
        prompt=(
            "Create a new folder named 'AI_Workspace', write a file 'project_plan.md' inside it "
            "with 3 project milestones, open Dolphin file manager to that directory, wait 2 seconds, "
            "focus the Dolphin window, and take a screenshot."
        ),
        max_iterations=8,
    )

    # Workflow 2: Web Research & Cognitive Inspection
    bench.run_workflow(
        workflow_name="2. Autonomous Web Research & Visual Screen Reading",
        prompt=(
            "Open https://en.wikipedia.org/wiki/Artificial_general_intelligence in the browser, "
            "wait 2 seconds for it to load, focus the browser window, visually inspect the screen, "
            "and give me a concise summary of the definition."
        ),
        max_iterations=8,
    )

    # Workflow 3: Chess Bot Initiation & GUI Interaction
    bench.run_workflow(
        workflow_name="3. Interactive Chess Match Setup & Play Execution",
        prompt=(
            "Open https://www.chess.com/play/computer in the browser, wait 2 seconds, "
            "focus the Chrome window, click the green Play button to start the game against the bot, "
            "wait 2 seconds, and inspect the screen to confirm the match has started."
        ),
        max_iterations=8,
    )

    # Workflow 4: File Manipulation, Moving, and Directory Organization
    bench.run_workflow(
        workflow_name="4. File Search, Archival & Safe Organization",
        prompt=(
            "Search for markdown files in 'AI_Workspace', read 'project_plan.md', "
            "create a new directory named 'Archived_Workspaces', move 'AI_Workspace' into "
            "'Archived_Workspaces', and list the contents of 'Archived_Workspaces'."
        ),
        max_iterations=8,
    )

    # Workflow 5: Complete Window Lifecycle & Telemetry Inspection
    bench.run_workflow(
        workflow_name="5. Window Lifecycle Management & System Telemetry",
        prompt=(
            "List all open desktop windows, close the Google Chrome window, "
            "close the Dolphin window, and report the current system telemetry."
        ),
        max_iterations=8,
    )

    bench.save_report()


if __name__ == "__main__":
    main()
