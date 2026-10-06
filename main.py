import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown

from config.settings import settings
from core.agent import JarvisAgent
from core.llm_client import LLMClient
from core.safety import SafetyEngine
from tools.base import registry

# Register all capabilities
import tools.system_info
import tools.app_control
import tools.file_ops
import tools.system_ctl
import tools.shell_runner
import tools.browser_dom
import tools.memory_ops
import vision.capture
import vision.grounding
import vision.visual_diff
import gui_driver.mouse_keyboard
import gui_driver.navigator
import voice.speaker
import voice.listener
import voice.wake_word
from memory.store import memory_store
from voice.speaker import speaker
from voice.listener import listener

console = Console()


def display_welcome_banner():
    banner_text = Text()
    banner_text.append("J . A . R . V . I . S .\n", style="bold cyan")
    banner_text.append("Just A Rather Very Intelligent System\n", style="italic white")
    banner_text.append(f"Operating System: Linux (Session: {settings.WORKSPACE_DIR})\n", style="dim cyan")
    mem_count = len(memory_store.list_all())
    banner_text.append(f"Active Provider: {settings.DEFAULT_PROVIDER.upper()} | Model: {settings.DEFAULT_MODEL} | Memory: {mem_count} items", style="dim green")

    console.print(Panel(banner_text, border_style="cyan", expand=False))


def display_tools_table():
    table = Table(title="🔧 Active System Tools", border_style="cyan")
    table.add_column("Tool Name", style="bold green")
    table.add_column("Description", style="white")

    for name, tool in registry.tools.items():
        table.add_row(name, tool.description or "No description provided.")

    console.print(table)


def display_memory_table():
    profile = memory_store.get_profile()
    memories = memory_store.list_all(limit=50)

    table = Table(title="🧠 Persistent Memory Bank", border_style="cyan")
    table.add_column("Category", style="bold yellow")
    table.add_column("Content", style="white")
    table.add_column("Importance", style="green")

    if profile:
        for k, v in profile.items():
            table.add_row("Profile", f"{k.capitalize()}: {v}", "5/5")

    for m in memories:
        table.add_row(m.category.capitalize(), m.content, f"{m.importance}/5")

    if not profile and not memories:
        console.print("[dim yellow]No memories currently stored in the memory bank.[/dim yellow]")
    else:
        console.print(table)


def main():
    display_welcome_banner()

    llm = LLMClient()
    if not llm.is_configured:
        console.print(
            Panel(
                "[bold yellow]⚠️ No API Key Detected![/bold yellow]\n\n"
                "To connect Jarvis to his neural backend, set [bold cyan]GROQ_API_KEY[/bold cyan] or [bold cyan]OPENAI_API_KEY[/bold cyan] in your [bold white].env[/bold white] file.\n"
                "Example:\n"
                "  [green]GROQ_API_KEY=gsk_your_groq_api_key_here[/green]\n\n"
                "[dim](Commands available without API key: 'tools', 'status', 'exit')[/dim]",
                border_style="yellow",
                title="Configuration Required",
            )
        )

    def on_thought(thought: str):
        console.print(f"[dim italic cyan]⚡ {thought}[/dim italic cyan]")

    def on_workflow_plan(plan: str):
        console.print(
            Panel(
                Markdown(plan),
                title="📋 [bold cyan]Planned Execution Flow[/bold cyan]",
                border_style="cyan",
                expand=False,
            )
        )

    def on_tool_call(name: str, args: dict):
        console.print(f"[bold cyan]⚡ Calling Tool:[/bold cyan] [yellow]{name}[/yellow] [dim]args={args}[/dim]")

    def on_tool_result(name: str, output: any, success: bool):
        status_symbol = "[bold green]✓[/bold green]" if success else "[bold red]✗[/bold red]"
        console.print(f"  {status_symbol} [dim green]Result from {name}:[/dim green] [dim]{str(output)[:200]}...[/dim]")

    def confirmation_prompt(tool_name: str, args: dict, reason: str) -> bool:
        console.print(f"\n[bold red]⚠️ SAFETY CONFIRMATION REQUIRED[/bold red]")
        console.print(f"  [yellow]Tool:[/yellow] {tool_name}")
        console.print(f"  [yellow]Arguments:[/yellow] {args}")
        console.print(f"  [yellow]Risk Assessment:[/yellow] {reason}")
        try:
            ans = console.input("[bold white]Authorise Jarvis to execute this action? (y/N): [/bold white]").strip().lower()
            return ans in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            return False

    safety = SafetyEngine(confirmation_handler=confirmation_prompt)

    agent = JarvisAgent(
        llm_client=llm,
        tool_registry=registry,
        safety_engine=safety,
        on_thought=on_thought,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        on_workflow_plan=on_workflow_plan,
    )

    # Hook model fallback notifications into terminal UI
    llm.on_fallback = lambda old_m, new_m, reason: console.print(
        f"\n[bold yellow]⚠️ Rate limit reached for {old_m}. Automatically cascading to {new_m}...[/bold yellow]"
    )

    console.print("\n[dim]Type your command below. Commands: 'exit' to quit, 'reset' to clear context, 'tools' to view tools, 'memory' to view memories, 'status' for telemetry, 'voice' to speak.[/dim]\n")

    while True:
        try:
            user_input = console.input(f"[bold green]{settings.USER_TITLE} > [/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[cyan]Jarvis: Shutting down systems. Have a pleasant day, Sir.[/cyan]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("exit", "quit", "bye"):
            console.print("[cyan]Jarvis: Powering down core systems. Goodbye, Sir.[/cyan]")
            break
        elif cmd == "reset":
            agent.reset()
            console.print("[green]✓ Context buffer reset to baseline.[/green]")
            continue
        elif cmd == "tools":
            display_tools_table()
            continue
        elif cmd in ("memory", "memories"):
            display_memory_table()
            continue
        elif cmd == "status":
            result = registry.execute("get_system_telemetry", {})
            console.print(Panel(str(result.output), title="System Telemetry", border_style="cyan"))
            continue

        is_voice = False
        if cmd in ("voice", "listen"):
            is_voice = True
            console.print("[bold cyan]🎤 Jarvis Voice Mode Active[/bold cyan] [dim](Listening with real-time VAD...)[/dim]")
            speaker.speak("Listening, Sir.", wait=True)
            with console.status("[bold cyan]Listening to microphone (speak now)...[/bold cyan]", spinner="arc"):
                try:
                    speech_text = listener.listen(duration_seconds=8.0, use_vad=True)
                except Exception as e:
                    console.print(f"[red]Error recording speech: {e}[/red]")
                    continue

            if not speech_text:
                console.print("[yellow]Jarvis: I didn't catch that, Sir.[/yellow]")
                continue

            console.print(f"[bold green]{settings.USER_TITLE} (Voice):[/bold green] [italic white]{speech_text}[/italic white]")
            user_input = speech_text

        with console.status("[bold cyan]Jarvis is processing...[/bold cyan]", spinner="dots"):
            response = agent.step(user_input)

        console.print(f"\n[bold cyan]{settings.ASSISTANT_NAME}:[/bold cyan]")
        console.print(Markdown(response))
        console.print()

        if is_voice:
            speaker.speak(response, wait=False)


if __name__ == "__main__":
    main()
