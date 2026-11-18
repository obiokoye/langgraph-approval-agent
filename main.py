"""Terminal UI for the LangGraph approval agent.

Usage:
    python main.py                          # chat; a new thread each run
    python main.py --thread shop-1          # chat on a named thread (resumes it if paused)
    python main.py "Refund order A1042"     # one message, then exit
    python main.py --auto reject "Refund order A1042"   # non-interactive decisions
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from agent import Shop, build_graph
from agent.providers import make_client

console = Console()


def fmt_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())


def ask_decisions(actions: list[dict[str, Any]], auto: str | None) -> dict[str, dict[str, Any]]:
    table = Table(show_header=True, header_style="bold yellow", box=None, padding=(0, 1))
    table.add_column("#", style="dim")
    table.add_column("action", style="bold")
    table.add_column("arguments")
    for i, a in enumerate(actions, 1):
        table.add_row(str(i), a["tool"], "\n".join(f"{k}: {v}" for k, v in a["args"].items()))
    console.print(Panel(table, title="[bold]‖ Approval needed[/bold]", subtitle="graph paused · state saved to SQLite",
                        border_style="yellow", expand=False))

    decisions: dict[str, dict[str, Any]] = {}
    for i, a in enumerate(actions, 1):
        if auto:
            choice = auto[0]
            console.print(f"  [{i}] {a['tool']}: [bold]{auto}[/bold] (--auto)")
        else:
            choice = Prompt.ask(f"  [{i}] {a['tool']}: [green]a[/green]pprove / [red]r[/red]eject / [cyan]e[/cyan]dit",
                                choices=["a", "r", "e"], default="a", console=console)
        if choice == "a":
            decisions[a["id"]] = {"action": "approve"}
        elif choice == "r":
            reason = "rejected by reviewer" if auto else Prompt.ask("      reason", default="rejected by reviewer", console=console)
            decisions[a["id"]] = {"action": "reject", "reason": reason}
        else:
            new_args = {}
            for key, value in a["args"].items():
                answer = Prompt.ask(f"      {key}", default=str(value), console=console)
                if isinstance(value, (int, float)):
                    number = float(answer)
                    new_args[key] = int(number) if number.is_integer() else number
                else:
                    new_args[key] = answer
            decisions[a["id"]] = {"action": "edit", "args": new_args, "reason": "edited by reviewer"}
    return decisions


def run_turn(graph: Any, config: dict[str, Any], payload: Any, auto: str | None, thread: str) -> None:
    """Stream graph updates, pausing for approval whenever the graph interrupts."""
    path: list[str] = []
    # call_id -> tool name, seeded from saved state so resumed runs can label results.
    names = {tc["id"]: tc["function"]["name"]
             for m in graph.get_state(config).values.get("messages", [])
             for tc in m.get("tool_calls") or []}
    while True:
        interrupted: list[dict[str, Any]] | None = None
        for chunk in graph.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    interrupted = update[0].value["actions"]
                    path.append("review ‖ paused")
                    continue
                if not (node == "review" and path and path[-1] == "review ‖ paused"):
                    path.append(node)
                if node == "agent":
                    msg = update["messages"][-1]
                    calls = msg.get("tool_calls") or []
                    if calls:
                        if msg["content"]:
                            console.print(f"[blue]agent ›[/blue] [italic]{msg['content']}[/italic]")
                        for tc in calls:
                            names[tc["id"]] = tc["function"]["name"]
                            args = json.loads(tc["function"]["arguments"] or "{}")
                            console.print(f"  [dim]↳ calls[/dim] [magenta]{tc['function']['name']}[/magenta][dim]({fmt_args(args)})[/dim]")
                    else:
                        console.print(f"[bold blue]agent ›[/bold blue] {msg['content']}")
                elif node == "tools":
                    for m in update["messages"]:
                        result = json.loads(m["content"])
                        style = "red" if result.get("status") == "rejected" or "error" in result else "green"
                        short = ", ".join(f"{k}: {v}" for k, v in list(result.items())[:4])
                        console.print(f"  [{style}]◆ {names.get(m['tool_call_id'], 'tool')}[/{style}] [dim]→ {short}[/dim]")
                    for entry in update.get("audit", []):
                        mark = {"approved": "[green]✔ approved[/green]", "edited": "[cyan]✎ edited + approved[/cyan]",
                                "rejected": "[red]✘ rejected[/red]"}[entry["decision"]]
                        console.print(f"  [dim]audit:[/dim] {entry['tool']} {mark}")
        if interrupted is None:
            break
        try:
            decisions = ask_decisions(interrupted, auto)
        except (KeyboardInterrupt, EOFError):
            console.print(f"\n[yellow]Left paused.[/yellow] Resume any time with: "
                          f"[bold]python main.py --thread {thread}[/bold]")
            raise SystemExit(0)
        console.print("[dim]▶ resuming graph with your decisions[/dim]")
        payload = Command(resume=decisions)
    console.print(f"[dim]path: {' → '.join(path)} → END[/dim]\n")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Support agent that pauses for human approval (LangGraph).")
    parser.add_argument("message", nargs="*", help="send one message and exit")
    parser.add_argument("--thread", default=None, help="thread id to start or resume")
    parser.add_argument("--db", default="checkpoints.sqlite", help="SQLite file for saved graph state")
    parser.add_argument("--auto", choices=["approve", "reject"], help="answer approvals automatically")
    opts = parser.parse_args()

    info = make_client()
    thread = opts.thread or f"t-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 20}
    checkpointer = SqliteSaver(sqlite3.connect(opts.db, check_same_thread=False))
    graph = build_graph(info.client, info.model, Shop(), checkpointer)

    console.print(Panel.fit(
        f"[bold green]LangGraph Approval Agent[/bold green]  model [cyan]{info.model}[/cyan]  thread [cyan]{thread}[/cyan]\n"
        "[dim]Refunds and customer emails pause for your approval. Type 'exit' to quit.[/dim]",
        border_style="green"))
    if info.demo:
        console.print("[bold yellow]Demo mode: scripted AI replies[/bold yellow] [dim](set GEMINI_API_KEY in .env for a real model)[/dim]\n")

    def send(text: str) -> None:
        try:
            run_turn(graph, config, {"messages": [{"role": "user", "content": text}]}, opts.auto, thread)
        except GraphRecursionError:
            console.print("[red]Stopped: too many steps in one turn.[/red]\n")

    # A thread that was left waiting for approval picks up where it stopped.
    if graph.get_state(config).next:
        console.print(f"[yellow]Thread {thread} is paused for approval. Resuming…[/yellow]")
        run_turn(graph, config, None, opts.auto, thread)

    if opts.message:
        send(" ".join(opts.message))
        return
    while True:
        try:
            text = console.input("[bold]you › [/bold]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit"}:
            break
        if text:
            send(text)


if __name__ == "__main__":
    sys.exit(main())
