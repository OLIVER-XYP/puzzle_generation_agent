"""CLI entry point: conversational PuzzleAgent with multi-agent pipeline.

Supports:
  - Interactive mode: python scripts/chat.py
  - Single query:    python scripts/chat.py --query "..."
  - Tool mode:       python scripts/chat.py --tools --query "..."

The agent uses the full stack: QueryRewriter, MultiAgentPipeline
(Generator->Solver->Reviewer), Tracer, SessionManager, and
first/supplement mode routing.

With --tools, Generator/Solver/Reviewer gain access to deterministic
tools (brute-force solvers, validators) via OpenAI function calling.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from puzzle_agent.agent import create_agent          # noqa: E402
from puzzle_agent.tracer import get_tracer           # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Conversational puzzle generation agent")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--query", default=None,
                    help="single query mode (non-interactive)")
    ap.add_argument("--trace", action="store_true",
                    help="print tracer summary after query")
    ap.add_argument("--tools", dest="tools", action="store_true", default=None,
                    help="enable LLM function calling (default: on)")
    ap.add_argument("--no-tools", dest="tools", action="store_false",
                    help="disable LLM function calling (plain text generation)")
    args = ap.parse_args()

    print("Loading PuzzleAgent...")
    agent = create_agent(args.config)
    # Tools default to the pipeline/config setting; CLI flag overrides
    if args.tools is not None:
        agent.pipeline.use_tools = args.tools
    if agent.pipeline.use_tools:
        print("[tools] Function calling enabled "
              f"({len(agent.tools.list_tools())} tools available)")
    else:
        print("[tools] Function calling disabled")
    print(f"Ready. {len(agent.rule_index)} rules loaded.\n")

    if args.query:
        response = agent.chat(args.query)
        print(response)
        if args.trace:
            _print_trace(agent)
    else:
        tool_note = " (tools ON)" if agent.pipeline.use_tools else ""
        print(f"Type 'quit' to exit, 'help' for commands, 'trace' for stats.{tool_note}\n")
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if user_input.lower() == "trace":
                _print_trace(agent)
                continue
            if user_input.lower() in ("help", "?"):
                _print_help()
                continue
            if user_input.lower() == "tools":
                _print_tools(agent)
                continue

            response = agent.chat(user_input)
            print(response)
            print()


def _print_trace(agent=None):
    tracer = get_tracer()
    summary = tracer.summary()
    print(f"\n--- Tracer Stats ---")
    print(f"  LLM calls: {summary['total_calls']}")
    print(f"  Failed:    {summary['failed']}")
    print(f"  Pass rate: {summary['pass_rate']}")
    print(f"  By role:   {summary.get('by_role', {})}")
    print(f"  Session:   {summary['session_minutes']:.1f} min")

    # Tool stats
    if agent and agent.pipeline.use_tools:
        ts = agent.pipeline.get_tool_stats()
        print(f"\n--- Tool Stats ---")
        print(f"  Generator tool rounds: {ts['generator'].get('tool_rounds', 0)}")
        print(f"  Solver tool rounds:    {ts['solver'].get('tool_rounds', 0)}")
        print(f"  Reviewer tool rounds:  {ts['reviewer'].get('tool_rounds', 0)}")
        print(f"  Total tool calls:      {ts['total_rounds']}")
    print()


def _print_tools(agent=None):
    """List available tools."""
    if agent:
        tools = agent.tools.list_tools()
        print(f"\n--- {len(tools)} Tools Available ---")
        for t in tools:
            print(f"  • {t['name']}: {t['desc'][:80]}")
        print()
    else:
        from puzzle_agent.tools import get_tools
        tools = get_tools().list_tools()
        print(f"\n--- {len(tools)} Tools Defined ---")
        for t in tools:
            print(f"  • {t['name']}: {t['desc'][:80]}")
        print()


def _print_help():
    print("""
Commands:
  help  - show this
  trace - show tracer + tool statistics
  tools - list available function-calling tools
  quit  - exit

You can say things like:
  "列出所有规则"
  "查看规则10"
  "给规则4生成5道题"
  "出几道简单的数学题"
  "验证一下质量"
  "导出到 data/out.jsonl"
""")


if __name__ == "__main__":
    main()
