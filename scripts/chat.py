"""CLI entry point: conversational PuzzleAgent with multi-agent pipeline.

Supports:
  - Interactive mode: python scripts/chat.py
  - Single query:    python scripts/chat.py --query "..."

The agent uses the full stack: QueryRewriter, MultiAgentPipeline
(Generator->Solver->Reviewer), Tracer, SessionManager, and
first/supplement mode routing.
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
    args = ap.parse_args()

    print("Loading PuzzleAgent...")
    agent = create_agent(args.config)
    print(f"Ready. {len(agent.rule_index)} rules loaded.\n")

    if args.query:
        response = agent.chat(args.query)
        print(response)
        if args.trace:
            _print_trace()
    else:
        print("Type 'quit' to exit, 'help' for commands, 'trace' for stats.\n")
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
                _print_trace()
                continue
            if user_input.lower() in ("help", "?"):
                _print_help()
                continue

            response = agent.chat(user_input)
            print(response)
            print()


def _print_trace():
    tracer = get_tracer()
    summary = tracer.summary()
    print(f"\n--- Tracer Stats ---")
    print(f"  LLM calls: {summary['total_calls']}")
    print(f"  Failed:    {summary['failed']}")
    print(f"  Pass rate: {summary['pass_rate']}")
    print(f"  By role:   {summary.get('by_role', {})}")
    print(f"  Session:   {summary['session_minutes']:.1f} min\n")


def _print_help():
    print("""
Commands:
  help  - show this
  trace - show tracer statistics
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
