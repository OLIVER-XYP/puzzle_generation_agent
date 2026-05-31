"""Interactive chat with the puzzle generation agent.

Usage:
    python scripts/chat.py                     # interactive mode
    python scripts/chat.py "给规则1生成5道题"   # single message
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from puzzle_agent.agent import create_agent


def main():
    agent = create_agent()

    if len(sys.argv) > 1:
        # Single message mode
        msg = " ".join(sys.argv[1:])
        print(f"\n👤 You: {msg}\n")
        resp = agent.chat(msg)
        print(f"🤖 Agent: {resp}\n")
    else:
        # Interactive mode
        print("=" * 60)
        print("  Puzzle Generation Agent (Type 'exit' to quit)")
        print("  Try: 帮我看看有哪些规则")
        print("       给规则25生成3道中等难度的题")
        print("       检查一下现在的数据质量")
        print("=" * 60)
        while True:
            try:
                user = input("\n👤 You: ").strip()
                if not user:
                    continue
                if user.lower() in ("exit", "quit", "q"):
                    print("Bye!")
                    break
                resp = agent.chat(user)
                print(f"\n🤖 Agent: {resp}\n")
            except KeyboardInterrupt:
                print("\nBye!")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {e}")


if __name__ == "__main__":
    main()
