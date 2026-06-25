# Is NOT integrated with memory module. Just a simple CLI to run the agent with a single message or in REPL mode.
from __future__ import annotations

import asyncio
import sys

from assistant.agent import agent


def _result_text(result) -> str:
    # pydantic-ai result surface changed across versions.
    if hasattr(result, "data"):
        return result.data
    if hasattr(result, "output"):
        return result.output
    if hasattr(result, "result"):
        return result.result
    return str(result)


async def _run_once(message: str) -> int:
    result = await agent.run(message)
    print(_result_text(result))
    return 0


async def _repl() -> int:
    print("Second Brain agent (local CLI). Type 'exit' to quit.\n")
    while True:
        msg = input("> ").strip()
        if not msg:
            continue
        if msg.lower() in {"exit", "quit", "q"}:
            return 0
        result = await agent.run(msg)
        print(_result_text(result))
        print()

async def _amain(argv: list[str]) -> int:
    if len(argv) > 1:
        return await _run_once(" ".join(argv[1:]))
    return await _repl()

def main() -> None:
    raise SystemExit(asyncio.run(_amain(sys.argv)))


if __name__ == "__main__":
    main()

