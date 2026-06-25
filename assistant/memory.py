# assistant/memory.py
"""
File-based memory layer for the Second Brain agent.

Layout:
    memory/
        MEMORY.md                # long-term, always loaded
        YYYY-MM-DD.md            # daily logs (append-only)

Two operations:
    - read(): returns a string with MEMORY.md + recent daily logs,
      formatted for inclusion in the LLM system prompt.
    - append(): writes a new turn pair to today's daily log file.

Daily log entry format (per turn, markdown):
    - **[HH:MM]** **You:** <user message>
      **Assistant:** <assistant reply>

Long-term MEMORY.md layout (curated, manually edited):
    # Long-Term Memory
    ## User Preferences
    - ...
    ## Key Decisions
    - ...
    ## Lessons Learned
    - ...
    ## Standard Procedures
    - ...
"""
from __future__ import annotations

import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional

# Resolve memory/ relative to the repo root (parent of the assistant/ package).
_REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_ROOT = _REPO_ROOT / "memory"
MEMORY_FILE = MEMORY_ROOT / "MEMORY.md"

# Default skeleton for MEMORY.md the first time it's read.
_MEMORY_SKELETON = """# Long-Term Memory

## User Preferences
- (none yet)

## Key Decisions
- (none yet)

## Lessons Learned
- (none yet)

## Standard Procedures
- (none yet)
"""

def _today_log_path(day: Optional[date] = None) -> Path:
    d = day or date.today()
    return MEMORY_ROOT / f"{d.isoformat()}.md"


def _ensure_root() -> None:
    MEMORY_ROOT.mkdir(parents=True, exist_ok=True)


def _ensure_memory_file() -> None:
    """Create MEMORY.md w/ skeleton if it doesn't exist."""
    _ensure_root()
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(_MEMORY_SKELETON, encoding="utf-8")


def _ensure_today_log() -> Path:
    """Create today's log file w/ a heading if it doesn't exist."""
    _ensure_root()
    path = _today_log_path()
    if not path.exists():
        path.write_text(f"# Daily Log — {date.today().isoformat()}\n\n", encoding="utf-8")
    return path


def _read_recent_logs(days: int) -> str:
    """Read the last `days` daily logs (today + previous), oldest first."""
    today = date.today()
    blocks: list[str] = []
    # Iterate from oldest -> newest so the LLM reads them in order.
    for offset in range(days - 1, -1, -1):
        target = date.fromordinal(today.toordinal() - offset)
        path = _today_log_path(target)
        if path.exists():
            blocks.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(blocks)


def read_context(days: int = 2) -> str:
    """Return the full memory context: MEMORY.md + recent daily logs.

    Suitable for direct injection into the system prompt.
    """
    _ensure_memory_file()
    long_term = MEMORY_FILE.read_text(encoding="utf-8").strip()
    recent = _read_recent_logs(days).strip()
    if not recent:
        return long_term
    return f"{long_term}\n\n---\n\n{recent}"


def append_turn(user_msg: str, assistant_msg: str) -> None:
    """Append a single turn pair (user + assistant) to today's daily log."""
    if not user_msg.strip() and not assistant_msg.strip():
        return
    path = _ensure_today_log()
    ts = datetime.now().strftime("%H:%M")
    entry = (
        f"### {ts}\n"
        f"[IN-CONVERSATION CONTEXT — not knowledge base]\n"
        f"User: {user_msg.strip()}\n"
        f"Assistant: {assistant_msg.strip()}\n\n"
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(entry)

def list_daily_logs() -> list[str]:
    """Return names of all daily log files, newest first."""
    if not MEMORY_ROOT.exists():
        return []
    return sorted(
        (p.name for p in MEMORY_ROOT.glob("????-??-??.md")),
        reverse=True,
    )