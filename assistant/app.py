# assistant/app.py
import os
import sys
import asyncio
from pathlib import Path

# Ensure project root on sys.path before local imports
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from assistant.agent import run_agent
from assistant.memory import read_context, append_turn


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _recent_log_days() -> int:
    raw = os.getenv("RECENT_LOG_DAYS", "2").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 2


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Second Brain", page_icon="🧠")
st.title("🧠 Second Brain — Local Assistant")
st.caption("Ask questions about your indexed notes or fetch live Notion pages.")

# ── Session state (UI cache only; memory files are the source of truth) ──────

if "history" not in st.session_state:
    st.session_state.history = []

# ── Input form ────────────────────────────────────────────────────────────────

with st.form("query_form", clear_on_submit=True):
    user_input = st.text_input("Your question", placeholder="What did I write about X?")
    submit = st.form_submit_button("Ask")

if submit and user_input:
    st.session_state.history.append({"role": "user", "text": user_input})

    enable_memory = _env_bool("ENABLE_MEMORY", default=True)
    memory_block = (
        read_context(days=_recent_log_days()) if enable_memory else ""
    )

    with st.spinner("Thinking…"):
        try:
            text = asyncio.run(run_agent(user_input, memory_block=memory_block))
        except Exception as e:
            text = f"Agent error: {e}"

    st.session_state.history.append({"role": "assistant", "text": text})

    if enable_memory:
        try:
            append_turn(user_input, text)
        except Exception as e:
            print(f"[memory] append failed: {e}", file=sys.stderr)

# ── Conversation history ──────────────────────────────────────────────────────

if st.session_state.history:
    st.divider()
    for turn in reversed(st.session_state.history):
        role = turn["role"]
        with st.chat_message(role):
            st.markdown(turn["text"])