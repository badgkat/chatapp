"""
worker.py ― core back-end logic
• Loads a local GGUF model with llama-cpp-python
• Handles STT (Whisper) and TTS (Kokoro)
• Maintains lightweight chat state per session
• Guarantees plain-text answers without role labels
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Tuple

import soundfile as sf
import whisper
from kokoro import KPipeline
from llama_cpp import Llama

from context_state import ChatState   # small @dataclass storing a .window list

# --------------------------------------------------------------------------- #
# Config & globals
# --------------------------------------------------------------------------- #
CHAT_STATES: dict[str, ChatState] = {}          # keyed by session ID
ROOT              = Path(__file__).parent
CONFIG_PATH       = ROOT / "config.json"
DEFAULT_MODEL     = ROOT / "models" / "gemma-3-4b-it-Q4_K_M.gguf"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    )

# Regex to strip role labels that sometimes bleed from fine-tuned models
ROLE_RE = re.compile(r'^\s*(?:###\s*)?(?:Response|Assistant)\s*:?\s*', re.I)

llm: Llama | None = None               # lazily loaded singleton
whisper_model = whisper.load_model("tiny.en")
tts_pipeline  = KPipeline(lang_code="a")


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def load_voice(voice_name):
    local_path = f"./voices/{voice_name}.pt"
    if os.path.exists(local_path):
        def tts_func(text):
            return tts_pipeline(text, voice=local_path)
    else:
        def tts_func(text):
            return tts_pipeline(text, voice=voice_name)
    return tts_func

def strip_role(text: str) -> str:
    """Remove 'Assistant:' or '### Response' etc. from start of string."""
    return ROLE_RE.sub("", text).lstrip()


def clean_tts_text(text: str) -> str:
    """Sanitize for speech: strip role tags, markdown, and special chars."""
    text = strip_role(text)
    text = re.sub(r"[*_~`>#-]", "", text)          # remove markdown symbols
    text = re.sub(r"\s{2,}", " ", text)            # compress extra spaces
    return text.strip()


def estimate_tokens(text: str) -> int:
    """Crude token estimator (≈ 4 chars/token)."""
    return max(1, len(text) // 4)


def load_model() -> Llama:
    """Lazy-load and cache llama-cpp model based on config.json."""
    global llm
    if llm is not None:
        return llm

    # read optional config
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open() as f:
            cfg = json.load(f)
        model_path = Path(cfg.get("model_path", DEFAULT_MODEL))
        params     = cfg.get("params", {})
    else:
        model_path = DEFAULT_MODEL
        params     = {}

    llm = Llama(
        model_path=str(model_path),
        n_ctx=params.get("n_ctx", 2048),
        n_threads=params.get("n_threads", os.cpu_count() or 8),
        n_batch=params.get("n_batch", 64),
        n_gpu_layers=params.get("n_gpu_layers", 0),   # CPU by default
    )
    return llm


def build_prompt(state: ChatState,
                 user_msg: str,
                 system_prompt: str,
                 budget_tokens: int) -> str:
    """
    Assemble system + recent chat history + new user message
    ensuring token count < budget_tokens.
    """
    hist: List[Tuple[str, str]] = state.window.copy()
    hist.append(("user", user_msg))

    # walk history backwards until under budget
    parts: List[str] = [f"### System\n{system_prompt}"]
    running_tokens = estimate_tokens(parts[0])

    for role, content in reversed(hist):
        segment = f"\n### {role.capitalize()}\n{content}"
        seg_tokens = estimate_tokens(segment)
        if running_tokens + seg_tokens > budget_tokens:
            break
        parts.insert(1, segment)          # keep chronological order
        running_tokens += seg_tokens

    return "".join(parts).rstrip()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def process_message(user_text: str, session_id: str = "default") -> str:
    """
    Run one inference turn and update chat history.
    Returns assistant reply (plain text, no role tags, no <END>).
    """

    cfg         = json.load(CONFIG_PATH.open()) if CONFIG_PATH.exists() else {}
    gen_params  = cfg.get("params", {})
    system_p    = cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    max_tokens = int(gen_params.get("max_tokens", 200))
    n_ctx      = int(gen_params.get("n_ctx", 2048))
    budget     = n_ctx - max_tokens - 32            # margin

    state  = CHAT_STATES.setdefault(session_id, ChatState())
    prompt = build_prompt(state, user_text, system_p, budget)

    constraint = (
        "Reply with plain text only. No role labels or markdown headings.\n"
        f"Reply ≤{max_tokens} tokens and finish with <END>"
    )

    raw = load_model()(
        prompt + "\n### Constraint\n" + constraint,
        max_tokens=max_tokens,
        temperature=gen_params.get("temperature", 0.7),
        top_p=gen_params.get("top_p", 0.95),
        stop=["<END>"]
    )["choices"][0]["text"]

    # Extract only the portion intended as the assistant's message
    # Remove <END> and any trailing junk
    text = raw.strip()
    text = text.split("<END>")[0].strip()

    # Optionally strip any leading label
    reply = strip_role(text)

    state.window.append(("assistant", reply))
    return reply


def speech_to_text(audio_bytes: bytes) -> str:
    """Whisper STT from raw audio bytes (webm, wav, etc.)."""
    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = whisper_model.transcribe(tmp.name, language="en")
    return result["text"].strip()


def text_to_speech(text: str, voice: str = "af_heart") -> bytes:
    if voice not in list_voices():
        voice = "af_heart"

    cleaned_text = clean_tts_text(text)
    tts_func = load_voice(voice)
    generator = tts_func(cleaned_text)

    full_audio = []
    for _, _, audio in generator:
        full_audio.extend(audio)

    buf = io.BytesIO()
    sf.write(buf, full_audio, 24000, format='WAV')
    return buf.getvalue()



def list_voices() -> List[str]:
    """Return available voice IDs by scanning ./voices/*.pt"""
    voice_dir = ROOT / "voices"
    return sorted(p.stem for p in voice_dir.glob("*.pt")) if voice_dir.is_dir() else []
