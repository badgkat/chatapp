"""server.py – stable version
Flask front‑end for the local speech assistant stack.

Highlights
----------
* Single absolute CONFIG_PATH.
* Robust config helpers without recursion.
* `/download-model` skips download when file present.
* All JSON routes safe.
* Flask session enabled via a per‑run random secret‑key (override with $SECRET_KEY).
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from pathlib import Path
from typing import Any, Dict

import requests
from flask import Flask, Response, jsonify, render_template, request, session
from flask_cors import CORS

# ---------------------------------------------------------------------------
# local worker API                                                            
# ---------------------------------------------------------------------------
from worker import (  # type: ignore
    speech_to_text,
    text_to_speech,
    process_message,
    list_voices,
    load_model,
    DEFAULT_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Flask app                                                                   
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))  # enables sessions
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------------------------------------------------------------------
# Configuration handling                                                      
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = BASE_DIR / "config.json"

DEFAULT_PARAMS: Dict[str, Any] = {
    "n_ctx": 2048,
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95,
    "n_threads": 12,
    "n_batch": 64,
    "n_gpu_layers": 0,
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "model_path": "",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "params": DEFAULT_PARAMS,
}


def _read_cfg() -> Dict[str, Any]:
    """Return cfg dict; never raises."""
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_CONFIG.copy()


def _write_cfg(cfg: Dict[str, Any]) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def save_cfg(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Apply *patch* to stored config and write back. Return new cfg."""
    cfg = _read_cfg()
    cfg.update(patch)
    _write_cfg(cfg)
    return cfg

# Ensure the file exists on first run
if not CONFIG_PATH.exists():
    _write_cfg(DEFAULT_CONFIG)

# ---------------------------------------------------------------------------
# Static model catalogue                                                      
# ---------------------------------------------------------------------------

MODEL_OPTIONS: Dict[str, str] = {
    "gemma-3-4b-it-Q4_K_M.gguf": "https://huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
    "phi-2-Q4_K_M.gguf": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
}

# ---------------------------------------------------------------------------
# Helpers                                                                     
# ---------------------------------------------------------------------------

def ensure_models_dir() -> Path:
    path = BASE_DIR / "models"
    path.mkdir(exist_ok=True)
    return path

# ---------------------------------------------------------------------------
# Page routes                                                                 
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Main assistant page. Shows warning if no model configured."""
    cfg = _read_cfg()
    model_ready = bool(cfg.get("model_path")) and Path(cfg["model_path"]).is_file()
    return render_template("index.html", model_ready=model_ready)


@app.route("/setup", methods=["GET"])
def setup():
    return render_template("setup.html", model_options=MODEL_OPTIONS.keys())

# ---------------------------------------------------------------------------
# Model management                                                            
# ---------------------------------------------------------------------------

@app.route("/download-model", methods=["POST"])
def download_model() -> Response:
    """Download model if missing or just register its path."""
    data = request.get_json(force=True) or {}
    label = data.get("label") or data.get("model")
    if not label:
        return jsonify({"error": "no model supplied"}), 400
    if label not in MODEL_OPTIONS:
        return jsonify({"error": "invalid model label"}), 400

    url = MODEL_OPTIONS[label]
    target = ensure_models_dir() / Path(url).name

    def _gen():
        # already have it
        if target.is_file():
            save_cfg({"model_path": str(target)})
            yield json.dumps({"progress": 100, "status": "done"}) + "\n"
            return
        # download
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                size = int(r.headers.get("content-length", 0)) or 1
                done = 0
                with open(target, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        pct = int(done * 100 / size)
                        yield json.dumps({"progress": pct}) + "\n"
        except Exception as exc:
            yield json.dumps({"error": str(exc)}) + "\n"
            return
        save_cfg({"model_path": str(target)})
        yield json.dumps({"progress": 100, "status": "done"}) + "\n"

    return Response(_gen(), mimetype="text/plain")


@app.route("/model-info", methods=["GET"])
def model_info():
    cfg = _read_cfg()
    model_path = cfg.get("model_path")
    ctx_max = 0
    if model_path and Path(model_path).is_file():
        try:
            llm = load_model()
            attr = getattr(llm, "n_ctx", 0)
            ctx_max = attr() if callable(attr) else int(attr)
        except Exception:
            ctx_max = 0
    return jsonify({"max_context": ctx_max})

# ---------------------------------------------------------------------------
# Config endpoints                                                            
# ---------------------------------------------------------------------------

@app.route("/get-system-prompt", methods=["GET"])
def get_system_prompt():
    return jsonify({"prompt": _read_cfg().get("system_prompt", DEFAULT_SYSTEM_PROMPT)})


@app.route("/update-system-prompt", methods=["POST"])
def update_system_prompt():
    new_prompt = request.json.get("prompt", "").strip()
    save_cfg({"system_prompt": new_prompt})
    return jsonify({"status": "ok"})


@app.route("/get-params", methods=["GET"])
def get_params():
    return jsonify(_read_cfg().get("params", DEFAULT_PARAMS))


@app.route("/update-params", methods=["POST"])
def update_params():
    body = request.json or {}
    params = _read_cfg().get("params", DEFAULT_PARAMS).copy()
    for k, v in body.items():
        if k in params:
            params[k] = v
    save_cfg({"params": params})
    return jsonify({"status": "ok"})

# ---------------------------------------------------------------------------
# Voices                                                                      
# ---------------------------------------------------------------------------

@app.route("/list-voices", methods=["GET"])
def list_voices_route():
    return jsonify({"voices": list_voices() or []})

# ---------------------------------------------------------------------------
# Assistant endpoints                                                         
# ---------------------------------------------------------------------------

@app.route("/speech-to-text", methods=["POST"])
def stt():                                   # server.py
    audio_data = request.get_data()          # raw body
    if not audio_data:
        return jsonify({"error": "no audio"}), 400
    text = speech_to_text(audio_data)
    return jsonify({"text": text})


@app.route("/text-to-speech", methods=["POST"])
def tts():
    text = request.json.get("text", "")
    voice = request.json.get("voice", "af_heart")
    wav = text_to_speech(text, voice)
    return jsonify({"audio": base64.b64encode(wav).decode()})


@app.route("/process-message", methods=["POST"])
def process_route():                        # server.py
    data = request.get_json(force=True) or {}
    user_msg = data.get("userMessage", "")
    voice    = data.get("voice") or "af_heart"

    # session id
    sid = session.get("sid") or secrets.token_hex(8)
    session["sid"] = sid

    # LLM reply
    response_txt = process_message(user_msg, session_id=sid).strip()

    # TTS
    wav        = text_to_speech(response_txt, voice)
    audio_b64  = base64.b64encode(wav).decode()

    return jsonify({
        "ResponseText":  response_txt,
        "ResponseSpeech": audio_b64
    })

# ---------------------------------------------------------------------------
# Run app                                                                     
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
