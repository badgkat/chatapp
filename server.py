import base64
import json
from flask import Flask, render_template, request
from flask_cors import CORS
import os
import requests
from worker import (
    speech_to_text,
    text_to_speech,
    process_message,
    list_voices,
    is_model_ready
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
CONFIG_PATH = "config.json"

MODEL_OPTIONS = {
    "Gemma 3B Q4": "https://huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf"
}

@app.route("/", methods=["GET"])
def index():
    model_ready = is_model_ready()
    return render_template("index.html", model_ready=model_ready)

@app.route("/setup", methods=["GET", "POST"])
def setup_model():
    os.makedirs("models", exist_ok=True)

    if request.method == "POST":
        model_label = request.form["model"]
        model_url = MODEL_OPTIONS[model_label]
        local_path = os.path.join("models", os.path.basename(model_url))

        try:
            with requests.get(model_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            with open("config.json", "w") as f:
                json.dump({"model_path": local_path}, f)

            return f"Downloaded and configured: {model_label}"
        except Exception as e:
            return f"Download failed: {e}", 500
    return render_template("setup.html", model_options=MODEL_OPTIONS.keys(), model_ready=is_model_ready())

@app.route("/list-voices", methods=["GET"])
def list_voices_route():
    return app.response_class(
        response=json.dumps({"voices": list_voices()}),
        status=200,
        mimetype="application/json"
    )

@app.route("/speech-to-text", methods=["POST"])
def speech_to_text_route():
    print("Processing speech-to-text")
    audio_binary = request.data
    text = speech_to_text(audio_binary)
    return app.response_class(
        response=json.dumps({'text': text}),
        status=200,
        mimetype='application/json'
    )

@app.route("/process-message", methods=["POST"])
def process_message_route():
    data = request.json
    user_message = data.get("userMessage", "")
    voice = data.get("voice", "af_heart")

    print("user_message:", user_message)
    print("voice:", voice)

    response_text = process_message(user_message).strip()
    response_text = os.linesep.join([s for s in response_text.splitlines() if s])

    response_audio = text_to_speech(response_text, voice)
    response_audio_base64 = base64.b64encode(response_audio).decode("utf-8")

    return app.response_class(
        response=json.dumps({
            "ResponseText": response_text,
            "ResponseSpeech": response_audio_base64
        }),
        status=200,
        mimetype="application/json"
    )

@app.route("/download-model", methods=["POST"])
def download_model():
    from flask import Response

    data = request.json
    model_label = data["model"]
    model_url = MODEL_OPTIONS[model_label]
    local_path = os.path.join("models", os.path.basename(model_url))

    def generate():
        try:
            with requests.get(model_url, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                downloaded = 0

                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            percent = int((downloaded / total) * 100)
                            status = f"{percent}% downloaded..."
                            yield json.dumps({"progress": percent, "status": status}) + "\n"

            with open(CONFIG_PATH, "w") as f:
                json.dump({"model_path": local_path}, f)

            yield json.dumps({"progress": 100, "status": "Download complete."}) + "\n"

        except Exception as e:
            yield json.dumps({"progress": 0, "status": f"Download failed: {str(e)}"}) + "\n"

    return Response(generate(), mimetype="text/plain")

# This must come last
if __name__ == "__main__":
    app.run(port=8000, host='0.0.0.0')