# Local Speech-to-Speech Voice Assistant

A Flask-based voice assistant powered by local models for speech-to-text (Whisper), language generation (LLM), and text-to-speech (Kokoro). Runs entirely offline. No external APIs required.

## Features

- Record voice and transcribe with Whisper  
- Respond using a locally hosted LLM (e.g., Gemma 3B Q4)  
- Reply using realistic TTS (Kokoro voices)  
- Web interface with real-time chat and audio playback  
- Model setup and download from HuggingFace with progress tracking  

---

## Requirements

- Python 3.11.x (not compatible with 3.12+)
- FFmpeg (for audio conversion)
- pip packages in `requirements.txt`

---

## Setup (Manual)

1. Clone the repo:

```
git clone https://github.com/yourusername/chatapp.git
cd chatapp
```

2. Create and activate a virtual environment:

```
python3.11 -m venv kokoroenv
source kokoroenv/bin/activate  # or use Scripts\activate on Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the app:

```
python server.py
```

5. Open http://localhost:8000 in your browser.

---

## Setup (Docker)

1. Build the image:

```
docker build -t chatapp .
```

2. Run the container:

```
docker run -p 8000:8000 chatapp
```

---

## Model Setup

On first launch, select and download an LLM model from the setup screen.

- Models are saved to `models/`
- `.gguf` files are gitignored
- Config is saved to `config.json`

---

## Voice Configuration

Drop `.pt` voice files into the `voices/` directory. They’ll show up in the web UI dynamically.

---

## Notes

- Ensure `ffmpeg` is available in your system path
- Only works with Python 3.11.x (not 3.12+)
- GPU usage is optional and disabled by default

---

## Roadmap

- [ ] Multi-model support
- [ ] User profiles
- [ ] Config export/import
- [ ] Streamed generation

---

## License

MIT License. See `LICENSE` file.
