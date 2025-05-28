import whisper
from kokoro import KPipeline
import soundfile as sf
import io
from llama_cpp import Llama
import os
import re
import subprocess
import tempfile
import json



# Load Whisper and Kokoro models once
whisper_model = whisper.load_model("tiny.en")
tts_pipeline = KPipeline(lang_code='a')

CONFIG_PATH = "config.json"
DEFAULT_MODEL_PATH = "./models/gemma-3-4b-it-Q4_K_M.gguf"
llm = None

def load_model():
    global llm
    if llm is not None:
        return llm

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            config = json.load(f)
            model_path = config.get("model_path", DEFAULT_MODEL_PATH)
    else:
        model_path = DEFAULT_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Visit http://localhost:8000/setup to download one."
        )

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=12,
        n_batch=64,
        n_gpu_layers=0
    )
    return llm

def is_model_ready():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            model_path = json.load(f).get("model_path")
            return model_path and os.path.exists(model_path)
    return False

def load_voice(voice_name):
    local_path = f"./voices/{voice_name}.pt"
    if os.path.exists(local_path):
        def tts_func(text):
            return tts_pipeline(text, voice=local_path)
    else:
        def tts_func(text):
            return tts_pipeline(text, voice=voice_name)
    return tts_func

    
def clean_tts_text(text: str) -> str:
    # Remove markdown-style emphasis (*text* or _text_)
    return re.sub(r"[*_](.*?)[*_]", r"\1", text)

def process_message(user_input: str) -> str:
    """Generate a response using the local LLM."""
    prompt = f"""<start_of_turn>user
    {user_input}<end_of_turn>
    <start_of_turn>model
    """
    llm_instance = load_model()
    output = llm_instance(
        prompt,
        max_tokens=200,
        stop=["<end_of_turn>"],
        temperature=0.7,
        top_p=0.95,
    )
    return output["choices"][0]["text"].strip()


def speech_to_text(audio_data: bytes) -> str:
    """Transcribe WebM audio using Whisper after conversion to WAV."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_in:
        temp_in.write(audio_data)
        temp_in.flush()

    temp_out_path = temp_in.name.replace(".webm", ".wav")

    # Convert to WAV using ffmpeg
    subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_in.name,
        "-ar", "16000",  # downsample to 16kHz if needed
        "-ac", "1",      # mono channel
        temp_out_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Transcribe
    result = whisper_model.transcribe(temp_out_path)

    os.remove(temp_in.name)
    os.remove(temp_out_path)

    return result["text"]

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


def list_voices() -> list:
    """Scan the voices/ directory and return available voice IDs."""
    voices = []
    voice_dir = "./voices"
    if os.path.isdir(voice_dir):
        for f in os.listdir(voice_dir):
            if f.endswith(".pt"):
                voice_id = os.path.splitext(f)[0]
                voices.append(voice_id)
    return sorted(voices)
