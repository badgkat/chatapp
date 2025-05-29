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
from context_state import ChatState

CHAT_STATES: dict[str, ChatState] = {}     # keyed by session ID

# Default System Prompt
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


# Load Whisper and Kokoro models once
whisper_model = whisper.load_model("tiny.en")
tts_pipeline = KPipeline(lang_code='a')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEFAULT_MODEL_PATH = "./models/gemma-3-4b-it-Q4_K_M.gguf"
llm = None

def load_model():
    global llm
    if llm is not None:
        return llm

    # read config
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
            model_path = cfg.get("model_path", DEFAULT_MODEL_PATH)
            p = cfg.get("params", {})
    else:
        model_path = DEFAULT_MODEL_PATH
        p = {}

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Visit http://localhost:8000/setup to download one."
        )

    llm = Llama(
        model_path=model_path,
        n_ctx=p.get("n_ctx", 2048),
        n_threads=p.get("n_threads", 12),
        n_batch=p.get("n_batch", 64),
        n_gpu_layers=p.get("n_gpu_layers", 0),
    )
    return llm

def count_tokens(txt: str) -> int:
    llm = load_model()                     # cached
    # llama_cpp expects bytes and returns a list[int]
    return len(llm.tokenize(txt.encode()))

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

def build_prompt(state: ChatState,
                 user_in: str,
                 sys_prompt: str,
                 budget: int) -> str:
    state.window.append(("user", user_in))

    parts = [f"<sys>{sys_prompt}</sys>"]
    if state.summary:
        parts.append(f"<summary>{state.summary}</summary>")
    parts += [f"<{r}>{t}</{r}>" for r, t in state.window]
    prompt = "".join(parts)

    if count_tokens(prompt) > budget:
        half = len(state.window) // 2
        to_sum = "".join(f"<{r}>{t}</{r}>" for r, t in list(state.window)[:half])

        # quick summary call
        summary_text = load_model()(
            f"{sys_prompt}\nSummarise:\n{to_sum}\nEnd with <END>",
            max_tokens=120,
            stop=["<END>"]
        )["choices"][0]["text"].strip()
        state.summary = summary_text
        for _ in range(half):
            state.window.popleft()

        parts = [f"<sys>{sys_prompt}</sys>",
                 f"<summary>{state.summary}</summary>",
                 *[f"<{r}>{t}</{r}>" for r, t in state.window]]
        prompt = "".join(parts)

    return prompt

def process_message(user_text: str, session_id: str = "default") -> str:
    """
    Run one turn of inference.
    The system prompt, generation params, and model path live in config.json.
    A constraint note tells the model its token budget and asks it to finish with <END>.
    """

    cfg = json.load(open(CONFIG_PATH))
    gen  = cfg.get("params", {})
    sysp = cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    limit = int(gen.get("max_tokens", 200))
    n_ctx = int(gen.get("n_ctx", 2048))
    budget = n_ctx - limit - 32           # small safety margin

    state = CHAT_STATES.setdefault(session_id, ChatState())
    prompt = build_prompt(state, user_text, sysp, budget)

    reply = load_model()(
        prompt +
        "\n### Constraint\n" +
        f"Reply ≤{limit} tokens and finish with <END>",
        max_tokens=limit,
        temperature=gen.get("temperature", 0.7),
        top_p=gen.get("top_p", 0.95),
        stop=["<END>"]
    )["choices"][0]["text"].strip()

    if reply.endswith("<END>"):
        reply = reply[:-5].rstrip()

    state.window.append(("assistant", reply))
    return reply



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
