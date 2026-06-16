from pathlib import Path
import whisper

MODEL_NAME = "base"

print(f"[ASR] Loading Whisper {MODEL_NAME}")
_model = whisper.load_model(MODEL_NAME).to("cuda")

def transcribe(audio: Path) -> str:
    print(f"[ASR] Transcribing {audio}...")
    result = _model.transcribe(str(audio))
    text = result["text"].strip()
    print(f"Transcribed Text: {text}")
    return text
