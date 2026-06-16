from pathlib import Path
import soundfile as sf
from kokoro_onnx import Kokoro

MODEL = "models/kokoro-v1.0.onnx"
VOICES = "models/voices-v1.0.bin"

print("[TTS] Loading TTS...")
kokoro = Kokoro(model_path=MODEL, voices_path=VOICES)

def create_wav(text: str, output: Path, voice: str = "af_heart", speed: float = 1.0):
    print("[TTS] Creating Speech Wav...")
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
    sf.write(str(output), samples, samplerate=sample_rate)
    print(f"[TTS] Saved Audio File")