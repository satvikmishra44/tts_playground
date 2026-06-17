import time
import tempfile
import numpy as np

import sounddevice as sd
import soundfile as sf
import torch

from silero_vad import load_silero_vad, get_speech_timestamps
from faster_whisper import WhisperModel

# Config
SAMPLE_RATE = 16000
CHUNK_DURATION = 2  # seconds

# ==========================================
# LOAD MODELS
# ==========================================

print("[INFO] Loading Silero VAD...")

vad_model = load_silero_vad()

print("[INFO] Loading Whisper...")

model = WhisperModel(
    "base",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16"
    if torch.cuda.is_available()
    else "int8"
)

print("[INFO] Models Loaded")


# ==========================================
# RECORD AUDIO
# ==========================================

def record_audio(duration):

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    return audio.flatten()


# ==========================================
# VAD
# ==========================================

def contains_speech(audio):

    audio_tensor = torch.tensor(audio)

    speech_segments = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=SAMPLE_RATE
    )

    return len(speech_segments) > 0


# ==========================================
# TRANSCRIPTION
# ==========================================

def transcribe_audio(audio):

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as f:

        wav_path = f.name

    sf.write(
        wav_path,
        audio,
        SAMPLE_RATE
    )

    segments, info = model.transcribe(
        wav_path,
        beam_size=1
    )

    text = ""

    for segment in segments:
        text += segment.text.strip() + " "

    return text.strip()


# ==========================================
# MAIN LOOP
# ==========================================

print("\nListening...\n")

while True:

    total_start = time.perf_counter()

    # -------------------------
    # RECORDING LATENCY
    # -------------------------

    capture_start = time.perf_counter()

    audio = record_audio(
        CHUNK_DURATION
    )

    capture_ms = (
        time.perf_counter()
        - capture_start
    ) * 1000

    # -------------------------
    # VAD LATENCY
    # -------------------------

    vad_start = time.perf_counter()

    speech_detected = contains_speech(
        audio
    )

    vad_ms = (
        time.perf_counter()
        - vad_start
    ) * 1000

    if not speech_detected:
        print("[VAD] Silence")
        continue

    print("\n[VAD] Speech Detected")

    # -------------------------
    # ASR LATENCY
    # -------------------------

    asr_start = time.perf_counter()

    transcript = transcribe_audio(
        audio
    )

    asr_ms = (
        time.perf_counter()
        - asr_start
    ) * 1000

    total_ms = (
        time.perf_counter()
        - total_start
    ) * 1000

    # -------------------------
    # OUTPUT
    # -------------------------

    print("\n[TRANSCRIPT]")
    print(transcript)

    print("\n[LATENCY]")
    print(
        f"Capture : {capture_ms:.2f} ms"
    )

    print(
        f"VAD     : {vad_ms:.2f} ms"
    )

    print(
        f"ASR     : {asr_ms:.2f} ms"
    )

    print(
        f"Total   : {total_ms:.2f} ms"
    )

    print("\n" + "=" * 60 + "\n")

