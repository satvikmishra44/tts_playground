from pathlib import Path
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

SAMPLE_RATE = 16000
CHANNELS = 1

def record(seconds: float, output: Path):
    print(f"[Audio] Recording for {seconds:.1f} seconds...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()

    audio = np.squeeze(audio)
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    wav.write(str(output), SAMPLE_RATE, audio_int16)
    print(f"[Audio] Saved recording to {output}")

def play_wav(path: Path):
    sr, data = wav.read(str(path))
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    data = data.astype("float32") / 32767.0
    print(f"[Audio] Playing {path}...")
    sd.play(data, sr)
    sd.wait()