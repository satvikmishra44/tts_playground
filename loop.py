from pathlib import Path
from audio_io import record, play_wav
from asr import transcribe

if __name__ == "__main__":
    path = Path("test.wav")
    record(4.0, path)
    transcribe(path)
    play_wav(path)