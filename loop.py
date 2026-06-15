from pathlib import Path
from audio_io import record, play_wav

if __name__ == "__main__":
    path = Path("test.wav")
    record(4.0, path)
    play_wav(path)