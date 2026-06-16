from pathlib import Path
from audio_io import record, play_wav
from asr import transcribe
from llm import generate
from tts import create_wav

INPUT = Path("input.wav")
OUTPUT = Path("output.wav")

def one_turn(history: list[str] | None = None):
    # Recording User Speech
    record(5.0, INPUT)
    
    # Transcribing Locally
    query = transcribe(INPUT)

    if not query:
        print("[System] No speech detected or transcript empty.")
        return "", "I didn't catch that."
    
    # Generating LLM Response
    response = generate(query=query, history=history)

    # Generating Audio of response from Kokoro
    create_wav(text=response, output=OUTPUT)

    # Playing Response
    play_wav(OUTPUT)

    return query, response


if __name__ == "__main__":
    history: list[str] = []

    print("Voice agent started.")
    print("Press Enter to speak, or type q to quit.")

    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break

        user_text, reply_text = one_turn(history)

        if user_text:
            history.append(user_text)
            history.append(reply_text)