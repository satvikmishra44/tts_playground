import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

KEY = os.getenv("GEMINI_API_KEY")
if not KEY:
    raise ValueError("Gemini API Key Not Found")

client = genai.Client(KEY)

MODEL = "gemini-3.1-flash-lite"

def generate(query: str, history: list[str] | None = None) -> str:
    history = history or []

    conversation_parts = [
        "You are a helpful, concise voice assistant.",
        "Keep answers natural, short, and easy to speak aloud."
    ]

    for i, turn in enumerate(history):
        role = "User" if i % 2 == 0 else "Assistant"
        conversation_parts.append(f"{role}: {turn}")

    conversation_parts.append(f"User: {query}")
    conversation_parts.append("Assistant:")
    prompt = "\n".join(conversation_parts)

    print("[LLM] Sending Prompt...")
    response = client.models.generate_content(model=MODEL, contents=prompt)

    text = response.text.strip() if response.text else "Sorry, I couldn't generate a reply"

    print(f"[LLM] LLM Replied: {text}")