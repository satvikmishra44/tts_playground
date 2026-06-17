import asyncio
import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro

# Initialize Kokoro ONNX (Ensure CUDA provider is active)
kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")

async def play_audio_async(text, voice="af_bella", speed=1.0):
    # 1. Generate the audio directly into memory as a NumPy array
    # This completely skips writing a .wav file to disk
    samples, sample_rate = kokoro.create(
        text, 
        voice=voice, 
        speed=speed, 
        lang="en-us"
    )
    
    # 2. Convert to float32 (standard for sounddevice streaming)
    audio_data = np.array(samples, dtype=np.float32)
    
    # 3. Fire-and-forget playback on a background audio thread
    # The script continues immediately while audio plays
    sd.play(audio_data, sample_rate)
    
    print("🔊 Audio playback started in background...")

async def main():
    text_prompt = "Wait... hold on... I need to catch my breath. That was... much harder than I expected."
    
    # Trigger the asynchronous playback task
    await play_audio_async(text_prompt)
    
    # Prove the main thread is unblocked by performing another action immediately
    for i in range(5):
        print(f"🤖 Main thread is free! Doing other work... {i}")
        await asyncio.sleep(1)
        
    # Wait for the background audio buffer to finish playing before exiting the script
    sd.wait() 

if __name__ == "__main__":
    asyncio.run(main())
