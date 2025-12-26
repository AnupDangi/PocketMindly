import time
import os
from audio import record_audio, speak_text
from stt import PocketSTT
from llm import PocketLLM
import sys

def main():
    print("Initializing PocketMindly Prototype...")
    
    # Initialize implementation modules
    try:
        stt = PocketSTT()
        llm = PocketLLM()
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    print("\n✅ System Ready. Press Ctrl+C to exit.")
    
    while True:
        try:
            # 1. Record Audio
            input_file = "input.wav"
            if not record_audio(input_file):
                continue
            
            # 2. Transcribe (STT)
            user_text = stt.transcribe(input_file)
            print(f"\n🗣️ User: {user_text}")
            
            if not user_text.strip():
                print("No speech detected.")
                continue

            # 3. Generate Response (LLM)
            print("Thinking...")
            start_think = time.time()
            reply = llm.generate_response(user_text)
            think_time = time.time() - start_think
            print(f"🧠 AI: {reply} ({think_time:.2f}s)")
            
            # 4. Speak Response (TTS)
            speak_text(reply)
            
        except KeyboardInterrupt:
            print("\nExiting PocketMindly...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")

if __name__ == "__main__":
    main()
