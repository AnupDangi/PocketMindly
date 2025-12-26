import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import subprocess
import threading
import queue
import sys

# Audio configuration
SAMPLE_RATE = 44100  # Changed to 44100 for better Mac compatibility
CHANNELS = 1
DTYPE = 'int16'

def record_audio(filename="input.wav"):
    """
    Records audio from the microphone until the user presses Enter.
    Saves the audio to a WAV file.
    """
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print("\n🔴 Press Enter to START recording...")
    input()
    
    print("\n🎙️ Recording... Press Enter to STOP.")
    
    # Start recording in a non-blocking stream
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=callback)
    stream.start()

    # Wait for the user to press Enter to stop
    input()
    
    stream.stop()
    stream.close()
    print("⏹️ Recording stopped.")

    # Collect all data from the queue
    audio_data = []
    while not q.empty():
        audio_data.append(q.get())

    if not audio_data:
        print("Warning: No audio recorded.")
        return False

    # Concatenate and save
    recording = np.concatenate(audio_data, axis=0)
    
    # Normalize Audio (Boost volume)
    max_val = np.max(np.abs(recording))
    if max_val > 0:
        # Target 50% of max volume to avoid clipping but hear clearly
        recording = (recording / max_val) * (32767 * 0.8)
        recording = recording.astype(np.int16)
        
    wav.write(filename, SAMPLE_RATE, recording)
    return True

def speak_text(text):
    """
    Uses the system's TTS (macOS 'say' command) to speak the text.
    """
    try:
        # Using 'say' command on macOS
        # -r 175 sets the rate (optional, can adjust)
        subprocess.run(["say", text], check=True)
    except Exception as e:
        print(f"Error in TTS: {e}")

if __name__ == "__main__":
    # Test the module
    if record_audio("test.wav"):
        speak_text("Recording complete. Playing back.")
        # Optional: play back the recorded audio
        # subprocess.run(["afplay", "test.wav"])
