import sounddevice as sd
import numpy as np

def list_devices():
    print("Available Audio Devices:")
    print(sd.query_devices())
    print(f"\nDefault Input Device: {sd.default.device[0]}")
    print(f"Default Output Device: {sd.default.device[1]}")

def test_recording(fs=44100, duration=3):
    print(f"\nTesting recording at {fs}Hz for {duration} seconds...")
    try:
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        print("Recording successful (in memory).")
        print(f"Max amplitude: {np.max(np.abs(myrecording))}")
        if np.max(np.abs(myrecording)) < 0.01:
            print("⚠️ WARNING: Audio seems silent. Check microphone permissions or input gain.")
        return True
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return False

if __name__ == "__main__":
    try:
        list_devices()
        # Test standard Mac rate
        test_recording(fs=48000)
        # Test Whisper rate
        test_recording(fs=16000)
    except Exception as e:
        print(f"Global Error: {e}")
