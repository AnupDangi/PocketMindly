from faster_whisper import WhisperModel
import os
import time

# Configuration
MODEL_SIZE = "tiny"
# Use the local model path we downloaded to
MODEL_PATH = "./models/whisper-tiny"
COMPUTE_TYPE = "int8"
DEVICE = "cpu"

class PocketSTT:
    def __init__(self):
        print(f"Loading Whisper model '{MODEL_SIZE}'...")
        start_time = time.time()
        # Ensure path exists, otherwise default to downloading/cache
        if os.path.exists(MODEL_PATH):
            download_root = MODEL_PATH
        else:
            download_root = None # Let it download if missing
            
        self.model = WhisperModel(
            MODEL_SIZE, 
            device=DEVICE, 
            compute_type=COMPUTE_TYPE,
            download_root=download_root
        )
        print(f"STT Model loaded in {time.time() - start_time:.2f}s")

    def transcribe(self, audio_file):
        """
        Transcribes the given audio file.
        Returns the text string.
        """
        if not os.path.exists(audio_file):
            print(f"Error: Audio file {audio_file} not found.")
            return ""

        print("Transcribing...")
        start_time = time.time()
        
        segments, info = self.model.transcribe(audio_file, beam_size=5, language="en")
        
        full_text = ""
        for segment in segments:
            full_text += segment.text + " "
            
        latency = time.time() - start_time
        print(f"STT Latency: {latency:.2f}s")
        
        return full_text.strip()

if __name__ == "__main__":
    stt = PocketSTT()
    # Test with a dummy file if it exists, or one recorded by audio.py
    test_file = "test.wav"
    if os.path.exists(test_file):
        print(f"Test Result: {stt.transcribe(test_file)}")
