from faster_whisper import WhisperModel
import os
import time
from config import settings

# Configuration
MODEL_SIZE = settings.STT_MODEL_SIZE
# Use the local model path we downloaded to
MODEL_PATH = settings.STT_MODEL_PATH
COMPUTE_TYPE = settings.STT_COMPUTE_TYPE
DEVICE = settings.STT_DEVICE

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
        
        segments, info = self.model.transcribe(
            audio_file, 
            beam_size=5, 
            language="en", 
            condition_on_previous_text=False
        )
        
        full_text = ""
        for segment in segments:
            full_text += segment.text + " "
            
        full_text = full_text.strip()
        
        # Filter common Whisper hallucinations
        hallucinations = ["You", "Thank you.", "Thank you", "MBC", "You."]
        if full_text in hallucinations:
            return ""
            
        
        return full_text

    def transcribe_stream(self, audio_buffer, sample_rate=16000):
        """
        Transcribe audio buffer for streaming (real-time partial transcripts).
        Returns dict with 'text' and 'is_final'.
        """
        import numpy as np
        import scipy.io.wavfile as wav
        import tempfile
        
        # Save buffer to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            
        # Convert float32 to int16
        if audio_buffer.dtype == np.float32:
            audio_int16 = (audio_buffer * 32767).astype(np.int16)
        else:
            audio_int16 = audio_buffer
            
        wav.write(temp_path, sample_rate, audio_int16)
        
        try:
            # Quick transcription for partial results
            segments, info = self.model.transcribe(
                temp_path,
                beam_size=1,  # Faster for partial
                language="en",
                condition_on_previous_text=False,
                vad_filter=False  # We handle VAD externally
            )
            
            text = ""
            for segment in segments:
                text += segment.text + " "
            
            text = text.strip()
            
            # Filter hallucinations
            hallucinations = ["You", "Thank you.", "Thank you", "MBC", "You."]
            if text in hallucinations:
                text = ""
            
            return {
                'text': text,
                'is_final': False
            }
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

if __name__ == "__main__":
    stt = PocketSTT()
    # Test with a dummy file if it exists, or one recorded by audio.py
    test_file = "test.wav"
    if os.path.exists(test_file):
        print(f"Test Result: {stt.transcribe(test_file)}")

