import onnxruntime
import numpy as np
import os
from config import settings

class SileroVAD:
    def __init__(self, model_path=None):
        if model_path is None:
            # Default to the downloaded path
            model_path = os.path.join("models", "onnx", "model.onnx")
            
        print(f"Loading VAD model from {model_path}...")
        
        # Initialize ONNX Runtime
        opts = onnxruntime.SessionOptions()
        opts.log_severity_level = 3
        
        try:
            self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            print("ONNX Inputs:", [i.name for i in self.session.get_inputs()])
        except Exception as e:
            print(f"Error loading VAD ONNX: {e}")
            self.session = None
            return

        # VAD State (h, c) - must be maintained between chunks for stream
        self.reset_states()
        
        # Audio params
        self.sr = 16000

    def reset_states(self):
        """Resets the internal hidden states of the RNN."""
        # ONNX model uses a single 'state' tensor of shape [2, 1, 128]
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, audio_chunk, threshold=0.5):
        """
        Returns probability of speech in the chunk.
        audio_chunk: numpy array of float32, usually 512 samples at 16k.
        """
        if self.session is None:
            return 0.0

        # Prepare Inputs
        # Audio: [1, N]
        if audio_chunk.ndim == 1:
            audio_chunk = audio_chunk[np.newaxis, :]
            
        # SR: scalar tensor (0-D)
        sr_tensor = np.array(self.sr, dtype=np.int64)
        
        # Run Inference
        ort_inputs = {
            'input': audio_chunk,
            'state': self._state,
            'sr': sr_tensor
        }
        
        try:
            out, state_out = self.session.run(None, ort_inputs)
            
            # Update state
            self._state = state_out
            
            # Output is probability [1, 1]
            prob = out[0][0]
            return prob
            
        except Exception as e:
            print(f"VAD Error: {e}")
            return 0.0

    def validate_chunk_size(self, size):
        """Silero allows 512, 1024, 1536 samples for 16k."""
        return size in [512, 1024, 1536]
    
    def process_frame(self, audio_frame: np.ndarray, threshold=0.5) -> dict:
        """
        Process a single audio frame for streaming VAD.
        Returns dict with:
            - 'probability': speech probability
            - 'is_speech': boolean
            - 'event': 'speech_start', 'speech_end', or None
        """
        prob = self.is_speech(audio_frame, threshold)
        is_speech_now = prob > threshold
        
        # Detect events
        event = None
        if is_speech_now and not self._was_speech:
            event = 'speech_start'
        elif not is_speech_now and self._was_speech:
            event = 'speech_end'
        
        self._was_speech = is_speech_now
        
        return {
            'probability': prob,
            'is_speech': is_speech_now,
            'event': event
        }
    
    def reset_for_new_utterance(self):
        """Reset state for a new utterance"""
        self.reset_states()
        self._was_speech = False

# Track speech state for event detection
SileroVAD._was_speech = False
