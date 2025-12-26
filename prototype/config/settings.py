import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Audio Settings
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 1
AUDIO_DTYPE = 'int16'

# STT Settings
STT_MODEL_SIZE = "base.en"
# Use absolute path or relative from execution context carefully. 
# Here we point to the models dir we just defined.
STT_MODEL_PATH = os.path.join(MODELS_DIR, "whisper-base.en")
STT_COMPUTE_TYPE = "int8"
STT_DEVICE = "cpu"

# LLM Settings
LLM_MODEL_FILENAME = "gemma-2b-it.Q4_K_M.gguf"
LLM_MODEL_PATH = os.path.join(MODELS_DIR, LLM_MODEL_FILENAME)
LLM_CONTEXT_WINDOW = 2048
