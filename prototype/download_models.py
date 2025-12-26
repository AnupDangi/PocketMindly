from huggingface_hub import hf_hub_download
import os

MODEL_REPO = "MaziyarPanahi/gemma-2b-it-GGUF"
MODEL_FILE = "gemma-2b-it.Q4_K_M.gguf"
LOCAL_DIR = "./models"


if not os.path.exists(LOCAL_DIR):
    os.makedirs(LOCAL_DIR)

print(f"Downloading {MODEL_FILE} from {MODEL_REPO}...")
file_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False

)
print(f"Model downloaded to: {file_path}")

from faster_whisper import WhisperModel

print("Downloading Whisper tiny model...")
model = WhisperModel("tiny", device="cpu", compute_type="int8", download_root="./models/whisper-tiny")
print("Whisper model downloaded.")

