"""
Configuration settings for the Multimodal Voice Assistant
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
TEMP_DIR = PROJECT_ROOT / "temp"

# Create directories if they don't exist
for directory in [MODELS_DIR, LOGS_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# Model configurations
MODEL_CONFIG = {
    "llava_model_id": "llava-hf/llava-1.5-7b-hf",
    "whisper_model_size": "medium",
    "max_new_tokens": 200,
    "device": "cuda" if os.environ.get("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
}

# Audio configurations
AUDIO_CONFIG = {
    "sample_rate": 44100,
    "language": "en",
    "temp_audio_file": TEMP_DIR / "temp_audio.mp3"
}

# UI configurations
UI_CONFIG = {
    "page_title": "Multimodal Voice Assistant",
    "page_icon": "🎤",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Logging configuration
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": LOGS_DIR / "app.log"
}
