"""
Configuration settings for the Multimodal Voice Assistant
"""
import os
from pathlib import Path

# Check for CUDA availability
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
TEMP_DIR = PROJECT_ROOT / "temp"

# Create directories if they don't exist
for directory in [MODELS_DIR, LOGS_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# Model configurations
# Use GPU if available (check actual CUDA availability, not just env var)
# For low-memory GPUs (<4GB), consider using smaller models:
# - "llava-hf/llava-1.5-7b-hf" (default, requires ~4GB+ GPU)
# - "llava-hf/llava-1.5-13b-hf" (larger, requires ~8GB+ GPU)
# Note: For 2GB GPUs, the code will automatically fallback to CPU if OOM occurs
MODEL_CONFIG = {
    "llava_model_id": "llava-hf/llava-1.5-7b-hf",
    "whisper_model_size": "base",  # Changed from "medium" to "base" for lower memory usage
    "max_new_tokens": 200,
    "device": "cuda" if CUDA_AVAILABLE else "cpu"
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
