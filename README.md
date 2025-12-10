# 🎤 Multimodal Voice Assistant

Streamlit app that combines LLaVA for image understanding, Whisper for speech-to-text, and gTTS for text-to-speech. Upload an image, speak a question, and get a spoken AI response.

## Requirements
- Python 3.12 (tested on 3.12.12)
- FFmpeg available on `PATH` (required by Whisper/pydub)
- (Optional) CUDA-capable GPU for faster inference

## Quick Start (Windows / Python 3.12)
1) Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

2) Install dependencies (CPU default)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) (Optional, GPU + quantization) Install a CUDA build of PyTorch and bitsandbytes manually, e.g.:
```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes
```

4) Download required NLTK data
```bash
python -c "import nltk; nltk.download('punkt')"
```

5) Run the app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

## Project Structure
```
multimodal-voice-assistant/
├── app.py                      # Streamlit entry point
├── requirements.txt            # Python dependencies
├── config/
│   └── settings.py             # Model + UI configuration
├── components/
│   ├── image_processor.py      # LLaVA image-to-text pipeline
│   ├── audio_processor.py      # Whisper STT + gTTS TTS
│   └── multimodal_processor.py # Orchestration of image/audio flows
├── utils/
│   └── logger.py               # Logging utilities + history
├── models/                     # Model cache (created at runtime)
├── logs/                       # Application logs
└── temp/                       # Temporary files
```

## Configuration
`config/settings.py`
- `MODEL_CONFIG`: LLaVA model id, Whisper size, device preference
- `AUDIO_CONFIG`: language, sample rate, temp file names
- `UI_CONFIG`: Streamlit page settings

## How It Works
1. User uploads an image and records/uploads audio in the UI.
2. `AudioProcessor` runs Whisper to transcribe speech.
3. `ImageProcessor` runs LLaVA, optionally using the transcript as context.
4. Response is converted to speech with gTTS and played back.

## Notes & Tips
- First run will download models (can be several GB). Cached under `models/`.
- On CPU, image generation with LLaVA will be slow; consider smaller models if needed.
- If Whisper is missing you will see a clear error; install via `pip install openai-whisper` (already in requirements).
- Ensure FFmpeg is installed and on PATH for audio processing.

## Troubleshooting
- Install errors on Windows for bitsandbytes: omit it (CPU path) or install a wheel compatible with your CUDA setup.
- CUDA out of memory: use CPU (`CUDA_AVAILABLE=false`) or a smaller model.
- Audio not saved/played: verify FFmpeg install and file permissions under `temp/`.

## 📧 Contact
[Md Ruhul Amin](https://www.linkedin.com/in/ruhul-duet-cse/); \
Email: ruhul.cse.duet@gmail.com

For questions or issues, please open an issue on GitHub.

---

The first run downloads embedding weights; keep an internet connection for that step.

## License
MIT
