# 🎤 Multimodal Voice Assistant

A powerful Streamlit application that combines image analysis and voice interaction using state-of-the-art AI models. Upload an image, ask questions via voice, and get intelligent spoken responses!

## ✨ Features

- **🖼️ Image Analysis**: Powered by LLaVA (Large Language and Vision Assistant) for detailed image understanding
- **🎤 Speech-to-Text**: Whisper model for accurate speech recognition in multiple languages
- **🔊 Text-to-Speech**: Natural-sounding voice responses using Google Text-to-Speech
- **🤖 Multimodal AI**: Combines visual and audio understanding for intelligent interactions
- **📱 Modern UI**: Clean, responsive Streamlit interface
- **⚡ Modular Design**: Well-organized codebase ready for GitHub deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for better performance)
- FFmpeg (for audio processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd multimodal-voice-assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
multimodal-voice-assistant/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
├── config/
│   └── settings.py               # Configuration settings
├── components/
│   ├── image_processor.py        # Image processing with LLaVA
│   ├── audio_processor.py        # Audio processing with Whisper
│   └── multimodal_processor.py   # Combined multimodal processing
├── utils/
│   └── logger.py                 # Logging utilities
├── models/                       # Model cache directory
├── logs/                         # Application logs
└── temp/                         # Temporary files
```

## 🛠️ Usage

1. **Upload an Image**: Choose any image file (JPG, PNG, etc.)
2. **Record or Upload Audio**: Ask questions about the image via voice
3. **Get AI Response**: The system will analyze both inputs and provide a spoken response

### Example Interactions

- "What colors are predominant in this image?"
- "Describe what you see in this photograph"
- "Is this a painting or a photograph?"
- "What objects can you identify in this scene?"

## ⚙️ Configuration

Edit `config/settings.py` to customize:

- Model configurations (LLaVA model, Whisper size)
- Audio settings (sample rate, language)
- UI preferences
- Logging options

## 🔧 Technical Details

### Models Used

- **LLaVA-1.5-7B**: For image understanding and text generation
- **Whisper Medium**: For speech-to-text conversion
- **Google TTS**: For text-to-speech synthesis

### Performance Notes

- First run will download models (~1.5GB for Whisper, ~13GB for LLaVA)
- GPU acceleration recommended for faster processing
- Models are cached locally for subsequent runs

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
1. **Streamlit Cloud**: Connect your GitHub repo to Streamlit Cloud
2. **Docker**: Use the provided Dockerfile for containerized deployment
3. **Heroku**: Deploy using the included Procfile
4. **AWS/GCP/Azure**: Deploy using cloud-specific configurations

### GitHub Setup
1. Initialize git repository
2. Add all files: `git add .`
3. Commit: `git commit -m "Initial commit"`
4. Push to GitHub: `git push origin main`

## 📝 API Reference

### ImageProcessor
- `process_image(image_path, input_text)`: Process image with optional text prompt
- `is_image_file(file_path)`: Validate image file

### AudioProcessor
- `speech_to_text(audio_path)`: Convert speech to text
- `text_to_speech(text, output_path)`: Convert text to speech
- `is_audio_file(file_path)`: Validate audio file

### MultimodalProcessor
- `process_multimodal_input(audio_path, image_path)`: Process both inputs
- `process_image_only(image_path, prompt)`: Process image only
- `process_audio_only(audio_path)`: Process audio only

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce model size or use CPU
2. **Model download fails**: Check internet connection and Hugging Face access
3. **Audio not playing**: Ensure FFmpeg is installed
4. **Import errors**: Verify all dependencies are installed

### Logs

Check the `logs/` directory for detailed application logs and error messages.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Commit: `git commit -m "Add feature"`
5. Push: `git push origin feature-name`
6. Create a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA) for vision-language understanding
- [Whisper](https://github.com/openai/whisper) for speech recognition
- [Streamlit](https://streamlit.io/) for the web framework
- [Hugging Face](https://huggingface.co/) for model hosting

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/ruhul-cse-duet/Multimodal-Voice-Assistant-llava/issues) page
2. Create a new issue with detailed description
3. Include logs and system information

---

**Happy coding! 🚀 Upload this to GitHub and share your amazing multimodal AI assistant with the world!**
# Multimodal-Voice-Assistant-llava
