"""
Audio processing component for the Multimodal Voice Assistant
"""
# Import whisper with a friendly error if the optional dependency is missing.
# The package name on PyPI is `openai-whisper` even though it is imported as `whisper`.
try:
    import whisper
    _WHISPER_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - defensive import guard
    whisper = None
    _WHISPER_IMPORT_ERROR = exc
import torch
import numpy as np
from gtts import gTTS
from pathlib import Path
from config.settings import MODEL_CONFIG, AUDIO_CONFIG
from utils.logger import setup_logger, write_history

logger = setup_logger(__name__)

class AudioProcessor:
    """Handles audio processing including speech-to-text and text-to-speech"""
    
    def __init__(self):
        """Initialize the audio processor with Whisper model"""
        if whisper is None:
            raise ImportError(
                "The `whisper` module is not installed. "
                "Install it with `pip install openai-whisper` or `pip install -r requirements.txt` "
                "before running the app."
            ) from _WHISPER_IMPORT_ERROR
        
        # Check CUDA availability and use GPU if available
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device = "cuda"
            logger.info(f"Whisper will use GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = MODEL_CONFIG["device"]  # Fallback to config
            logger.info("Whisper will use CPU")
        
        self.whisper_model_size = MODEL_CONFIG["whisper_model_size"]
        self.language = AUDIO_CONFIG["language"]
        self.temp_audio_file = AUDIO_CONFIG["temp_audio_file"]
        
        # Initialize Whisper model
        self._setup_whisper()
    
    def _setup_whisper(self):
        """Setup the Whisper model for speech recognition"""
        try:
            self.whisper_model = whisper.load_model(self.whisper_model_size, device=self.device)
            logger.info(f"✅ Whisper model loaded: {self.whisper_model_size} on {self.device.upper()}")
            
            # Verify device placement
            if self.cuda_available:
                try:
                    # Check if model is on GPU
                    model_device = next(self.whisper_model.encoder.parameters()).device
                    if model_device.type == "cuda":
                        logger.info(f"✅ Whisper model confirmed on GPU: {model_device}")
                    else:
                        logger.warning(f"⚠️ Whisper model on {model_device}, expected GPU")
                except:
                    pass
            
            logger.info(
                f"Model is {'multilingual' if self.whisper_model.is_multilingual else 'English-only'} "
                f"and has {sum(np.prod(p.shape) for p in self.whisper_model.parameters()):,} parameters."
            )
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            # Try CPU fallback if GPU fails
            if self.device == "cuda":
                logger.warning("GPU initialization failed, trying CPU...")
                try:
                    self.device = "cpu"
                    self.whisper_model = whisper.load_model(self.whisper_model_size, device=self.device)
                    logger.info(f"Whisper model loaded on CPU (fallback)")
                except Exception as e2:
                    logger.error(f"CPU fallback also failed: {e2}")
                    raise
            else:
                raise
    
    def speech_to_text(self, audio_path: str) -> str:
        """
        Convert speech to text using Whisper
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            if not audio_path or audio_path == '':
                logger.warning("No audio file provided")
                return ""
            
            write_history(f"Processing audio: {audio_path}")
            
            # Load and preprocess audio
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Create mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
            
            # Detect language
            _, probs = self.whisper_model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            logger.info(f"Detected language: {detected_language}")
            
            # Decode audio
            options = whisper.DecodingOptions()
            result = whisper.decode(self.whisper_model, mel, options)
            transcribed_text = result.text.strip()
            
            write_history(f"Transcribed text: {transcribed_text}")
            logger.info("Speech-to-text conversion completed successfully")
            
            return transcribed_text
            
        except Exception as e:
            error_msg = f"Error in speech-to-text conversion: {e}"
            logger.error(error_msg)
            write_history(f"ERROR: {error_msg}")
            return ""
    
    def text_to_speech(self, text: str, output_path: str = None) -> str:
        """
        Convert text to speech using gTTS
        
        Args:
            text: Text to convert to speech
            output_path: Optional output path for the audio file
            
        Returns:
            Path to the generated audio file
        """
        try:
            if not text or text.strip() == "":
                logger.warning("No text provided for text-to-speech")
                return ""
            
            # Set default output path if not provided
            if output_path is None:
                output_path = str(self.temp_audio_file)
            
            write_history(f"Converting text to speech: {text[:100]}...")
            
            # Create gTTS object
            tts = gTTS(text=text, lang=self.language, slow=False)
            
            # Save audio file
            tts.save(output_path)
            
            write_history(f"Audio saved to: {output_path}")
            logger.info("Text-to-speech conversion completed successfully")
            
            return output_path
            
        except Exception as e:
            error_msg = f"Error in text-to-speech conversion: {e}"
            logger.error(error_msg)
            write_history(f"ERROR: {error_msg}")
            return ""
    
    def is_audio_file(self, file_path: str) -> bool:
        """
        Check if the file is a valid audio file
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is a valid audio file, False otherwise
        """
        try:
            # Try to load the audio file with whisper
            whisper.load_audio(file_path)
            return True
        except Exception:
            return False
