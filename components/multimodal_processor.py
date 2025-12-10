"""
Multimodal processing component that combines image and audio processing
"""
from typing import Tuple, Optional
from pathlib import Path
from components.image_processor import ImageProcessor
from components.audio_processor import AudioProcessor
from utils.logger import setup_logger, write_history

logger = setup_logger(__name__)

class MultimodalProcessor:
    """Handles multimodal processing combining image and audio inputs"""
    
    def __init__(self):
        """Initialize the multimodal processor"""
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        logger.info("Multimodal processor initialized")
    
    def process_multimodal_input(
        self, 
        audio_path: Optional[str] = None, 
        image_path: Optional[str] = None,
        text_input: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """
        Process audio, image, and/or text inputs
        
        Args:
            audio_path: Path to audio file (optional)
            image_path: Path to image file (optional)
            text_input: Text prompt/question (optional)
            
        Returns:
            Tuple of (transcribed_text, generated_response, audio_output_path)
        """
        try:
            transcribed_text = ""
            input_prompt = ""
            generated_response = ""
            audio_output_path = ""
            
            # Process audio if provided
            if audio_path and Path(audio_path).exists():
                transcribed_text = self.audio_processor.speech_to_text(audio_path)
                write_history(f"Audio processed: {transcribed_text}")
                input_prompt = transcribed_text
            
            # Use text input if provided (takes precedence over transcribed audio)
            if text_input and text_input.strip():
                input_prompt = text_input.strip()
                write_history(f"Text input provided: {input_prompt}")
            
            # Process image if provided
            if image_path and Path(image_path).exists():
                # Use input_prompt (from text or audio) as prompt for image analysis
                prompt = input_prompt if input_prompt else None
                generated_response = self.image_processor.process_image(image_path, prompt)
                write_history(f"Image processed with response: {generated_response}")
            else:
                logger.info("No image file provided or file doesn't exist")
                if input_prompt:
                    # If we have text/audio but no image, provide a simple response
                    generated_response = f"I understand: {input_prompt}. However, I need an image to provide a detailed analysis."
                elif transcribed_text:
                    # Fallback for audio only
                    generated_response = f"I heard you say: {transcribed_text}"
            
            # Generate audio response if we have text to speak
            if generated_response:
                audio_output_path = self.audio_processor.text_to_speech(generated_response)
                write_history(f"Audio response generated: {audio_output_path}")
            
            logger.info("Multimodal processing completed successfully")
            return transcribed_text, generated_response, audio_output_path
            
        except Exception as e:
            error_msg = f"Error in multimodal processing: {e}"
            logger.error(error_msg)
            write_history(f"ERROR: {error_msg}")
            return "", f"Error: {error_msg}", ""
    
    def process_image_only(self, image_path: str, prompt: str = None) -> str:
        """
        Process only image input
        
        Args:
            image_path: Path to image file
            prompt: Optional text prompt
            
        Returns:
            Generated response
        """
        try:
            if not Path(image_path).exists():
                return "Image file not found."
            
            response = self.image_processor.process_image(image_path, prompt)
            return response
            
        except Exception as e:
            error_msg = f"Error processing image: {e}"
            logger.error(error_msg)
            return error_msg
    
    def process_audio_only(self, audio_path: str) -> Tuple[str, str]:
        """
        Process only audio input
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (transcribed_text, audio_response_path)
        """
        try:
            if not Path(audio_path).exists():
                return "Audio file not found.", ""
            
            transcribed_text = self.audio_processor.speech_to_text(audio_path)
            
            if transcribed_text:
                # Generate a simple response
                response = f"I heard you say: {transcribed_text}"
                audio_response_path = self.audio_processor.text_to_speech(response)
                return transcribed_text, audio_response_path
            else:
                return "Could not transcribe audio.", ""
                
        except Exception as e:
            error_msg = f"Error processing audio: {e}"
            logger.error(error_msg)
            return error_msg, ""
