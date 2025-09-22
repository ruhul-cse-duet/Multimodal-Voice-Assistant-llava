"""
Image processing component for the Multimodal Voice Assistant
"""
import re
import torch
from PIL import Image
from transformers import BitsAndBytesConfig, pipeline
from config.settings import MODEL_CONFIG
from utils.logger import setup_logger, write_history

logger = setup_logger(__name__)

class ImageProcessor:
    """Handles image processing and text generation from images"""
    
    def __init__(self):
        """Initialize the image processor with LLaVA model"""
        self.model_id = MODEL_CONFIG["llava_model_id"]
        self.max_new_tokens = MODEL_CONFIG["max_new_tokens"]
        # Prefer actual runtime CUDA availability over config env
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_quantization = self.device == "cuda"
        
        # Setup quantization config (GPU only)
        if self.use_quantization:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            self.quantization_config = None
        
        # Initialize pipeline
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup the image-to-text pipeline"""
        try:
            pipe_kwargs = {
                "task": "image-to-text",
                "model": self.model_id,
            }
            if self.use_quantization and self.quantization_config is not None:
                pipe_kwargs["model_kwargs"] = {"quantization_config": self.quantization_config}
                pipe_kwargs["device_map"] = "auto"
                logger.info("Initializing LLaVA with 4-bit quantization on GPU")
            else:
                # CPU fallback: no quantization
                pipe_kwargs["device_map"] = "cpu"
                logger.warning("CUDA not available. Loading LLaVA on CPU without quantization. This will be slow.")

            self.pipe = pipeline(**pipe_kwargs)
            logger.info(f"Image processor initialized with model: {self.model_id} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize image processor: {e}")
            raise
    
    def process_image(self, image_path: str, input_text: str = None) -> str:
        """
        Process image and generate text response
        
        Args:
            image_path: Path to the input image
            input_text: Optional text prompt for the image
            
        Returns:
            Generated text response
        """
        try:
            # Load the image
            image = Image.open(image_path)
            
            # Log input
            write_history(f"Processing image: {image_path}")
            write_history(f"Input text: {input_text}")
            
            # Determine prompt instructions
            if input_text is None or input_text.strip() == "":
                prompt_instructions = """
                Describe the image using as much detail as possible,
                is it a painting, a photograph, what colors are predominant,
                what is the image about?
                """
            else:
                prompt_instructions = f"""
                Act as an expert in imagery descriptive analysis, using as much detail as possible from the image, respond to the following prompt:
                {input_text}
                """
            
            # Create prompt
            prompt = "USER: <image>\n" + prompt_instructions + "\nASSISTANT:"
            
            # Generate response
            outputs = self.pipe(
                image, 
                prompt=prompt, 
                generate_kwargs={"max_new_tokens": self.max_new_tokens}
            )
            
            # Extract response text
            response = self._extract_response(outputs[0]["generated_text"])
            
            write_history(f"Generated response: {response}")
            logger.info("Image processed successfully")
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing image: {e}"
            logger.error(error_msg)
            write_history(f"ERROR: {error_msg}")
            return "Error processing image. Please try again."
    
    def _extract_response(self, generated_text: str) -> str:
        """
        Extract the assistant's response from the generated text
        
        Args:
            generated_text: Full generated text from the model
            
        Returns:
            Extracted response text
        """
        if not generated_text:
            return "No response generated."
        
        # Try to extract text after "ASSISTANT:"
        match = re.search(r'ASSISTANT:\s*(.*)', generated_text)
        if match:
            return match.group(1).strip()
        else:
            # Fallback: return the full text if no ASSISTANT marker found
            return generated_text.strip()
    
    def is_image_file(self, file_path: str) -> bool:
        """
        Check if the file is a valid image
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is a valid image, False otherwise
        """
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False
