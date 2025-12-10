"""
Image processing component for the Multimodal Voice Assistant
"""
import os
import re
import warnings
import torch
from PIL import Image
from transformers import BitsAndBytesConfig, pipeline
from config.settings import MODEL_CONFIG
from utils.logger import setup_logger, write_history

# Set PyTorch CUDA memory management for low-memory GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Suppress deprecation warning for AutoModelForVision2Seq
# This is an internal transformers warning that will be fixed in v5.0
warnings.filterwarnings("ignore", category=FutureWarning, message=".*AutoModelForVision2Seq.*")

# Check if accelerate is available
try:
    import accelerate
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

logger = setup_logger(__name__)

class ImageProcessor:
    """Handles image processing and text generation from images"""
    
    def __init__(self):
        """Initialize the image processor with LLaVA model"""
        self.model_id = MODEL_CONFIG["llava_model_id"]
        self.max_new_tokens = MODEL_CONFIG["max_new_tokens"]
        
        # Check CUDA availability and memory
        self.cuda_available = torch.cuda.is_available()
        self.gpu_memory_gb = 0
        self.use_gpu = False
        
        if self.cuda_available:
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available! GPU: {gpu_name}")
            logger.info(f"GPU Memory: {self.gpu_memory_gb:.2f} GB")
            
            # For GPUs with less than 4GB, use CPU or smaller model
            if self.gpu_memory_gb < 4.0:
                logger.warning(f"⚠️ GPU has only {self.gpu_memory_gb:.2f} GB. This may cause OOM errors.")
                logger.warning("⚠️ Consider using CPU mode or a smaller model for better stability.")
                # Still try GPU but with aggressive memory management
                self.use_gpu = True
                self.device = "cuda"
                self.device_id = 0
                # Clear cache before loading
                torch.cuda.empty_cache()
            else:
                self.use_gpu = True
                self.device = "cuda"
                self.device_id = 0
        else:
            self.device = "cpu"
            self.device_id = -1
            logger.warning("CUDA not available. Using CPU (will be slow).")
        
        # Always use quantization for GPU to save memory
        self.use_quantization = self.use_gpu
        
        # Setup quantization config (GPU only)
        if self.use_quantization:
            try:
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                logger.info("4-bit quantization enabled for GPU")
            except Exception as e:
                logger.warning(f"Could not setup quantization: {e}. Using full precision on GPU.")
                self.quantization_config = None
                self.use_quantization = False
        else:
            self.quantization_config = None
        
        # Initialize pipeline
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup the image-to-text pipeline"""
        try:
            # Clear GPU cache before loading
            if self.use_gpu:
                torch.cuda.empty_cache()
                logger.info("Cleared GPU cache before model loading")
            
            # Use image-text-to-text task (newer API)
            pipe_kwargs = {
                "task": "image-text-to-text",
                "model": self.model_id,
            }
            
            if self.use_gpu:
                # For low-memory GPUs, always use quantization if available
                if self.gpu_memory_gb < 4.0:
                    logger.info(f"Low GPU memory ({self.gpu_memory_gb:.2f} GB) - using aggressive memory optimization")
                    if self.use_quantization and self.quantization_config is not None and ACCELERATE_AVAILABLE:
                        # GPU with quantization (requires accelerate)
                        pipe_kwargs["model_kwargs"] = {"quantization_config": self.quantization_config}
                        pipe_kwargs["device_map"] = "auto"
                        logger.info("Initializing LLaVA with 4-bit quantization on GPU (low-memory mode)")
                    else:
                        # If quantization not available, fallback to CPU for low-memory GPUs
                        logger.warning("Quantization not available. Falling back to CPU for stability.")
                        self.device = "cpu"
                        self.device_id = -1
                        self.use_gpu = False
                        pipe_kwargs["device"] = -1
                elif self.use_quantization and self.quantization_config is not None and ACCELERATE_AVAILABLE:
                    # GPU with quantization (requires accelerate)
                    pipe_kwargs["model_kwargs"] = {"quantization_config": self.quantization_config}
                    pipe_kwargs["device_map"] = "auto"
                    logger.info("Initializing LLaVA with 4-bit quantization on GPU")
                else:
                    # GPU without quantization - explicitly set device
                    pipe_kwargs["device"] = self.device_id
                    if not ACCELERATE_AVAILABLE and self.use_quantization:
                        logger.warning("Accelerate not available. Using GPU without quantization.")
                    else:
                        logger.info("Loading LLaVA on GPU (full precision)")
            else:
                # CPU fallback
                pipe_kwargs["device"] = -1
                logger.warning("Loading LLaVA on CPU (will be slow)")

            self.pipe = pipeline(**pipe_kwargs)
            
            # Verify device placement
            if self.cuda_available:
                # Check if model is actually on GPU
                try:
                    # Try to get model device
                    model_device = next(self.pipe.model.parameters()).device
                    if model_device.type == "cuda":
                        logger.info(f"✅ Model successfully loaded on GPU: {model_device}")
                    else:
                        logger.warning(f"⚠️ Model loaded on {model_device}, expected GPU")
                except:
                    pass
            
            logger.info(f"Image processor initialized with model: {self.model_id} on {self.device}")
        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                logger.error(f"CUDA out of memory: {e}")
                logger.warning("Attempting to recover by switching to CPU mode...")
                # Clear GPU cache
                if self.use_gpu:
                    torch.cuda.empty_cache()
                # Retry on CPU
                try:
                    pipe_kwargs = {
                        "task": "image-text-to-text",
                        "model": self.model_id,
                        "device": -1,
                    }
                    self.pipe = pipeline(**pipe_kwargs)
                    self.device = "cpu"
                    self.device_id = -1
                    self.use_gpu = False
                    logger.info(f"Successfully loaded model on CPU (fallback after OOM)")
                except Exception as e2:
                    logger.error(f"CPU fallback also failed: {e2}")
                    raise RuntimeError(f"Failed to load model on both GPU and CPU. GPU OOM error: {e}")
            else:
                logger.error(f"Failed to initialize image processor: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to initialize image processor: {e}")
            # Try fallback without device_map
            if "device_map" in str(e) or "accelerate" in str(e).lower():
                logger.info("Retrying without device_map...")
                try:
                    pipe_kwargs = {
                        "task": "image-text-to-text",
                        "model": self.model_id,
                        "device": self.device_id,
                    }
                    if self.use_quantization and self.quantization_config is not None:
                        # Try without device_map
                        pipe_kwargs["model_kwargs"] = {"quantization_config": self.quantization_config}
                    self.pipe = pipeline(**pipe_kwargs)
                    logger.info(f"Image processor initialized (fallback mode) with model: {self.model_id} on {self.device}")
                except Exception as e2:
                    logger.error(f"Fallback initialization also failed: {e2}")
                    # Last resort: try without quantization
                    if self.use_quantization:
                        logger.info("Trying without quantization...")
                        try:
                            pipe_kwargs = {
                                "task": "image-text-to-text",
                                "model": self.model_id,
                                "device": self.device_id,
                            }
                            self.pipe = pipeline(**pipe_kwargs)
                            logger.info(f"Image processor initialized (no quantization) with model: {self.model_id} on {self.device}")
                        except Exception as e3:
                            logger.error(f"Final fallback also failed: {e3}")
                            raise
                    else:
                        raise
            else:
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
                content:
                Describe the image using as much detail as possible,
                is it a painting, a photograph, what colors are predominant,
                role:
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
                text=prompt, 
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
