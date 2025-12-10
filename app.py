"""
Main Streamlit application for the Multimodal Voice Assistant
"""
import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Optional
from io import BytesIO

# Set PyTorch CUDA memory management for low-memory GPUs
# This helps prevent OOM errors on GPUs with limited memory
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Import our custom components
from components.multimodal_processor import MultimodalProcessor
from config.settings import UI_CONFIG
from utils.logger import setup_logger
try:
    from streamlit_audiorecorder import st_audiorec
    AUDIO_RECORDER_AVAILABLE = True
    USE_LEGACY_RECORDER = False
except ImportError:
    try:
        from audio_recorder_streamlit import audio_recorder
        AUDIO_RECORDER_AVAILABLE = True
        USE_LEGACY_RECORDER = True
    except ImportError:
        AUDIO_RECORDER_AVAILABLE = False
        USE_LEGACY_RECORDER = False
        st_audiorec = None
        audio_recorder = None

# Setup logger
logger = setup_logger(__name__)

# Page configuration
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"]
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .result-section {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'recorded_audio_path' not in st.session_state:
        st.session_state.recorded_audio_path = None
    if 'recording_status' not in st.session_state:
        st.session_state.recording_status = "Ready to record"
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""

def check_gpu_status():
    """Check and display GPU status"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return True, f"✅ GPU Available: {gpu_name} ({gpu_memory:.2f} GB)"
        else:
            return False, "❌ GPU Not Available - Using CPU"
    except Exception as e:
        return False, f"⚠️ GPU Check Failed: {e}"

def load_processor():
    """Load the multimodal processor"""
    if st.session_state.processor is None:
        # Check GPU status first
        gpu_available, gpu_status = check_gpu_status()
        if gpu_available:
            st.info(gpu_status)
            # Check GPU memory and warn if low
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    if gpu_memory < 4.0:
                        st.warning(f"⚠️ Low GPU memory ({gpu_memory:.2f} GB). The app will automatically use CPU if GPU runs out of memory.")
            except:
                pass
        else:
            st.warning(gpu_status)
        
        with st.spinner("Loading AI models... This may take a moment."):
            try:
                st.session_state.processor = MultimodalProcessor()
                st.success("Models loaded successfully!")
                
                # Display device info after loading
                if hasattr(st.session_state.processor, 'image_processor'):
                    img_device = st.session_state.processor.image_processor.device
                    if img_device == "cuda":
                        st.success(f"🖼️ Image Model: Running on GPU")
                    else:
                        st.info(f"🖼️ Image Model: Running on CPU (GPU memory insufficient or not available)")
                
                if hasattr(st.session_state.processor, 'audio_processor'):
                    audio_device = st.session_state.processor.audio_processor.device
                    if audio_device == "cuda":
                        st.success(f"🎤 Audio Model: Running on GPU")
                    else:
                        st.info(f"🎤 Audio Model: Running on CPU")
                        
            except RuntimeError as e:
                error_msg = str(e)
                if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                    st.error("❌ GPU ran out of memory. The app will try to use CPU instead.")
                    st.info("💡 **Tips to reduce memory usage:**\n"
                           "- Close other GPU-intensive applications\n"
                           "- Restart the app to clear GPU memory\n"
                           "- Consider using CPU mode if GPU memory is consistently insufficient")
                    logger.error(f"GPU OOM error: {e}")
                else:
                    st.error(f"Failed to load models: {e}")
                    logger.error(f"Failed to load processor: {e}")
                return False
            except Exception as e:
                st.error(f"Failed to load models: {e}")
                logger.error(f"Failed to load processor: {e}")
                return False
    return True

def save_uploaded_file(uploaded_file, file_type: str) -> Optional[str]:
    """Save uploaded file to temporary directory"""
    try:
        # Create temporary file
        temp_dir = Path(tempfile.gettempdir()) / "multimodal_assistant"
        temp_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        file_extension = Path(uploaded_file.name).suffix
        temp_file = temp_dir / f"{file_type}_{uploaded_file.name}"
        
        # Save file
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(temp_file)
    except Exception as e:
        st.error(f"Error saving file: {e}")
        logger.error(f"Error saving uploaded file: {e}")
        return None

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎤 Multimodal Voice Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image and interact via voice, text, or both with AI-powered responses</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload an Image** (Optional): Choose an image file (JPG, PNG, etc.)
        2. **Provide Input**: 
           - 🎤 Record or upload audio
           - ✍️ Type your question/prompt
           - Or use both!
        3. **Get Response**: The AI will analyze and provide a spoken response
        
        **Features:**
        - 🖼️ Image analysis with LLaVA
        - 🎤 Speech-to-text with Whisper
        - ✍️ Text input support
        - 🔊 Text-to-speech responses
        - 🤖 AI-powered multimodal understanding
        """)
        
        st.header("⚙️ Settings")
        
        # Display GPU status
        gpu_available, gpu_status = check_gpu_status()
        if gpu_available:
            st.success(gpu_status)
        else:
            st.warning(gpu_status)
        
        st.info("Models will be loaded automatically when you first use the app.")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📁 Upload Image")
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
            help="Upload an image that you want to ask questions about"
        )
        # Preview uploaded image
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Input Image", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("🎤 Record Audio")
        uploaded_audio = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'm4a', 'ogg'],
            help="Upload an audio file with your question"
        )
        
        # Alternative: Record audio directly
        st.subheader("🎙️ Or Record Live")
        
        if not AUDIO_RECORDER_AVAILABLE:
            st.error("⚠️ Audio recorder not available. Please install: `pip install streamlit-audiorecorder`")
        else:
            try:
                if USE_LEGACY_RECORDER:
                    # Use legacy audio_recorder_streamlit
                    recorded_bytes = audio_recorder(
                        text="🎤 Click to Record",
                        recording_color="#e74c3c",
                        neutral_color="#2ecc71",
                        icon_name="microphone",
                        icon_size="2x",
                        sample_rate=44100,
                        key="live_audio_recorder",
                    )
                    
                    if recorded_bytes:
                        # Save recorded audio bytes to temp WAV file
                        temp_dir = Path(tempfile.gettempdir()) / "multimodal_assistant"
                        temp_dir.mkdir(exist_ok=True)
                        recorded_audio_path = temp_dir / f"recorded_audio_{len(recorded_bytes)}.wav"
                        
                        try:
                            with open(recorded_audio_path, "wb") as f:
                                f.write(recorded_bytes)
                            
                            st.success(f"✅ Recording saved! ({len(recorded_bytes)} bytes)")
                            st.audio(recorded_bytes, format="audio/wav")
                            st.session_state.recorded_audio_path = str(recorded_audio_path)
                            st.session_state.recording_status = "Recording saved successfully"
                        except Exception as e:
                            st.error(f"Error saving recording: {e}")
                            logger.error(f"Error saving audio: {e}")
                    else:
                        st.info("💡 Click the microphone button above to start recording")
                        if st.session_state.recorded_audio_path:
                            st.info(f"📁 Last recording: {Path(st.session_state.recorded_audio_path).name}")
                else:
                    # Use streamlit-audiorecorder (preferred)
                    st.info("💡 Click the microphone button below to start recording")
                    recorded_bytes = st_audiorec()
                    
                    if recorded_bytes:
                        # Save recorded audio bytes to temp WAV file
                        temp_dir = Path(tempfile.gettempdir()) / "multimodal_assistant"
                        temp_dir.mkdir(exist_ok=True)
                        recorded_audio_path = temp_dir / f"recorded_audio_{len(recorded_bytes)}.wav"
                        
                        try:
                            with open(recorded_audio_path, "wb") as f:
                                f.write(recorded_bytes)
                            
                            st.success(f"✅ Recording saved! ({len(recorded_bytes)} bytes)")
                            st.audio(recorded_bytes, format="audio/wav")
                            st.session_state.recorded_audio_path = str(recorded_audio_path)
                            st.session_state.recording_status = "Recording saved successfully"
                        except Exception as e:
                            st.error(f"Error saving recording: {e}")
                            logger.error(f"Error saving audio: {e}")
                    else:
                        if st.session_state.recorded_audio_path:
                            st.info(f"📁 Last recording: {Path(st.session_state.recorded_audio_path).name}")
                            st.audio(st.session_state.recorded_audio_path)
                            
            except Exception as e:
                st.error(f"❌ Recording error: {e}")
                logger.error(f"Audio recording error: {e}")
                st.info("💡 **Troubleshooting:**\n"
                        "- Check browser microphone permissions\n"
                        "- Try Chrome or Edge browser\n"
                        "- Ensure you're on HTTPS or localhost\n"
                        "- Try uploading an audio file instead")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Text input section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("✍️ Or Type Your Question")
    text_input = st.text_area(
        "Enter your question or prompt here",
        value=st.session_state.get('text_input', ''),
        placeholder="e.g., What do you see in this image? Describe the colors and objects.",
        help="Type your question or prompt. This will be used along with the image (if provided) or as standalone input.",
        height=100,
        key="text_input_area"
    )
    if text_input:
        st.info(f"📝 Text input: {text_input[:100]}{'...' if len(text_input) > 100 else ''}")
        st.session_state.text_input = text_input
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process button
    if st.button("🚀 Process Inputs", type="primary", use_container_width=True):
        # Check if we have any input
        has_image = uploaded_image is not None
        has_uploaded_audio = uploaded_audio is not None
        has_recorded_audio = st.session_state.get("recorded_audio_path") is not None and os.path.exists(st.session_state.get("recorded_audio_path", ""))
        has_text = text_input and text_input.strip() if text_input else False
        
        if not has_image and not has_uploaded_audio and not has_recorded_audio and not has_text:
            st.warning("⚠️ Please provide at least one input: upload an image, record/upload audio, or type a question.")
        else:
            # Load processor
            if not load_processor():
                st.stop()
            
            # Save uploaded files
            image_path = None
            audio_path = None
            
            if uploaded_image:
                image_path = save_uploaded_file(uploaded_image, "image")
                if image_path:
                    st.success(f"Image saved: {uploaded_image.name}")
                    st.session_state.image_path = image_path
            
            if uploaded_audio:
                audio_path = save_uploaded_file(uploaded_audio, "audio")
                if audio_path:
                    st.success(f"Audio saved: {uploaded_audio.name}")
            elif st.session_state.get("recorded_audio_path") and os.path.exists(st.session_state.get("recorded_audio_path", "")):
                audio_path = st.session_state.get("recorded_audio_path")
                st.info(f"Using recorded audio: {Path(audio_path).name}")
            
            # Process inputs
            text_input_value = text_input.strip() if text_input and text_input.strip() else None
            if image_path or audio_path or text_input_value:
                with st.spinner("Processing your inputs... This may take a moment."):
                    try:
                        transcribed_text, generated_response, audio_output_path = st.session_state.processor.process_multimodal_input(
                            audio_path=audio_path,
                            image_path=image_path,
                            text_input=text_input_value
                        )
                        
                        st.session_state.processing_complete = True
                        st.session_state.transcribed_text = transcribed_text
                        st.session_state.generated_response = generated_response
                        st.session_state.audio_output_path = audio_output_path
                        
                    except Exception as e:
                        st.error(f"Error processing inputs: {e}")
                        logger.error(f"Error in main processing: {e}")
    
    # Display results
    if st.session_state.get('processing_complete', False):
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.subheader("📝 Results")
        
        # Show input image
        if st.session_state.get('image_path') and os.path.exists(st.session_state.image_path):
            st.image(st.session_state.image_path, caption="Input Image", width="stretch")
        
        # Show input text (if provided)
        if st.session_state.get('text_input') and st.session_state.text_input.strip():
            st.write("**✍️ Your Text Input:**")
            st.write(st.session_state.text_input)
        
        # Show transcribed text
        if st.session_state.get('transcribed_text'):
            st.write("**🎤 What you said (from audio):**")
            st.write(st.session_state.transcribed_text)
        
        # Show generated response
        if st.session_state.get('generated_response'):
            st.write("**🤖 AI Response:**")
            st.write(st.session_state.generated_response)
        
        # Play audio response
        if st.session_state.get('audio_output_path') and os.path.exists(st.session_state.audio_output_path):
            st.write("**🔊 Audio Response:**")
            st.audio(st.session_state.audio_output_path)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit, LLaVA, and Whisper</p>
            <p>Upload your code to GitHub and share with the world! 🌍</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

# streamlit run app.py
