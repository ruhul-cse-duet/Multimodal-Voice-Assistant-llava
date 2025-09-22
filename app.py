"""
Main Streamlit application for the Multimodal Voice Assistant
"""
import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Optional
from io import BytesIO

# Import our custom components
from components.multimodal_processor import MultimodalProcessor
from config.settings import UI_CONFIG
from utils.logger import setup_logger
from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment

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

def load_processor():
    """Load the multimodal processor"""
    if st.session_state.processor is None:
        with st.spinner("Loading AI models... This may take a moment."):
            try:
                st.session_state.processor = MultimodalProcessor()
                st.success("Models loaded successfully!")
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
    st.markdown('<p class="sub-header">Upload an image and interact via voice input with AI-powered responses</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload an Image**: Choose an image file (JPG, PNG, etc.)
        2. **Record Audio**: Use the microphone to ask questions about the image
        3. **Get Response**: The AI will analyze both and provide a spoken response
        
        **Features:**
        - 🖼️ Image analysis with LLaVA
        - 🎤 Speech-to-text with Whisper
        - 🔊 Text-to-speech responses
        - 🤖 AI-powered multimodal understanding
        """)
        
        st.header("⚙️ Settings")
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
            st.image(uploaded_image, caption="Input Image", use_column_width=True)
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
        recorded_bytes = audio_recorder(text="🎤 Click to Record", recording_color="#e74c3c", neutral_color="#2ecc71", icon_name="microphone", icon_size="2x")

        recorded_audio_path = None
        if recorded_bytes:
            # Save recorded audio bytes to temp WAV file
            temp_dir = Path(tempfile.gettempdir()) / "multimodal_assistant"
            temp_dir.mkdir(exist_ok=True)
            recorded_audio_path = temp_dir / "recorded_audio.wav"
            try:
                audio_seg = AudioSegment.from_file(BytesIO(recorded_bytes), format="wav")
                audio_seg.export(recorded_audio_path, format="wav")
                st.success(f"Recorded audio saved: {recorded_audio_path.name}")
                st.session_state["recorded_audio_path"] = str(recorded_audio_path)
            except Exception:
                # Fallback: write bytes directly
                with open(recorded_audio_path, "wb") as f:
                    f.write(recorded_bytes)
                st.success(f"Recorded audio saved: {recorded_audio_path.name}")
                st.session_state["recorded_audio_path"] = str(recorded_audio_path)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Process button
    if st.button("🚀 Process Inputs", type="primary", use_container_width=True):
        if not uploaded_image and not uploaded_audio:
            st.warning("Please upload at least an image or audio file.")
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
            elif st.session_state.get("recorded_audio_path"):
                audio_path = st.session_state.get("recorded_audio_path")
            
            # Process inputs
            if image_path or audio_path:
                with st.spinner("Processing your inputs... This may take a moment."):
                    try:
                        transcribed_text, generated_response, audio_output_path = st.session_state.processor.process_multimodal_input(
                            audio_path=audio_path,
                            image_path=image_path
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
            st.image(st.session_state.image_path, caption="Input Image", use_column_width=True)
        
        # Show transcribed text
        if st.session_state.get('transcribed_text'):
            st.write("**🎤 What you said:**")
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
