@echo off
REM Run Streamlit app with GPU memory optimization for low-memory GPUs
echo Setting up GPU memory optimization...
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set PYTORCH_NO_CUDA_MEMORY_CACHING=1
echo Starting Streamlit app...
streamlit run app.py
pause

