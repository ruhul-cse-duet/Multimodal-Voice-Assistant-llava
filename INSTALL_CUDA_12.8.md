# Installation Guide for CUDA 12.8

## Quick Start for CUDA 12.8

Since you already have CUDA Toolkit 12.8 installed, follow these steps:

### Step 1: Uninstall existing PyTorch (if installed)
```bash
pip uninstall torch torchvision torchaudio
```

### Step 2: Install PyTorch with CUDA 12.8 Support
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Step 3: Verify Installation
```bash
python test_gpu.py
```

You should see:
- ✅ CUDA Available: True
- ✅ CUDA Version: 12.8
- ✅ GPU Device: [Your GPU Name]

### Step 4: Install Other Dependencies
```bash
pip install -r requirements.txt
```

Note: The requirements.txt will skip PyTorch since you've already installed it with CUDA support.

### Step 5: Verify in Python
```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

## Complete Installation Script

For Windows (PowerShell or CMD):
```bash
# 1. Uninstall old PyTorch
pip uninstall torch torchvision torchaudio -y

# 2. Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install other dependencies (excluding torch)
pip install streamlit pydub==0.25.1 streamlit-audiorecorder audio-recorder-streamlit==0.0.8 transformers==4.57.1 accelerate>=0.25.0 openai-whisper gradio gTTS Pillow==10.1.0 nltk==3.8.1 requests==2.32.5

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# 5. Test GPU
python test_gpu.py
```

## Troubleshooting

### Issue: "cu128" index not found
**Solution**: Use cu124 instead (backward compatible with CUDA 12.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Issue: CUDA still not detected
**Solution**: 
1. Verify CUDA toolkit: `nvcc --version` (should show 12.8)
2. Check NVIDIA drivers: `nvidia-smi` (should show your GPU)
3. Restart your terminal/IDE after installation

### Issue: Version conflicts
**Solution**: Create a fresh virtual environment:
```bash
python -m venv venv_cuda128
venv_cuda128\Scripts\activate  # Windows
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Expected Output

After successful installation:
```
PyTorch Version: 2.7.x+cu128
CUDA Available: True
CUDA Version: 12.8
GPU: [Your GPU Name]
GPU Memory: [X.XX] GB
```

## Next Steps

1. Run the test: `python test_gpu.py`
2. Start the app: `streamlit run app.py`
3. Check GPU status in the sidebar
4. Monitor GPU usage: `nvidia-smi -l 1`

Your project is now ready to use CUDA 12.8! 🚀

