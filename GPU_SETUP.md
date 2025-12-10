# GPU Setup Guide

This guide will help you ensure your GPU is being used by the Multimodal Voice Assistant.

## Step 1: Verify GPU Detection

Run the GPU test script:
```bash
python test_gpu.py
```

This will show you:
- ✅ PyTorch version
- ✅ CUDA availability
- ✅ GPU name and memory
- ✅ GPU computation test

## Step 2: Install PyTorch with CUDA 12.8 Support

Since you have CUDA Toolkit 12.8 installed, install PyTorch with CUDA 12.8 support:

### For CUDA 12.8 (Recommended for your setup):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

This will install the latest PyTorch 2.7.x with CUDA 12.8 support.

### Alternative: If cu128 is not available, use CUDA 12.4 (backward compatible):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### For other CUDA versions:
- **CUDA 11.8**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- **CUDA 12.1**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

### Verify Installation:
After installation, verify CUDA is working:
```python
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Step 3: Verify in the App

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Check the sidebar - you should see:
   - ✅ GPU Available: [Your GPU Name] ([X.XX] GB)

3. When you click "Process Inputs", check the console/logs for:
   - ✅ CUDA available! Using GPU: [GPU Name]
   - ✅ Model successfully loaded on GPU: cuda:0
   - ✅ Whisper model confirmed on GPU: cuda:0

## Step 4: Monitor GPU Usage

### Windows (NVIDIA GPU):
```bash
nvidia-smi -l 1
```

This will show real-time GPU usage. You should see:
- Memory usage increase when models load
- GPU utilization during processing

## Troubleshooting

### Issue: GPU not detected
**Solution:**
1. Check if CUDA drivers are installed: `nvidia-smi` (should show your GPU)
2. Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with correct CUDA version

### Issue: Models still using CPU
**Solution:**
1. Check logs for errors
2. Verify `torch.cuda.is_available()` returns `True`
3. Check if models are actually on GPU in logs

### Issue: Out of Memory
**Solution:**
1. The code will automatically use quantization if available
2. If still OOM, reduce model size in `config/settings.py`:
   - Change `whisper_model_size` from "medium" to "base" or "small"
   - Consider using a smaller LLaVA model

## Current Configuration

The project automatically:
- ✅ Detects GPU availability
- ✅ Uses GPU for LLaVA (image processing)
- ✅ Uses GPU for Whisper (audio processing)
- ✅ Falls back to CPU if GPU unavailable
- ✅ Shows GPU status in the UI

## Verification Commands

```python
# In Python console
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

