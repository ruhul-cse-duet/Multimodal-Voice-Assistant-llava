# GPU Memory Optimization Guide

## Problem: CUDA Out of Memory (OOM) Error

If you're getting "CUDA out of memory" errors, especially with a 2GB GPU, this guide will help you resolve it.

## Solutions Implemented

### 1. **Automatic Memory Management**
- Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation
- GPU cache is cleared before model loading
- Automatic fallback to CPU if GPU runs out of memory

### 2. **Low-Memory GPU Detection**
- The app automatically detects GPUs with less than 4GB
- For 2GB GPUs, it will:
  - Use aggressive quantization (4-bit) if available
  - Automatically fallback to CPU if OOM occurs
  - Show warnings about low memory

### 3. **Model Size Optimization**
- Changed Whisper model from "medium" to "base" (uses less memory)
- LLaVA model uses 4-bit quantization when possible

### 4. **Error Recovery**
- If GPU OOM occurs, the app automatically tries CPU mode
- Clear error messages and recovery suggestions

## How to Use

### Option 1: Run Normally (Recommended)
The app will automatically handle memory optimization:
```bash
streamlit run app.py
```

### Option 2: Use Memory Optimization Script (Windows)
```bash
run_with_memory_optimization.bat
```

### Option 3: Use Memory Optimization Script (Linux/Mac)
```bash
chmod +x run_with_memory_optimization.sh
./run_with_memory_optimization.sh
```

## Manual Memory Management

### Clear GPU Memory Before Running
```python
import torch
torch.cuda.empty_cache()
```

### Force CPU Mode
If you want to force CPU mode (bypasses GPU entirely), set environment variable:
```bash
# Windows
set CUDA_VISIBLE_DEVICES=""
streamlit run app.py

# Linux/Mac
export CUDA_VISIBLE_DEVICES=""
streamlit run app.py
```

## Troubleshooting

### Issue: Still Getting OOM Errors

**Solution 1: Close Other Applications**
- Close any other applications using GPU (games, other ML models, etc.)
- Check GPU usage: `nvidia-smi`

**Solution 2: Restart the App**
- Completely close Streamlit
- Restart to clear GPU memory

**Solution 3: Use CPU Mode**
- The app will automatically fallback to CPU if GPU fails
- Or manually set `CUDA_VISIBLE_DEVICES=""` before running

**Solution 4: Reduce Model Size**
Edit `config/settings.py`:
```python
MODEL_CONFIG = {
    "llava_model_id": "llava-hf/llava-1.5-7b-hf",  # Keep this
    "whisper_model_size": "tiny",  # Change from "base" to "tiny" for even less memory
    "max_new_tokens": 150,  # Reduce from 200 to 150
}
```

### Issue: Models Load Slowly

This is normal for:
- First run (models are downloaded)
- CPU mode (slower but more stable)
- Low-memory GPUs (quantization adds overhead)

## Expected Behavior

### For 2GB GPU:
- ✅ App detects low memory
- ✅ Shows warning in UI
- ✅ Uses quantization if available
- ✅ Automatically falls back to CPU if OOM
- ⚠️ May be slower than larger GPUs

### For 4GB+ GPU:
- ✅ Uses GPU with quantization
- ✅ Faster processing
- ✅ More stable

### For CPU:
- ✅ Always works (no OOM)
- ⚠️ Slower processing
- ✅ More stable for low-memory systems

## Monitoring GPU Memory

### Check Current Usage
```bash
nvidia-smi
```

### Monitor in Real-Time
```bash
nvidia-smi -l 1
```

### In Python
```python
import torch
if torch.cuda.is_available():
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

## Summary

The app now:
- ✅ Automatically optimizes memory usage
- ✅ Detects low-memory GPUs
- ✅ Falls back to CPU if needed
- ✅ Provides clear error messages
- ✅ Uses quantization to reduce memory

**For a 2GB GPU, the app will work but may need to use CPU mode for stability.**

