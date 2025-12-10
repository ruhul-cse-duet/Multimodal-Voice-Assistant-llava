"""
Quick GPU verification script
Run this to verify your GPU is detected and working
"""
import sys

print("=" * 60)
print("GPU Verification Script")
print("=" * 60)

# Check PyTorch
try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
except ImportError:
    print("❌ PyTorch not installed!")
    sys.exit(1)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\nCUDA Available: {cuda_available}")

if cuda_available:
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ PyTorch CUDA Build: {torch.version.cuda}")
    print(f"✅ GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"✅ Number of GPUs: {torch.cuda.device_count()}")
    
    # Check if CUDA 12.8 is being used
    cuda_version = torch.version.cuda
    if cuda_version.startswith("12.8") or cuda_version.startswith("12.4"):
        print(f"✅ CUDA version compatible with CUDA Toolkit 12.8")
    else:
        print(f"ℹ️  CUDA version {cuda_version} (should work with CUDA Toolkit 12.8)")
    
    # Test GPU computation
    try:
        print("\n🧪 Testing GPU computation...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU computation test passed!")
        print(f"✅ Result tensor device: {z.device}")
    except Exception as e:
        print(f"❌ GPU computation test failed: {e}")
else:
    print("\n⚠️  CUDA not available. Possible reasons:")
    print("   1. PyTorch was installed without CUDA support")
    print("   2. CUDA drivers are not installed")
    print("   3. GPU is not compatible")
    print("\nTo install PyTorch with CUDA support:")
    print("   Visit: https://pytorch.org/get-started/locally/")

# Check transformers
try:
    from transformers import pipeline
    print("\n✅ Transformers library available")
except ImportError:
    print("\n❌ Transformers library not installed!")

# Check accelerate
try:
    import accelerate
    print(f"✅ Accelerate library available: {accelerate.__version__}")
except ImportError:
    print("⚠️  Accelerate library not installed (optional for quantization)")

print("\n" + "=" * 60)
print("Verification Complete!")
print("=" * 60)

