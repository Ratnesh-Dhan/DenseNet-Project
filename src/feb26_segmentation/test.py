import os
import sys
import subprocess
import platform

# Print system information safely
print("==== System Information ====")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")

# Check for NVIDIA GPU with nvidia-smi (safe, non-TensorFlow approach)
print("\n==== NVIDIA GPU Check ====")
try:
    nvidia_smi_output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
    print("nvidia-smi found and executed successfully:")
    print(nvidia_smi_output)
except Exception as e:
    print(f"Error running nvidia-smi: {e}")
    print("This suggests NVIDIA drivers might not be installed or functioning correctly.")

# Check CUDA environment variables
print("\n==== CUDA Environment Variables ====")
cuda_related_vars = [
    "CUDA_VISIBLE_DEVICES", 
    "LD_LIBRARY_PATH", 
    "CUDA_HOME", 
    "PATH"
]

for var in cuda_related_vars:
    print(f"{var}: {os.environ.get(var, 'Not set')}")

# Try importing TensorFlow with minimal operations
print("\n==== TensorFlow Import Test ====")
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Only check device availability (no operations)
    print("\n==== TensorFlow Device Information ====")
    print(f"Devices visible to TensorFlow: {tf.config.list_physical_devices()}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    
    # Print TensorFlow's CUDA build information
    print("\nTensorFlow CUDA build configuration:")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {tf.test.is_gpu_available()}")
    
except Exception as e:
    print(f"Error importing or using TensorFlow: {e}")

print("\n==== Basic GPU Memory Test ====")
print("This test will be skipped to avoid crashes. Run only after fixing initial issues.")

print("\n==== Recommendations ====")
print("Based on the output above, check for:")
print("1. Working NVIDIA drivers (nvidia-smi should show your GPU)")
print("2. Correct CUDA installation (environment variables should be set)")
print("3. Compatible TensorFlow-GPU installation")
print("4. Sufficient system memory and GPU memory")
