import os
import sys
from diagnostic_inference import main as run_diagnostic

# Script to run the diagnostic inference tool

def setup_paths():
    """Check and create necessary directories"""
    # Create model directory if it doesn't exist
    os.makedirs('./model', exist_ok=True)
    
    # Create test image directory if it doesn't exist
    os.makedirs('../test/image', exist_ok=True)
    
    print("Directories created successfully.")

def check_model_file():
    """Check if model file exists"""
    model_path = './model/unet_resnet50_multiclass.h5'
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        print("Please place your model file at this location before running the diagnostics.")
        return False
    return True

def check_test_image():
    """Check if test image exists"""
    image_path = "../test/image/1.png"
    if not os.path.exists(image_path):
        print(f"Test image not found at {image_path}")
        print("Please place a test image at this location before running the diagnostics.")
        return False
    return True

if __name__ == "__main__":
    print("Setting up environment...")
    setup_paths()
    
    all_good = True
    
    if not check_model_file():
        all_good = False
    
    if not check_test_image():
        all_good = False
    
    if all_good:
        print("\nAll files and directories are in place!")
        print("Running diagnostic inference...")
        
        # Run the diagnostic tool
        run_diagnostic(debug_mode=True)
    else:
        print("\nPlease fix the missing files before running the diagnostic.")