import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os

# Load the trained model
model_path = "best_segmentation_model.h5"  # Use the path to your best saved model
model = keras.models.load_model(model_path, compile=False)

# Image preprocessing function
def preprocess_image(image_path, target_size=(256, 256)):
    # Load and convert to RGB
    img = Image.open(image_path).convert("RGB")
    
    # Resize
    img_resized = img.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(img_resized) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch, np.array(img)  # Return preprocessed image and original image

# Postprocess the predicted mask
def postprocess_mask(mask, threshold=0.5, original_size=None):
    # Convert to binary using the threshold
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    
    # Remove batch dimension and channel dimension
    binary_mask = binary_mask[0, :, :, 0]
    
    # Resize to original image size if specified
    if original_size:
        binary_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]), 
                                interpolation=cv2.INTER_NEAREST)
    
    return binary_mask

# Function to perform segmentation on a new image
def segment_image(image_path, model, threshold=0.5, overlay_alpha=0.5):
    # Preprocess the image
    preprocessed_img, original_img = preprocess_image(image_path)
    
    # Get original image dimensions
    original_size = original_img.shape[:2]  # (height, width)
    
    # Predict
    prediction = model.predict(preprocessed_img)
    
    # Postprocess the mask
    binary_mask = postprocess_mask(prediction, threshold, original_size)
    
    # Create colored mask for visualization (you can choose any color)
    colored_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    colored_mask[binary_mask > 0] = [0, 255, 0]  # Green mask
    
    # Create the overlay
    overlay = cv2.addWeighted(
        original_img, 
        1.0, 
        colored_mask, 
        overlay_alpha, 
        0
    )
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return binary_mask, overlay

# Function to segment multiple images in a directory
def segment_directory(directory_path, model, output_dir=None, threshold=0.5):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "masks"))
        os.makedirs(os.path.join(output_dir, "overlays"))
    
    image_files = [f for f in os.listdir(directory_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    print(f"Found {len(image_files)} images in {directory_path}")
    
    for img_file in image_files:
        image_path = os.path.join(directory_path, img_file)
        print(f"Processing {img_file}...")
        
        try:
            mask, overlay = segment_image(image_path, model, threshold)
            
            if output_dir:
                # Save the mask
                mask_path = os.path.join(output_dir, "masks", f"{os.path.splitext(img_file)[0]}_mask.png")
                cv2.imwrite(mask_path, mask)
                
                # Save the overlay
                overlay_path = os.path.join(output_dir, "overlays", f"{os.path.splitext(img_file)[0]}_overlay.png")
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)  # Convert to RGB for saving
                cv2.imwrite(overlay_path, overlay_rgb)
                
                print(f"Saved results to {mask_path} and {overlay_path}")
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

# Example usage:
if __name__ == "__main__":
    # Single image segmentation
    image_path = "path/to/your/test_image.jpg"  # Replace with path to your test image
    mask, overlay = segment_image(image_path, model)
    
    # Optionally process a directory of images
    # segment_directory("path/to/test_images", model, output_dir="path/to/results")