import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load the trained model
model = keras.models.load_model("segmentation_model_Epoch_100.h5")

def predict_mask(image_path, threshold=0.5):
    """
    Predict segmentation mask for a given image.
    
    Args:
        image_path: Path to the input image
        threshold: Threshold value for binary mask (default 0.5)
        
    Returns:
        original_image: Original image as numpy array
        predicted_mask: Binary mask as numpy array
    """
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB").resize((256, 256))
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    input_tensor = np.expand_dims(img_array, axis=0)
    
    # Predict mask
    predicted_mask = model.predict(input_tensor)[0]
    
    # Apply threshold to get binary mask
    binary_mask = (predicted_mask > threshold).astype(np.uint8)
    
    return img_array, binary_mask

def visualize_prediction(image_path, threshold=0.5, save_path=None):
    """
    Visualize the original image, predicted mask, and overlay.
    
    Args:
        image_path: Path to the input image
        threshold: Threshold value for binary mask (default 0.5)
        save_path: Path to save the visualization (optional)
    """
    # Get prediction
    img, mask = predict_mask(image_path, threshold)
    
    # Squeeze mask to remove channel dimension
    mask = mask.squeeze()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Plot predicted mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")
    
    # Create overlay (original image with mask as alpha channel)
    overlay = img.copy()
    overlay_mask = np.zeros_like(img)
    overlay_mask[..., 0] = mask * 255  # Red channel
    alpha = 0.5
    overlay = (1-alpha) * img + alpha * (overlay_mask / 255)
    
    # Plot overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def process_directory(input_dir, output_dir, threshold=0.5):
    """
    Process all images in a directory and save the predicted masks.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save the masks
        threshold: Threshold value for binary mask (default 0.5)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        
        # Get prediction
        _, mask = predict_mask(img_path, threshold)
        
        # Save mask
        mask_img = Image.fromarray(mask.squeeze() * 255).convert("L")
        mask_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_mask.png")
        mask_img.save(mask_path)
        
        # Also save visualization
        viz_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_viz.png")
        visualize_prediction(img_path, threshold, viz_path)
        
    print(f"Processed {len(image_files)} images. Results saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    # Example 1: Predict and visualize a single image
    image_path = "./test2.jpg"
    visualize_prediction(image_path)
    
    # Example 2: Process a directory of images
    # process_directory("path/to/input/images", "path/to/output/masks")