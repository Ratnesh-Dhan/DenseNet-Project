import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Import your model building function
from trasferLearningModelHardMode import build_unet_with_transfer_learning

def load_trained_model(model_path):
    """
    Load a trained model from a saved file
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        model: Loaded Keras model
    """
    try:
        # Try to load the model directly
        model = load_model(model_path)
        print("Model loaded successfully!")
    except:
        # If loading fails, rebuild the model architecture and load weights
        print("Direct model loading failed. Rebuilding model and loading weights...")
        model = build_unet_with_transfer_learning()
        model.load_weights(model_path)
        print("Model weights loaded successfully!")
        
    return model

def preprocess_image(image_path, target_size=(512, 512)):
    """
    Preprocess an image for inference
    
    Args:
        image_path (str): Path to the input image
        target_size (tuple): Target size for resizing
        
    Returns:
        preprocessed_img: Preprocessed image ready for model input
        original_img: Original image for display
    """
    # Load and store original image for display
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Load and preprocess image for model input
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    
    # Normalize pixel values to [0,1]
    img = img / 255.0
    
    # Add batch dimension
    preprocessed_img = np.expand_dims(img, axis=0)
    
    return preprocessed_img, original_img

def perform_inference(model, preprocessed_img):
    """
    Perform inference using the loaded model
    
    Args:
        model: Trained Keras model
        preprocessed_img: Preprocessed input image
        
    Returns:
        prediction: Model prediction
    """
    # Run inference
    prediction = model.predict(preprocessed_img)
    
    return prediction

def process_prediction(prediction, num_classes=3):
    """
    Process model prediction into class masks
    
    Args:
        prediction: Raw model prediction
        num_classes: Number of segmentation classes
        
    Returns:
        segmentation_mask: Segmentation mask with class indices
        class_masks: Individual binary masks for each class
    """
    # Get predicted class for each pixel (argmax across class dimension)
    segmentation_mask = np.argmax(prediction[0], axis=-1)
    
    # Create individual binary masks for each class
    class_masks = []
    for class_idx in range(num_classes):
        class_mask = (segmentation_mask == class_idx).astype(np.uint8)
        class_masks.append(class_mask)
    
    return segmentation_mask, class_masks

def visualize_results(original_img, segmentation_mask, class_masks, class_colors=None):
    """
    Visualize the original image and segmentation results
    
    Args:
        original_img: Original input image
        segmentation_mask: Segmentation mask with class indices
        class_masks: Individual binary masks for each class
        class_colors: List of RGB colors for each class
    """
    # Default colors if none provided
    if class_colors is None:
        class_colors = [
            [255, 0, 0],    # Red for class 0
            [0, 255, 0],    # Green for class 1
            [0, 0, 255]     # Blue for class 2
        ]
    
    # Create colored segmentation mask
    height, width = segmentation_mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_idx, color in enumerate(class_colors):
        colored_mask[segmentation_mask == class_idx] = color
    
    # Create a blended image (original + segmentation overlay)
    alpha = 0.5  # Transparency factor
    blended = cv2.addWeighted(
        original_img, 
        1-alpha, 
        cv2.resize(colored_mask, (original_img.shape[1], original_img.shape[0])), 
        alpha, 
        0
    )
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_img)
    plt.axis('off')
    
    # Segmentation mask
    plt.subplot(2, 2, 2)
    plt.title('Segmentation Mask')
    plt.imshow(colored_mask)
    plt.axis('off')
    
    # Blended image
    plt.subplot(2, 2, 3)
    plt.title('Blended Result')
    plt.imshow(blended)
    plt.axis('off')
    
    # Individual class masks
    fig = plt.figure(figsize=(15, 5))
    for i, mask in enumerate(class_masks):
        plt.subplot(1, len(class_masks), i+1)
        plt.title(f'Class {i} Mask')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_segmentation_results(original_img, segmentation_mask, class_masks, output_dir, image_name, class_colors=None):
    """
    Save segmentation results to files
    
    Args:
        original_img: Original input image
        segmentation_mask: Segmentation mask with class indices
        class_masks: Individual binary masks for each class
        output_dir: Directory to save results
        image_name: Base name for output files
        class_colors: List of RGB colors for each class
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Default colors if none provided
    if class_colors is None:
        class_colors = [
            [255, 0, 0],    # Red for class 0
            [0, 255, 0],    # Green for class 1
            [0, 0, 255]     # Blue for class 2
        ]
    
    # Create colored segmentation mask
    height, width = segmentation_mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_idx, color in enumerate(class_colors):
        colored_mask[segmentation_mask == class_idx] = color
    
    # Create a blended image (original + segmentation overlay)
    alpha = 0.5  # Transparency factor
    blended = cv2.addWeighted(
        original_img, 
        1-alpha, 
        cv2.resize(colored_mask, (original_img.shape[1], original_img.shape[0])), 
        alpha, 
        0
    )
    
    # Save original image
    plt.figure(figsize=(10, 10))
    plt.imshow(original_img)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{image_name}_original.png"), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save segmentation mask
    plt.figure(figsize=(10, 10))
    plt.imshow(colored_mask)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{image_name}_segmentation.png"), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save blended result
    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{image_name}_blended.png"), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save individual class masks
    for i, mask in enumerate(class_masks):
        plt.figure(figsize=(10, 10))
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"{image_name}_class{i}_mask.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
    
    print(f"Results saved to {output_dir}")


def main():
    # Path to your trained model
    model_path = './model/unet_resnet50_multiclass.h5'
    
    # Path to input image for inference
    image_path = "../test/image/1.png"
    
    # Class colors for visualization (RGB format)
    class_colors = [
        [255, 0, 0],    # Red for class 0
        [0, 255, 0],    # Green for class 1
        [0, 0, 255]     # Blue for class 2
    ]
    
    # Load the trained model
    model = load_trained_model(model_path)
    
    # Preprocess the input image
    preprocessed_img, original_img = preprocess_image(image_path)
    
    # Perform inference
    prediction = perform_inference(model, preprocessed_img)
    
    # Process the prediction
    segmentation_mask, class_masks = process_prediction(prediction)
    
    # Visualize the results
    visualize_results(original_img, segmentation_mask, class_masks, class_colors)
    
    # Save the results
    save_segmentation_results(
        original_img, 
        segmentation_mask, 
        class_masks, 
        output_dir="segmentation_results", 
        image_name="test_image",
        class_colors=class_colors
    )


# For batch processing multiple images
def process_image_directory(model, input_dir, output_dir, class_colors=None):
    """
    Process all images in a directory
    
    Args:
        model: Trained Keras model
        input_dir: Directory containing input images
        output_dir: Directory to save results
        class_colors: List of RGB colors for each class
    """
    import os
    from tqdm import tqdm
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the input directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in image_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_file in tqdm(image_files):
        try:
            image_path = os.path.join(input_dir, image_file)
            image_name = os.path.splitext(image_file)[0]
            
            # Preprocess the image
            preprocessed_img, original_img = preprocess_image(image_path)
            
            # Perform inference
            prediction = perform_inference(model, preprocessed_img)
            
            # Process the prediction
            segmentation_mask, class_masks = process_prediction(prediction)
            
            # Save the results
            save_segmentation_results(
                original_img, 
                segmentation_mask, 
                class_masks, 
                output_dir=output_dir, 
                image_name=image_name,
                class_colors=class_colors
            )
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")


if __name__ == "__main__":
    main()
    
    # Alternatively, for batch processing:
    # model_path = "path/to/your/trained_model.h5"
    # model = load_trained_model(model_path)
    # process_image_directory(
    #     model,
    #     input_dir="path/to/input/images",
    #     output_dir="path/to/output/results"
    # )