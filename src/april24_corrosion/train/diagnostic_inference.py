import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Define the model building function here for direct use
def build_unet_with_transfer_learning(input_shape=(512, 512, 3), num_classes=3):
    # Load a pre-trained ResNet50 model without the top classification layers
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the layers of the base model to prevent training
    for layer in base_model.layers:
        layer.trainable = False

    # Encoder (using pre-trained ResNet50)
    encoder_output = base_model.output

    # Decoder (U-Net style: Conv2DTranspose layers with concatenation)
    u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(encoder_output)
    u6 = tf.keras.layers.concatenate([u6, base_model.get_layer("conv4_block6_out").output])  # Concatenate with ResNet feature map
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, base_model.get_layer("conv3_block4_out").output])  # Concatenate with ResNet feature map
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, base_model.get_layer("conv2_block3_out").output])  # Concatenate with ResNet feature map
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, base_model.get_layer("conv1_relu").output])  # Concatenate with ResNet feature map
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    u10 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
    c10 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u10)
    c10 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c10)
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax')(c10)

    # Create the model
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return model

# Import your model building function if needed
# from your_model_file import build_unet_with_transfer_learning

def load_trained_model(model_path):
    """
    Load a trained model from a saved file with better error handling
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        model: Loaded Keras model
    """
    try:
        # Try to load the model directly with custom_objects if needed
        # If your model has custom layers or losses, you might need to provide them here
        model = load_model(model_path)
        print("Model loaded successfully!")
        
        # Print model input and output shapes for diagnostic purposes
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        
        # Since we've included the build_unet_with_transfer_learning function directly,
        # we can use it for rebuilding
        try:
            print("Rebuilding model and loading weights...")
            model = build_unet_with_transfer_learning()
            model.load_weights(model_path)
            print("Model weights loaded successfully!")
            return model
        except Exception as nested_e:
            print(f"Failed to rebuild model: {str(nested_e)}")
            raise Exception("Could not load model using either method.")

def preprocess_image(image_path, target_size=(512, 512), normalization='standard'):
    """
    Preprocess an image for inference with different normalization options
    
    Args:
        image_path (str): Path to the input image
        target_size (tuple): Target size for resizing
        normalization (str): Normalization method ('standard', 'imagenet', or 'none')
        
    Returns:
        preprocessed_img: Preprocessed image ready for model input
        original_img: Original image for display
    """
    # Load and store original image for display
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Load and preprocess image for model input
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    
    # Apply different normalization based on parameter
    if normalization == 'standard':
        # Standard normalization to [0,1]
        img = img / 255.0
    elif normalization == 'imagenet':
        # ImageNet normalization (for models pre-trained on ImageNet)
        img = tf.keras.applications.resnet50.preprocess_input(img)
    elif normalization == 'none':
        # No normalization
        pass
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    
    # Add batch dimension
    preprocessed_img = np.expand_dims(img, axis=0)
    
    return preprocessed_img, original_img

def perform_inference(model, preprocessed_img, debug=False):
    """
    Perform inference using the loaded model with additional diagnostics
    
    Args:
        model: Trained Keras model
        preprocessed_img: Preprocessed input image
        debug: Whether to print debug information
        
    Returns:
        prediction: Model prediction
    """
    # Print input stats for debugging
    if debug:
        print(f"Input shape: {preprocessed_img.shape}")
        print(f"Input min: {np.min(preprocessed_img)}, max: {np.max(preprocessed_img)}, mean: {np.mean(preprocessed_img)}")
    
    # Run inference
    prediction = model.predict(preprocessed_img)
    
    # Print prediction stats for debugging
    if debug:
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction min: {np.min(prediction)}, max: {np.max(prediction)}, mean: {np.mean(prediction)}")
        
        # Print class distribution (how many pixels are assigned to each class)
        class_pred = np.argmax(prediction[0], axis=-1)
        unique, counts = np.unique(class_pred, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        print(f"Class distribution: {class_distribution}")
        
        # Print confidence statistics
        confidence = np.max(prediction[0], axis=-1)
        print(f"Confidence min: {np.min(confidence)}, max: {np.max(confidence)}, mean: {np.mean(confidence)}")
    
    return prediction

def process_prediction(prediction, num_classes=3, confidence_threshold=0.5):
    """
    Process model prediction into class masks with confidence threshold
    
    Args:
        prediction: Raw model prediction
        num_classes: Number of segmentation classes
        confidence_threshold: Minimum confidence threshold for class assignment
        
    Returns:
        segmentation_mask: Segmentation mask with class indices
        class_masks: Individual binary masks for each class
        confidence_map: Map of prediction confidence for each pixel
    """
    # Get raw class probabilities
    class_probs = prediction[0]  # Shape: [height, width, num_classes]
    
    # Get predicted class for each pixel (argmax across class dimension)
    segmentation_mask = np.argmax(class_probs, axis=-1)
    
    # Get confidence scores (max probability for each pixel)
    confidence_map = np.max(class_probs, axis=-1)
    
    # Apply confidence threshold (assign background class 0 to low confidence predictions)
    if confidence_threshold > 0:
        segmentation_mask[confidence_map < confidence_threshold] = 0
    
    # Create individual binary masks for each class
    class_masks = []
    for class_idx in range(num_classes):
        class_mask = (segmentation_mask == class_idx).astype(np.uint8)
        class_masks.append(class_mask)
    
    return segmentation_mask, class_masks, confidence_map

def visualize_results(original_img, segmentation_mask, class_masks, confidence_map=None, class_colors=None, class_names=None):
    """
    Visualize the original image, segmentation results, and confidence maps
    
    Args:
        original_img: Original input image
        segmentation_mask: Segmentation mask with class indices
        class_masks: Individual binary masks for each class
        confidence_map: Map of prediction confidence for each pixel
        class_colors: List of RGB colors for each class
        class_names: List of names for each class
    """
    # Default colors if none provided
    if class_colors is None:
        class_colors = [
            [255, 0, 0],    # Red for class 0
            [0, 255, 0],    # Green for class 1
            [0, 0, 255]     # Blue for class 2
        ]
    
    # Default class names if none provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(class_masks))]
    
    # Create colored segmentation mask
    height, width = segmentation_mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_idx, color in enumerate(class_colors):
        colored_mask[segmentation_mask == class_idx] = color
    
    # Resize colored mask to match original image size if needed
    if colored_mask.shape[:2] != original_img.shape[:2]:
        colored_mask = cv2.resize(
            colored_mask, 
            (original_img.shape[1], original_img.shape[0]),
            interpolation=cv2.INTER_NEAREST  # Use nearest neighbor to preserve class labels
        )
    
    # Create a blended image (original + segmentation overlay)
    alpha = 0.5  # Transparency factor
    blended = cv2.addWeighted(
        original_img, 
        1-alpha, 
        colored_mask, 
        alpha, 
        0
    )
    
    # Calculate number of plots needed
    num_plots = 3  # Original, segmentation, blended
    if confidence_map is not None:
        num_plots += 1
    
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
    
    # Confidence map if available
    if confidence_map is not None:
        plt.subplot(2, 2, 4)
        plt.title('Confidence Map')
        plt.imshow(confidence_map, cmap='viridis')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Individual class masks
    num_classes = len(class_masks)
    fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))
    
    # Handle the case of a single class
    if num_classes == 1:
        axes = [axes]
        
    for i, (mask, name) in enumerate(zip(class_masks, class_names)):
        axes[i].imshow(mask, cmap='gray')
        axes[i].set_title(f'{name} Mask')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics about the segmentation
    for i, (mask, name) in enumerate(zip(class_masks, class_names)):
        coverage = np.sum(mask) / (mask.shape[0] * mask.shape[1]) * 100
        print(f"{name}: {coverage:.2f}% of the image")

def visualize_raw_predictions(prediction):
    """
    Visualize raw prediction probabilities for each class
    
    Args:
        prediction: Raw model prediction [1, height, width, num_classes]
    """
    pred = prediction[0]  # Remove batch dimension
    num_classes = pred.shape[-1]
    
    fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))
    
    # Handle case of a single class
    if num_classes == 1:
        axes = [axes]
    
    for i in range(num_classes):
        im = axes[i].imshow(pred[..., i], cmap='viridis')
        axes[i].set_title(f'Class {i} Probability')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def try_different_preprocessing(model, image_path, target_size=(512, 512)):
    """
    Try different preprocessing methods to diagnose issues
    
    Args:
        model: Trained model
        image_path: Path to image
        target_size: Target size for resizing
    """
    normalization_methods = ['standard', 'imagenet', 'none']
    
    for method in normalization_methods:
        print(f"\nTrying normalization method: {method}")
        
        # Preprocess with current method
        preprocessed_img, original_img = preprocess_image(
            image_path, 
            target_size=target_size, 
            normalization=method
        )
        
        # Perform inference with debug info
        prediction = perform_inference(model, preprocessed_img, debug=True)
        
        # Process prediction
        segmentation_mask, class_masks, confidence_map = process_prediction(
            prediction, 
            confidence_threshold=0.0  # No threshold to see raw predictions
        )
        
        # Visualize raw predictions
        print("\nRaw class probabilities:")
        visualize_raw_predictions(prediction)
        
        # Visualize results
        print("\nSegmentation results:")
        visualize_results(original_img, segmentation_mask, class_masks, confidence_map)

def main(debug_mode=True):
    """
    Main function with debug_mode option
    
    Args:
        debug_mode: Whether to run in debug mode with additional diagnostics
    """
    # Path to your trained model
    model_path = './model/unet_resnet50_multiclass.h5'
    
    # Path to input image for inference
    image_path = "../test/image/1.png"
    
    # Class names for better visualization
    class_names = ["Background", "Feature 1", "Feature 2"]  # Replace with your actual class names
    
    # Class colors for visualization (RGB format)
    class_colors = [
        [255, 0, 0],    # Red for class 0
        [0, 255, 0],    # Green for class 1
        [0, 0, 255]     # Blue for class 2
    ]
    
    # Make sure model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Make sure image directory exists
    if not os.path.exists(os.path.dirname(image_path)):
        print(f"Warning: Image directory {os.path.dirname(image_path)} does not exist. Please check the path.")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} does not exist. Please check the path.")
        return
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Warning: Image file {image_path} does not exist. Please check the path.")
        return
    
    # Load the trained model
    try:
        model = load_trained_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return
    
    if debug_mode:
        # Try different preprocessing methods to diagnose issues
        try_different_preprocessing(model, image_path)
    else:
        # Standard flow
        # Preprocess the input image
        preprocessed_img, original_img = preprocess_image(image_path)
        
        # Perform inference
        prediction = perform_inference(model, preprocessed_img)
        
        # Process the prediction with lower confidence threshold for diagnostics
        segmentation_mask, class_masks, confidence_map = process_prediction(
            prediction, confidence_threshold=0.3
        )
        
        # Visualize the results
        visualize_results(
            original_img, 
            segmentation_mask, 
            class_masks, 
            confidence_map,
            class_colors=class_colors,
            class_names=class_names
        )


if __name__ == "__main__":
    # Run with debug mode enabled to diagnose issues
    main(debug_mode=True)