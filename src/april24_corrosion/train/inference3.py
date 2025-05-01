import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_predict(model_path, image_path):
    # Load model
    model = load_model(model_path)
    print(f"Model loaded: input shape {model.input_shape}, output shape {model.output_shape}")
    
    # Load and preprocess image - USING IMAGENET NORMALIZATION
    img = load_img(image_path, target_size=(512, 512))
    img_array = img_to_array(img)
    img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # Saving the original image for matplot show
    og_image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
    og_image = cv2.resize(og_image, (512, 512))
    
    # Run prediction
    prediction = model.predict(img_batch)
    
    # Get segmentation mask
    segmentation_mask = np.argmax(prediction[0], axis=-1)
    
    # Show results
    display_results(img_array, segmentation_mask, prediction[0], og_image=og_image)
    
    return segmentation_mask, prediction

def display_results(original, mask, pred_prob, og_image):
    # Define colors for visualization
    colors = [
        [255, 0, 0],    # Red for class 0
        [0, 255, 0],    # Green for class 1
        [0, 0, 255]     # Blue for class 2
    ]
    
    # Create colored mask
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        colored_mask[mask == i] = color
    
    # Display
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # plt.imshow(original.astype('uint8'))
    plt.imshow(og_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(colored_mask)
    plt.title('Segmentation Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    load_and_predict(
        model_path='./model/unet_resnet50_multiclass.h5',
        image_path='../test/image/corrosion2.jpg'
    )