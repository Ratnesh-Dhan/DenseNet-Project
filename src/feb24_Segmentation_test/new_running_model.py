import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

model = tf.keras.models.load_model("mask_rcnn_model.keras")
image_size = 256 # 512
def load_and_process_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size))
    image_array = np.array(image)/255.0  # Normalizing
    image_batch = np.expand_dims(image_array, axis=0)  # Adding batch dimension
    return image_array, image_batch  # Return both original and batched versions

def visualize(original_img, predicted_mask):
    plt.figure(figsize=(12, 5))
    
    # Display original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")
    
    # Display raw prediction (most probable class)
    plt.subplot(1, 3, 2)
    mask_display = np.argmax(predicted_mask, axis=-1)
    plt.imshow(mask_display, cmap='tab20')
    plt.title("Segmentation Mask (Class Indices)")
    plt.axis("off")
    
    # Display colored overlay
    plt.subplot(1, 3, 3)
    # Create a colored overlay using the class probabilities
    mask_overlay = np.zeros_like(original_img)
    for i in range(predicted_mask.shape[-1]):
        # Add color for each class based on probability
        color = plt.cm.tab20(i % 20)[:3]  # Get RGB from colormap
        for c in range(3):
            mask_overlay[:,:,c] += predicted_mask[:,:,i] * color[c]
    
    # Overlay the mask on the original image
    blended = 0.7 * original_img + 0.3 * mask_overlay
    plt.imshow(np.clip(blended, 0, 1))
    plt.title("Segmentation Overlay")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "../../Datasets/testDataset/img/2007_007948.jpg"
    original_img, image_batch = load_and_process_image(image_path)
    
    # Make prediction
    predicted_mask_batch = model.predict(image_batch)
    predicted_mask = predicted_mask_batch[0]  # Remove batch dimension
    
    print(f"Image shape: {original_img.shape}")
    print(f"Prediction shape: {predicted_mask.shape}")
    
    # Visualize the results
    visualize(original_img, predicted_mask)