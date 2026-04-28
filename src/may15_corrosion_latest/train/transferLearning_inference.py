import tensorflow as tf
import numpy as np
import cv2, os
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet import preprocess_input
from new_transfer_model import build_unet_with_resnet50

IMG_SIZE = (256, 256)

# Load trained model and weights
# model = build_unet_with_resnet50()
if os.path.exists('./models/best_model_transferLearning_test.h5'):
    print("\nModel's file location exists.\n")
# model.load_weights(r'.\models\best_model_transferLearning.h5')
model = tf.keras.models.load_model('./models/best_model_transferLearning_test.h5')

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # Important for ResNet
    return image

def predict_mask(image_path, save_output=True):
    # Preprocess image
    image = load_image(image_path)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    # Predict
    pred_mask = model.predict(image)[0, ..., 0]  # Remove batch and channel dims

    # Threshold the mask
    binary_mask = (pred_mask > 0.5).astype(np.uint8)

    # Load original image for display (non-preprocessed)
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, IMG_SIZE)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Overlay mask on image
    mask_overlay = np.zeros_like(original_image)
    mask_overlay[..., 0] = binary_mask * 255  # Red channel

    blended = cv2.addWeighted(original_image, 0.7, mask_overlay, 0.3, 0)

    # Show results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(binary_mask, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(blended)
    plt.axis("off")

    plt.tight_layout()
    if save_output:
        output_path = "inference_result.png"
        plt.savefig(output_path)
        print(f"Saved result to {output_path}")
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = r"D:\NML ML Works\Testing"
    # image_path = r"D:\NML ML Works\kaggle_semantic_segmentation_CORROSION_dataset\validate\images"
    image_path = os.path.join(image_path, "4.jpg")
    predict_mask(image_path)
