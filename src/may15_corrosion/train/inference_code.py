import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

# --- Load Model ---
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

model = load_model('./models/best_model_bce_dice_loss.h5', custom_objects={'dice_loss': dice_loss, 'bce_dice_loss': bce_dice_loss})
# model = load_model('./models/best_model_transferLearning.h5', custom_objects={'dice_loss': dice_loss, 'bce_dice_loss': bce_dice_loss})
# model = load_model('best_model_binary_crossEntropy.h5')
# --- Constants ---
IMAGE_SIZE = 256

# --- Image Preprocessing ---
def preprocess_image(image):
    image = cv2.rotate(image, cv2.ROTATE_180)  # Rotate 180 degrees
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0), image  # Return preprocessed + resized original for visualization

# --- Predict Mask ---
def predict_mask(model, image_array, threshold=0.5):
    prediction = model.predict(image_array)[0, :, :, 0]  # (H, W)
    mask = (prediction > threshold).astype(np.uint8) * 255
    return mask

# --- Visualize ---
def show_overlay(original_rgb, predicted_mask, x, y):
    plt.figure(figsize=(12, 5))

    # preprocessing before display
    original_rgb_copy = cv2.resize(original_rgb, (y, x))
    original_rgb_copy = cv2.rotate(original_rgb_copy, cv2.ROTATE_180)

    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb_copy)
    plt.title("Input image")

    # plt.subplot(1, 3, 2)
    # plt.imshow(predicted_mask, cmap='gray')
    # plt.title("Predicted Mask (Corrosion)")

    # Overlay mask on image
    # overlay = original_rgb.copy()

    # This is new changes
    original_uint8 = (original_rgb * 255).astype(np.uint8)
    overlay = original_uint8.copy()

        # Apply neon green overlay (with alpha blending)
    neon_color = np.array([57, 255, 50], dtype=np.uint8)
    alpha = 0.5  # transparency: 0 = fully original, 1 = fully neon
    
        # Where mask is present
    mask_indices = predicted_mask > 0
    overlay[mask_indices] = (
        alpha * neon_color + (1 - alpha) * overlay[mask_indices]
    ).astype(np.uint8)

    # Resize back to original size
    overlay = cv2.rotate(overlay, cv2.ROTATE_180)
    overlay = cv2.resize(overlay, (y, x))

    # /This is new changes

    # overlay[predicted_mask > 0] = [255, 0, 0]  # Red overlay
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Ouput with mask")

    plt.tight_layout()
    plt.show()

# --- Run Example ---
# image_path = r"D:\NML ML Works\Testing"
image_path = r"D:\NML 2nd working directory\Dr. sarma paswan-05-06-2025\Actual\SS\CROPED"
# image_path = os.path.join(image_path, "SS_ME_3399.jpg")
image_path = os.path.join(image_path, "SS_As_3395.jpg")
image = cv2.imread(image_path)
x, y, _ = image.shape
input_tensor, resized_img = preprocess_image(image)
mask = predict_mask(model, input_tensor)
show_overlay(resized_img, mask, x, y)
