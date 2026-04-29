import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import shutil

# ---------------------------
# CONFIG
# ---------------------------
IMG_SIZE = (512, 512)
MODEL_PATH = "./model_new/best_model_transferLearning.keras"
OUTPUT_DIR = "./inference_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Load custom objects
# ---------------------------
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    return 1 - tf.reduce_mean(
        (2. * intersection + smooth) /
        (tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) + smooth)
    )

def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-6)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

def focal_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    focal = tf.keras.losses.BinaryFocalCrossentropy(gamma=2)(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + focal + dice

# ---------------------------
# Load model
# ---------------------------
model = load_model(
    MODEL_PATH,
    custom_objects={
        "combined_loss": focal_dice_loss,
        "iou_metric": iou_metric
    },
    compile=False
)

print("Model loaded.")

# ---------------------------
# Preprocess image
# ---------------------------
def preprocess_image(path):
    img = cv2.imread(path)
    orig = img.copy()

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)
    img = preprocess_input(img)

    return img, orig

# ---------------------------
# Predict
# ---------------------------
def predict_image(image_path):
    img, orig = preprocess_image(image_path)

    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]

    # Convert to binary mask
    mask = (pred > 0.05).astype(np.uint8) * 255

    # Resize back to original size
    mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))

    return orig, mask

# ---------------------------
# Overlay function (optional)
# ---------------------------
def overlay_mask(image, mask):
    overlay = image.copy()
    overlay[mask == 255] = [0, 0, 255]  # red overlay

    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

# ---------------------------
# Run inference on folder
# ---------------------------
INPUT_DIR = "./test_images"   # change this

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(INPUT_DIR, fname)

    image, mask = predict_image(path)
    overlay = overlay_mask(image, mask)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname}_mask.png"), mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname}_overlay.png"), overlay)
    shutil.copy(path, OUTPUT_DIR)

    print(f"Processed: {fname}")

print("Done.")