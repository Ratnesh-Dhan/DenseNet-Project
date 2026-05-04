import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---- load model (IMPORTANT: custom objects) ----
from tensorflow.keras.models import load_model

# re-import your losses/metrics
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denominator  = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    return 1.0 - tf.reduce_mean((2.0 * intersection + smooth) / (denominator + smooth))

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    focal = alpha_t * tf.pow(1.0 - p_t, gamma) * bce
    return tf.reduce_mean(focal)

def combined_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred, threshold=0.5):
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) - intersection
    return (intersection + 1e-6) / (union + 1e-6)

# ---- load trained model ----
model = load_model(
    "../model/final_model.keras",   # change path
    custom_objects={
        "combined_loss": combined_loss,
        "iou_metric": iou_metric
    },
    compile=False,
    safe_mode=False
)

# ---- preprocess ----
def preprocess_image(img_path, size=(256, 256)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = img.copy()

    img = cv2.resize(img, size)
    img = img.astype("float32")  # DO NOT normalize (ResNet preprocess is inside model)

    img = np.expand_dims(img, axis=0)  # (1, H, W, 3)
    return img, original

# ---- inference ----
def predict_mask(img_path, threshold=0.5):
    img, original = preprocess_image(img_path)

    pred = model.predict(img)[0]   # (256,256,1)
    pred = pred.squeeze()

    # threshold
    mask = (pred > threshold).astype(np.uint8)

    return original, pred, mask

# ---- visualize ----
def show_results(img_path):
    original, pred, mask = predict_mask(img_path)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Raw Prediction")
    plt.imshow(pred, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Binary Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.show()

# ---- run ----
show_results("./images/0.jpeg")