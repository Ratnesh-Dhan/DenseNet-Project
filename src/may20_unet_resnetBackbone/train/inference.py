import os
import cv2
import numpy as np
import tensorflow as tf
from model import build_unet   # same file you used for training

# ---- CONFIG ----
MODEL_PATH = "../models/unet_resnet50_corrosion_office_version.keras"
IMG_SIZE = (256, 256)
THRESHOLD = 0.5   # adjust if predictions look weak

# ---- LOAD MODEL ----
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (numerator + smooth) / (denominator + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_loss": dice_loss
    }
)

print("Model loaded.")

# ---- PREPROCESS ----
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    original = img.copy()

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0

    return img, original

# ---- POSTPROCESS ----
def postprocess_mask(pred_mask):
    pred_mask = pred_mask.squeeze()  # (256,256,1) -> (256,256)

    # threshold
    pred_mask = (pred_mask > THRESHOLD).astype(np.uint8)

    return pred_mask * 255  # for visualization

# ---- INFERENCE ----
def predict_image(img_path, save_path=None):
    img, original = preprocess_image(img_path)

    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]

    mask = postprocess_mask(pred)

    # resize mask back to original size
    mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))

    # overlay (optional)
    overlay = original.copy()
    overlay[mask_resized > 0] = [0, 0, 255]  # red corrosion

    if save_path:
        cv2.imwrite(save_path, mask_resized)

    return original, mask_resized, overlay

def resize_for_display(img, max_width=800, max_height=600):
    h, w = img.shape[:2]

    scale = min(max_width / w, max_height / h)
    if scale >= 1:
        return img  # already small

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h))

# ---- RUN ----
if __name__ == "__main__":
    path_ = "../../may15_corrosion_latest/train/test_images/"
    images = os.listdir(path_)
    for i in images[:10]:
        test_img = os.path.join(path_, i)   # put your test image here

        original, mask, overlay = predict_image(test_img)

        cv2.imshow("Original", resize_for_display(original))
        cv2.imshow("Mask", resize_for_display(mask))
        cv2.imshow("Overlay", resize_for_display(overlay))
        cv2.waitKey(0)
        cv2.destroyAllWindows()