"""
inference.py — Corrosion segmentation inference
Edit the CONFIG section below and run:  python inference.py
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these
# ════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "../model/best_phase1.keras"   # or best_phase1.keras
IMAGE_PATH = "./images/0.jpeg"

THRESHOLD  = 0.5    # lower (e.g. 0.3) to detect more, raise (0.7) to be stricter

# ════════════════════════════════════════════════════════════════════════════

IMG_SIZE = (256, 256)

# ── Loss / metric definitions (must match what the model was trained with) ───
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred,                       [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denominator  = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    return 1.0 - tf.reduce_mean((2.0 * intersection + smooth) / (denominator + smooth))

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true  = tf.cast(y_true, tf.float32)
    bce     = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)

def combined_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred, threshold=0.5):
    y_pred_bin   = tf.cast(y_pred > threshold, tf.float32)
    y_true       = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin)
    union        = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) - intersection
    return (intersection + 1e-6) / (union + 1e-6)

# ── Load model ────────────────────────────────────────────────────────────────
# enable_unsafe_deserialization allows loading Lambda layers saved by Keras
# This is safe because YOU trained this model yourself
tf.keras.config.enable_unsafe_deserialization()

print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "combined_loss": combined_loss,
        "iou_metric":    iou_metric,
    }
)
print("Model loaded.")

# ── Load & preprocess image ───────────────────────────────────────────────────
orig    = Image.open(IMAGE_PATH).convert("RGB")
orig_np = np.array(orig)

inp = np.array(orig.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR), dtype=np.float32)
inp = inp[np.newaxis, ...]   # (1, 256, 256, 3)

# ── Predict ───────────────────────────────────────────────────────────────────
prob = model.predict(inp, verbose=0)[0, ..., 0]          # (256, 256)  float32
mask = (prob >= THRESHOLD).astype(np.uint8) * 255        # (256, 256)  {0, 255}

# Resize outputs back to original resolution
prob_orig = np.array(
    Image.fromarray((prob * 255).astype(np.uint8)).resize(
        (orig_np.shape[1], orig_np.shape[0]), Image.BILINEAR
    )
) / 255.0

mask_orig = np.array(
    Image.fromarray(mask).resize(
        (orig_np.shape[1], orig_np.shape[0]), Image.NEAREST
    )
)

# ── Overlay ───────────────────────────────────────────────────────────────────
overlay = orig_np.copy().astype(np.float32)
m = mask_orig > 0
overlay[m, 0] = overlay[m, 0] * 0.55 + 255 * 0.45
overlay[m, 1] = overlay[m, 1] * 0.55
overlay[m, 2] = overlay[m, 2] * 0.55
overlay = overlay.clip(0, 255).astype(np.uint8)

# ── Stats ─────────────────────────────────────────────────────────────────────
coverage  = (mask_orig > 0).sum() / mask_orig.size * 100
mean_conf = float(prob_orig[mask_orig > 0].mean()) if (mask_orig > 0).any() else 0.0
print(f"Corrosion coverage : {coverage:.2f}%")
print(f"Mean confidence    : {mean_conf:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(f"{IMAGE_PATH}  |  coverage: {coverage:.2f}%  conf: {mean_conf:.3f}", fontsize=12)

axes[0].imshow(orig_np);                                   axes[0].set_title("Original")
axes[1].imshow(prob_orig, cmap='hot', vmin=0, vmax=1);     axes[1].set_title("Probability Map")
axes[2].imshow(mask_orig, cmap='gray', vmin=0, vmax=255);  axes[2].set_title(f"Mask  (t={THRESHOLD})")
axes[3].imshow(overlay);                                   axes[3].set_title("Overlay")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig("result.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → result.png")