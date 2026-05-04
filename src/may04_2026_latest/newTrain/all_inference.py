"""
inference.py — Corrosion segmentation inference
Edit the CONFIG section and run:  python inference.py
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these two lines
# ════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "./newModel/best_phase1.keras"   # or best_phase1 / best_phase2a / best_phase2b
FOLDER_PATH = "/mnt/z/DATASETS/randomRoofCorrosion"
saveNewPath = "/mnt/z/DATASETS/RESULTS3"

os.makedirs(saveNewPath, exist_ok=True)

THRESHOLD = 0.5   # lower → detect more  |  raise → be stricter

# ════════════════════════════════════════════════════════════════════════════

IMG_SIZE = (256, 256)

# Import the registered custom objects — just importing the module is enough
# because @register_keras_serializable decorators fire on import.
import corrosion_model  # noqa: F401  (registers ResNetPreprocess, combined_loss, iou_metric)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)   # no custom_objects needed
print(f"Loaded  — input {model.input_shape}  output {model.output_shape}")


images = os.listdir(FOLDER_PATH)
for img in images:
    # ── Load & preprocess image ───────────────────────────────────────────────────
    orig    = Image.open(os.path.join(FOLDER_PATH, img)).convert("RGB")
    orig_np = np.array(orig)

    inp = np.array(orig.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR), dtype=np.float32)
    inp = inp[np.newaxis, ...]   # (1, 256, 256, 3)

    # ── Predict ───────────────────────────────────────────────────────────────────
    prob = model.predict(inp, verbose=0)[0, ..., 0]      # (256, 256) float32
    mask = (prob >= THRESHOLD).astype(np.uint8) * 255    # (256, 256) {0, 255}

    # Resize back to original resolution
    H, W = orig_np.shape[:2]
    prob_orig = np.array(
        Image.fromarray((prob * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ) / 255.0
    mask_orig = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST))

    # ── Red overlay ───────────────────────────────────────────────────────────────
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
    fig.suptitle(
        f"{img}  |  coverage: {coverage:.2f}%  conf: {mean_conf:.3f}",
        fontsize=12
    )
    axes[0].imshow(orig_np);                                   axes[0].set_title("Original")
    axes[1].imshow(prob_orig, cmap='hot', vmin=0, vmax=1);     axes[1].set_title("Probability Map")
    axes[2].imshow(mask_orig, cmap='gray', vmin=0, vmax=255);  axes[2].set_title(f"Mask  (t={THRESHOLD})")
    axes[3].imshow(overlay);                                   axes[3].set_title("Overlay")
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(saveNewPath, img), dpi=150, bbox_inches='tight')
    # plt.show()
    print("Saved → result.png")