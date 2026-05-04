"""
train.py — Two-phase training for UNet-ResNet50 corrosion segmentation
───────────────────────────────────────────────────────────────────────
Phase 1  (transfer learning)
    Encoder fully frozen.  Train only the decoder with Adam(1e-4).
    Early-stop on val_loss, patience=8.

Phase 2  (fine-tuning)
    Unfreeze ResNet50 conv4_block* and conv5_block* layers.
    Train with Adam(1e-5).  Early-stop on val_iou_metric, patience=7.
"""

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import tensorflow as tf

# ── GPU memory growth (do this BEFORE any model is built) ───────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[INFO] {len(gpus)} GPU(s) available: {[g.name for g in gpus]}")
else:
    print("[WARN] No GPU found — running on CPU.")

# ── Imports ─────────────────────────────────────────────────────────────────
from model import (
    build_unet_with_resnet50,
    combined_loss,
    iou_metric,
)
from dataloader import get_dataset
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
import matplotlib
matplotlib.use('Agg')          # headless — no display needed
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
base_dir        = r"/mnt/z/DATASETS/kaggle_semantic_segmentation_CORROSION_dataset"
model_path      = "../model"
plot_save_path  = "../plots"
os.makedirs(model_path,     exist_ok=True)
os.makedirs(plot_save_path, exist_ok=True)

BEST_PHASE1 = os.path.join(model_path, "best_phase1.keras")
BEST_PHASE2 = os.path.join(model_path, "best_phase2.keras")

# ── Datasets ─────────────────────────────────────────────────────────────────
BATCH_P1 = 4
BATCH_P2 = 2    # smaller batch for fine-tuning (more gradient noise helps)

train_ds = get_dataset(
    os.path.join(base_dir, "train/images"),
    os.path.join(base_dir, "train/masks"),
    batch_size=BATCH_P1,
)
val_ds = get_dataset(
    os.path.join(base_dir, "validate/images"),
    os.path.join(base_dir, "validate/masks"),
    batch_size=BATCH_P1,
)
train_ds_ft = get_dataset(
    os.path.join(base_dir, "train/images"),
    os.path.join(base_dir, "train/masks"),
    batch_size=BATCH_P2,
)
val_ds_ft = get_dataset(
    os.path.join(base_dir, "validate/images"),
    os.path.join(base_dir, "validate/masks"),
    batch_size=BATCH_P2,
)

# ════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Transfer Learning (frozen encoder)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE 1: Transfer Learning (encoder frozen)")
print("="*60 + "\n")

model = build_unet_with_resnet50(input_shape=(256, 256, 3), compile_model=True)
# compile_model=True → Adam(1e-4) + combined_loss (focal + dice)

trainable_p1 = sum(tf.size(w).numpy() for w in model.trainable_weights)
total_p1     = sum(tf.size(w).numpy() for w in model.weights)
print(f"[INFO] Trainable params: {trainable_p1:,} / {total_p1:,}")

callbacks_p1 = [
    EarlyStopping(
        monitor='val_loss', patience=8,
        restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        BEST_PHASE1, monitor='val_loss',
        save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.3,
        patience=4, min_lr=1e-7, verbose=1
    ),
    TensorBoard(log_dir='./logs/phase1', histogram_freq=0),
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=80,
    callbacks=callbacks_p1,
)

# ════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — Fine-tuning (unfreeze conv4 + conv5)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE 2: Fine-tuning (conv4 + conv5 unfrozen)")
print("="*60 + "\n")

# Reload the best phase-1 weights into a fresh model
#  (avoids any state left over from the training loop)
model = build_unet_with_resnet50(input_shape=(256, 256, 3), compile_model=False)
model.load_weights(BEST_PHASE1)

# Selective unfreezing strategy:
#   - BatchNorm stays FROZEN throughout (prevents encoder BN stats from
#     destroying the learned decoder representations on a small dataset)
#   - conv4_block* and conv5_block* are unfrozen for the encoder
#   - Everything else (conv1-3 + decoder) stays as-is
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    elif any(tag in layer.name for tag in ("conv4_block", "conv5_block")):
        layer.trainable = True
    # decoder layers keep their current trainable=True state

trainable_p2 = sum(tf.size(w).numpy() for w in model.trainable_weights)
print(f"[INFO] Trainable params after unfreeze: {trainable_p2:,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),   # must be 10× smaller
    loss=combined_loss,
    metrics=[
        iou_metric,
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
    ]
)

callbacks_p2 = [
    # Primary signal: IoU (what you actually care about)
    EarlyStopping(
        monitor='val_iou_metric', mode='max',
        patience=7, restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        BEST_PHASE2, monitor='val_iou_metric',
        mode='max', save_best_only=True, verbose=1
    ),
    # Secondary: also reduce LR on loss plateau
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.3,
        patience=3, min_lr=1e-8, verbose=1
    ),
    TensorBoard(log_dir='./logs/phase2', histogram_freq=0),
]

history2 = model.fit(
    train_ds_ft,
    validation_data=val_ds_ft,
    epochs=30,
    callbacks=callbacks_p2,
)

model.save(os.path.join(model_path, "final_model.keras"))
print("\n✔  Final model saved.\n")


# ════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def plot_history(history, tag: str):
    """Save loss + available metric curves for one training phase."""
    h = history.history
    epochs = range(1, len(h['loss']) + 1)

    metric_keys = [k for k in ('iou_metric', 'recall', 'precision', 'accuracy')
                   if k in h]
    n_plots = 1 + len(metric_keys)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # ── Loss ──
    ax = axes[0]
    ax.plot(epochs, h['loss'],     label='Train loss')
    ax.plot(epochs, h['val_loss'], label='Val loss',  linestyle='--')
    ax.set_title(f'{tag} — Loss')
    ax.set_xlabel('Epoch')
    ax.legend()

    # ── Other metrics ──
    for ax, key in zip(axes[1:], metric_keys):
        ax.plot(epochs, h[key],             label=f'Train {key}')
        ax.plot(epochs, h[f'val_{key}'],    label=f'Val {key}',  linestyle='--')
        ax.set_title(f'{tag} — {key}')
        ax.set_xlabel('Epoch')
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(plot_save_path, f'{tag}_history.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Plot saved → {out_path}")


plot_history(history1, "phase1_transfer")
plot_history(history2, "phase2_finetune")