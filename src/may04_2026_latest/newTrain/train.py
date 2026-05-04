"""
train.py — Two-phase training for UNet-ResNet50 corrosion segmentation
───────────────────────────────────────────────────────────────────────
Phase 1  Encoder frozen.  Train decoder with Adam(1e-4).
Phase 2  Unfreeze conv4+conv5. Fine-tune with Adam(1e-5).
"""

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[INFO] {len(gpus)} GPU(s) found.")
else:
    print("[WARN] No GPU found — running on CPU.")

from corrosion_model import (
    build_unet_with_resnet50,
    combined_loss,
    iou_metric,
)
from data_loader import get_dataset
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
base_dir       = r"/mnt/z/DATASETS/kaggle_semantic_segmentation_CORROSION_dataset"
model_path     = "./newModel"
plot_save_path = "./newPlots"
os.makedirs(model_path,     exist_ok=True)
os.makedirs(plot_save_path, exist_ok=True)

BEST_PHASE1 = os.path.join(model_path, "best_phase1.keras")
BEST_PHASE2 = os.path.join(model_path, "best_phase2.keras")

# ── Datasets ──────────────────────────────────────────────────────────────────
train_ds = get_dataset(os.path.join(base_dir, "train/images"),
                       os.path.join(base_dir, "train/masks"),    batch_size=4)
val_ds   = get_dataset(os.path.join(base_dir, "validate/images"),
                       os.path.join(base_dir, "validate/masks"), batch_size=4, training=False)

train_ds_ft = get_dataset(os.path.join(base_dir, "train/images"),
                          os.path.join(base_dir, "train/masks"),    batch_size=2)
val_ds_ft   = get_dataset(os.path.join(base_dir, "validate/images"),
                          os.path.join(base_dir, "validate/masks"), batch_size=2, training=False)

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Transfer Learning (frozen encoder)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE 1: Transfer Learning (encoder frozen)")
print("="*60 + "\n")

model = build_unet_with_resnet50(input_shape=(256, 256, 3), compile_model=True)

callbacks_p1 = [
    EarlyStopping(monitor='val_loss', patience=8,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(BEST_PHASE1, monitor='val_loss',
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                      patience=4, min_lr=1e-7, verbose=1),
    TensorBoard(log_dir='./logs/phase1'),
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=80,
    callbacks=callbacks_p1,
)

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2a — Fine-tune conv5 only  (gentle warm-up, 15 epochs)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE 2a: Fine-tuning conv5 only")
print("="*60 + "\n")

model = build_unet_with_resnet50(input_shape=(256, 256, 3), compile_model=False)
model.load_weights(BEST_PHASE1)

# Unfreeze ONLY conv5 — leave conv4 frozen for now
# BatchNorm always stays frozen (protects ImageNet statistics on small dataset)
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    elif "conv5_block" in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False  # decoder stays frozen too — only conv5 moves

# Re-enable decoder (it must train, only encoder conv1-4 is frozen)
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False   # BN always frozen — this overrides all above
    elif any(tag in layer.name for tag in (
        "conv2d_transpose", "concatenate", "conv2d", "spatial_dropout",
        "re_lu", "batch_normalization", "output_mask", "resnet_preprocess"
    )):
        # Keep decoder trainable — only freeze encoder conv1-4
        pass

# Cleaner approach: freeze by layer group explicitly
for layer in model.layers:
    layer.trainable = True   # start all trainable

for layer in model.layers:
    # Always freeze BN — no exceptions
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    # Freeze early encoder blocks (conv1, conv2, conv3, conv4)
    elif any(tag in layer.name for tag in (
        "conv1_pad", "conv1_conv", "conv1_bn", "conv1_relu",
        "pool1_", "conv2_block", "conv3_block", "conv4_block",
    )):
        layer.trainable = False
    # conv5_block and all decoder layers remain trainable

n_trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
print(f"[Phase 2a] Trainable params: {n_trainable:,}  (conv5 + decoder)")

# Use batch_size=4 — larger batches = more stable gradients during fine-tuning
train_ds_ft4 = get_dataset(os.path.join(base_dir, "train/images"),
                            os.path.join(base_dir, "train/masks"),
                            batch_size=4)
val_ds_ft4   = get_dataset(os.path.join(base_dir, "validate/images"),
                            os.path.join(base_dir, "validate/masks"),
                            batch_size=4, training=False)

model.compile(
    # clipnorm=1.0 prevents the large encoder gradients from spiking val metrics
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6, clipnorm=1.0),
    loss=combined_loss,
    metrics=[
        iou_metric,
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
    ]
)

BEST_PHASE2A = os.path.join(model_path, "best_phase2a.keras")

callbacks_p2a = [
    EarlyStopping(monitor='val_iou_metric', mode='max',
                  patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint(BEST_PHASE2A, monitor='val_iou_metric',
                    mode='max', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_iou_metric', mode='max', factor=0.5,
                      patience=3, min_lr=1e-8, verbose=1),
    TensorBoard(log_dir='./logs/phase2a'),
]

history2a = model.fit(
    train_ds_ft4,
    validation_data=val_ds_ft4,
    epochs=15,
    callbacks=callbacks_p2a,
)

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2b — Fine-tune conv4 + conv5  (deeper unlock, very low LR)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE 2b: Fine-tuning conv4 + conv5")
print("="*60 + "\n")

model = build_unet_with_resnet50(input_shape=(256, 256, 3), compile_model=False)
model.load_weights(BEST_PHASE2A)

for layer in model.layers:
    layer.trainable = True

for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    elif any(tag in layer.name for tag in (
        "conv1_pad", "conv1_conv", "conv1_bn", "conv1_relu",
        "pool1_", "conv2_block", "conv3_block",
    )):
        layer.trainable = False   # conv1-3 still frozen

n_trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
print(f"[Phase 2b] Trainable params: {n_trainable:,}  (conv4 + conv5 + decoder)")

model.compile(
    # Half the LR of 2a — conv4 is less adapted to our task than conv5
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-6, clipnorm=1.0),
    loss=combined_loss,
    metrics=[
        iou_metric,
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
    ]
)

BEST_PHASE2B = os.path.join(model_path, "best_phase2b.keras")

callbacks_p2b = [
    EarlyStopping(monitor='val_iou_metric', mode='max',
                  patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint(BEST_PHASE2B, monitor='val_iou_metric',
                    mode='max', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_iou_metric', mode='max', factor=0.5,
                      patience=3, min_lr=1e-9, verbose=1),
    TensorBoard(log_dir='./logs/phase2b'),
]

history2b = model.fit(
    train_ds_ft4,
    validation_data=val_ds_ft4,
    epochs=15,
    callbacks=callbacks_p2b,
)

model.save(os.path.join(model_path, "final_model.keras"))
print("\n✔  Final model saved.\n")

# alias for plotting
history2 = history2b

model.save(os.path.join(model_path, "final_model.keras"))
print("\n✔  Final model saved.\n")


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

def plot_history(history, tag):
    h      = history.history
    epochs = range(1, len(h['loss']) + 1)
    keys   = [k for k in ('iou_metric', 'recall', 'precision', 'accuracy') if k in h]
    n      = 1 + len(keys)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    axes[0].plot(epochs, h['loss'],     label='Train')
    axes[0].plot(epochs, h['val_loss'], label='Val', linestyle='--')
    axes[0].set_title(f'{tag} — Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()

    for ax, key in zip(axes[1:], keys):
        ax.plot(epochs, h[key],           label='Train')
        ax.plot(epochs, h[f'val_{key}'],  label='Val', linestyle='--')
        ax.set_title(f'{tag} — {key}'); ax.set_xlabel('Epoch'); ax.legend()

    plt.tight_layout()
    out = os.path.join(plot_save_path, f'{tag}_history.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"[INFO] Plot saved → {out}")


plot_history(history1,  "phase1_transfer")
plot_history(history2a, "phase2a_conv5only")
plot_history(history2b, "phase2b_conv4and5")