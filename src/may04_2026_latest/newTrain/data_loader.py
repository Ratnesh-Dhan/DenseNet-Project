"""
data_loader.py — Paired image/mask loader for binary corrosion segmentation
Masks: red area (#FF0000 or similar) on pure black background (PNG)
Images: JPEG or PNG, raw pixel values [0, 255]  (ResNet preprocess is in the model)
"""

import tensorflow as tf
import os

IMG_SIZE = (256, 256)


# ─────────────────────────────────────────────
#  CORE LOADER
# ─────────────────────────────────────────────

def load_image_mask(image_path, mask_path):
    # ── Image ────────────────────────────────────────────────────────────
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)          # bilinear (default) — fine for images
    image = tf.cast(image, tf.float32)                # keep raw [0, 255]; model preprocesses

    # ── Mask ─────────────────────────────────────────────────────────────
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)      # always PNG; 3-channel RGB
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')  # nearest — never interpolate labels
    mask = tf.cast(mask, tf.float32)

    # Red-channel extraction:
    #   red pixel  → R≈255, G≈0,  B≈0  → red_channel >> green_channel
    #   black pixel → R=0,   G=0,  B=0
    # Condition: R channel is substantially brighter than G and B combined.
    # This is robust to slight colour variations in the mask (e.g. #FF1010).
    r = mask[..., 0:1]   # shape (H, W, 1)
    g = mask[..., 1:2]
    b = mask[..., 2:3]
    mask = tf.cast((r > 50) & (r > g * 2) & (r > b * 2), tf.float32)
    # → values are exactly 0.0 or 1.0, shape (H, W, 1)

    return image, mask


# ─────────────────────────────────────────────
#  AUGMENTATION  (training only)
# ─────────────────────────────────────────────

def augment(image, mask):
    """
    All spatial transforms applied identically to image AND mask using a
    single shared random seed per operation — no desync risk.
    Colour jitter applied to image only (never the mask).
    """
    # ── Shared spatial augmentations ─────────────────────────────────────
    # Concatenate along channel axis so both see the same random op
    combined = tf.concat([image, mask], axis=-1)   # (H, W, 4)

    # Random horizontal flip
    combined = tf.image.random_flip_left_right(combined)
    # Random vertical flip
    combined = tf.image.random_flip_up_down(combined)

    # Random 90° rotation (k ∈ {0,1,2,3})
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    combined = tf.image.rot90(combined, k)

    # Split back
    image = combined[..., :3]
    mask  = combined[..., 3:]   # (H, W, 1)

    # ── Image-only colour jitter ──────────────────────────────────────────
    image = tf.image.random_brightness(image, max_delta=30.0)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 255.0)  # keep in valid range

    return image, mask


# ─────────────────────────────────────────────
#  DATASET BUILDER
# ─────────────────────────────────────────────

def get_dataset(image_dir, mask_dir, batch_size=8, training=True):
    """
    Args:
        image_dir:  folder with .jpg / .jpeg / .png images
        mask_dir:   folder with .png masks (red-on-black)
        batch_size: samples per batch
        training:   if True → shuffle + augment; if False → deterministic, no augment
    Returns:
        tf.data.Dataset yielding (image, mask) batches
        image: float32 [0,255]  shape (B, 256, 256, 3)
        mask:  float32 {0,1}    shape (B, 256, 256, 1)
    """
    VALID_EXTS = ('.jpg', '.jpeg', '.png')

    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(VALID_EXTS)
    ])
    mask_paths = sorted([
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir)
        if f.lower().endswith(VALID_EXTS)
    ])

    # ── Safety check ─────────────────────────────────────────────────────
    assert len(image_paths) > 0, f"No images found in {image_dir}"
    assert len(mask_paths)  > 0, f"No masks found in {mask_dir}"
    assert len(image_paths) == len(mask_paths), (
        f"Image/mask count mismatch: {len(image_paths)} images vs {len(mask_paths)} masks.\n"
        f"First image: {os.path.basename(image_paths[0])}\n"
        f"First mask:  {os.path.basename(mask_paths[0])}"
    )

    # Optional: verify stems match (catches misaligned sorted lists)
    for img_p, msk_p in zip(image_paths[:5], mask_paths[:5]):
        img_stem = os.path.splitext(os.path.basename(img_p))[0]
        msk_stem = os.path.splitext(os.path.basename(msk_p))[0]
        assert img_stem == msk_stem, (
            f"Filename mismatch: image '{img_stem}' vs mask '{msk_stem}'. "
            "Make sure images and masks share the same base filename."
        )

    print(f"[DataLoader] {'Train' if training else 'Val'} — "
          f"{len(image_paths)} image/mask pairs  |  batch_size={batch_size}")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    if training:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    dataset = dataset.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    # Cache after augment for val; for training, cache before augment so
    # augmentation runs fresh every epoch (more variety).
    # Layout: train → shuffle → load → augment → batch → prefetch  (no cache — augment varies)
    #         val   → load → cache → batch → prefetch               (cache — deterministic)
    if not training:
        dataset = dataset.cache()   # safe: val set is fixed

    dataset = dataset.batch(batch_size, drop_remainder=training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ─────────────────────────────────────────────
#  QUICK VISUAL SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import sys

    img_dir  = sys.argv[1] if len(sys.argv) > 2 else "/mnt/z/DATASETS/kaggle_semantic_segmentation_CORROSION_dataset/train/images"
    mask_dir = sys.argv[2] if len(sys.argv) > 2 else "/mnt/z/DATASETS/kaggle_semantic_segmentation_CORROSION_dataset/train/masks"

    ds = get_dataset(img_dir, mask_dir, batch_size=4, training=True)
    images, masks = next(iter(ds))
    print("Image batch:", images.shape, images.dtype,
          f"  range [{images.numpy().min():.1f}, {images.numpy().max():.1f}]")
    print("Mask batch: ", masks.shape,  masks.dtype,
          f"  unique values: {set(masks.numpy().flatten().tolist()[:200])}")

    fig, axes = plt.subplots(4, 2, figsize=(6, 12))
    for i in range(4):
        axes[i, 0].imshow(images[i].numpy().astype('uint8'))
        axes[i, 0].set_title("Image")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(masks[i, ..., 0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Mask  (corrosion={masks[i].numpy().mean():.2%})")
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.savefig("dataloader_check.png", dpi=120)
    print("Saved dataloader_check.png")