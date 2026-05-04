"""
inference.py — Corrosion segmentation inference & comparison
─────────────────────────────────────────────────────────────
Usage:
    # Single image
    python inference.py --image path/to/image.jpg

    # Folder of images
    python inference.py --image path/to/folder/

    # Compare both models side-by-side
    python inference.py --image path/to/image.jpg --compare

    # Save overlay outputs to a directory
    python inference.py --image path/to/folder/ --save_dir ./results/
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── Paths (edit these if your model files are elsewhere) ────────────────────
DEFAULT_PHASE1_PATH = "./model/best_phase1.keras"
DEFAULT_PHASE2_PATH = "./model/best_phase2.keras"

IMG_SIZE  = (256, 256)
THRESHOLD = 0.5          # sigmoid threshold for binary mask
OVERLAY_ALPHA = 0.45     # transparency of the red overlay
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')


# ════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_model(path: str) -> tf.keras.Model:
    """Load a saved .keras model with custom objects."""
    from corrosion_model import combined_loss, iou_metric
    print(f"[INFO] Loading model: {path}")
    model = tf.keras.models.load_model(
        path,
        custom_objects={
            "combined_loss": combined_loss,
            "iou_metric":    iou_metric,
        }
    )
    print(f"[INFO] Loaded OK — input: {model.input_shape}  output: {model.output_shape}")
    return model


# ════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING / POSTPROCESSING
# ════════════════════════════════════════════════════════════════════════════

def preprocess(image_path: str) -> tuple[np.ndarray, tuple[int,int]]:
    """
    Load an image, resize to model input, return:
        - batch tensor  float32  (1, H, W, 3)  range [0, 255]
        - original (H, W) for resizing the mask back
    """
    img = Image.open(image_path).convert("RGB")
    orig_size = (img.height, img.width)            # H, W
    img_resized = img.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)  # (256, 256, 3)
    return arr[np.newaxis, ...], orig_size          # add batch dim


def predict(model: tf.keras.Model, batch: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """
    Run inference.
    Returns:
        prob_map  float32  (H, W)   raw sigmoid probability
        bin_mask  uint8   (H, W)   thresholded binary mask  {0, 255}
    """
    prob = model.predict(batch, verbose=0)[0, ..., 0]   # (H, W)
    bin_mask = (prob >= threshold).astype(np.uint8) * 255
    return prob, bin_mask


def resize_mask_to_original(mask: np.ndarray, orig_size: tuple[int,int]) -> np.ndarray:
    """Resize mask back to original image dimensions using nearest-neighbour."""
    pil = Image.fromarray(mask)
    pil = pil.resize((orig_size[1], orig_size[0]), Image.NEAREST)  # PIL: (W, H)
    return np.array(pil)


def compute_stats(prob_map: np.ndarray, bin_mask: np.ndarray) -> dict:
    """Return basic statistics about the prediction."""
    total_px    = bin_mask.size
    corr_px     = int((bin_mask > 0).sum())
    coverage    = corr_px / total_px * 100
    mean_conf   = float(prob_map[bin_mask > 0].mean()) if corr_px > 0 else 0.0
    return {
        "total_pixels":     total_px,
        "corrosion_pixels": corr_px,
        "coverage_%":       round(coverage, 3),
        "mean_confidence":  round(mean_conf, 4),
    }


# ════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

def make_overlay(image_rgb: np.ndarray, bin_mask: np.ndarray, color=(255, 30, 30)) -> np.ndarray:
    """
    Blend a coloured overlay onto the image wherever the mask is active.
    image_rgb : uint8  (H, W, 3)
    bin_mask  : uint8  (H, W)  {0, 255}
    Returns   : uint8  (H, W, 3)
    """
    overlay = image_rgb.copy().astype(np.float32)
    mask_bool = bin_mask > 0
    for c, val in enumerate(color):
        overlay[..., c] = np.where(
            mask_bool,
            overlay[..., c] * (1 - OVERLAY_ALPHA) + val * OVERLAY_ALPHA,
            overlay[..., c]
        )
    return overlay.clip(0, 255).astype(np.uint8)


def plot_single(image_path: str, model: tf.keras.Model,
                model_label: str = "Model", save_path: str = None):
    """4-panel plot: original | probability map | binary mask | overlay."""
    batch, orig_size = preprocess(image_path)
    prob_map, bin_mask = predict(model, batch)

    # Resize outputs back to original resolution
    prob_orig = np.array(
        Image.fromarray((prob_map * 255).astype(np.uint8)).resize(
            (orig_size[1], orig_size[0]), Image.BILINEAR)) / 255.0
    mask_orig = resize_mask_to_original(bin_mask, orig_size)

    orig_img  = np.array(Image.open(image_path).convert("RGB"))
    overlay   = make_overlay(orig_img, mask_orig)
    stats     = compute_stats(prob_orig, mask_orig)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"{model_label}  |  {os.path.basename(image_path)}\n"
        f"Coverage: {stats['coverage_%']:.2f}%   "
        f"Confidence: {stats['mean_confidence']:.3f}",
        fontsize=13, fontweight='bold'
    )

    axes[0].imshow(orig_img);                    axes[0].set_title("Original Image")
    axes[1].imshow(prob_orig, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title("Probability Map")
    fig.colorbar(
        plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(0, 1)),
        ax=axes[1], fraction=0.046, pad=0.04
    )
    axes[2].imshow(mask_orig, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f"Binary Mask  (t={THRESHOLD})")
    axes[3].imshow(overlay);                     axes[3].set_title("Overlay")

    red_patch = mpatches.Patch(color=(1, 0.12, 0.12), label='Corrosion detected')
    axes[3].legend(handles=[red_patch], loc='lower right', fontsize=9)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  → Saved: {save_path}")
    else:
        plt.show()
    plt.close()

    return stats


def plot_comparison(image_path: str,
                    model_p1: tf.keras.Model, model_p2: tf.keras.Model,
                    save_path: str = None):
    """
    Side-by-side comparison of Phase-1 vs Phase-2 on a single image.
    Columns: Original | P1 mask | P1 overlay | P2 mask | P2 overlay
    """
    batch, orig_size = preprocess(image_path)

    prob1, mask1 = predict(model_p1, batch)
    prob2, mask2 = predict(model_p2, batch)

    mask1_orig = resize_mask_to_original(mask1, orig_size)
    mask2_orig = resize_mask_to_original(mask2, orig_size)

    orig_img  = np.array(Image.open(image_path).convert("RGB"))
    overlay1  = make_overlay(orig_img, mask1_orig)
    overlay2  = make_overlay(orig_img, mask2_orig)

    stats1 = compute_stats(prob1, mask1)
    stats2 = compute_stats(prob2, mask2)

    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    fig.suptitle(
        f"Phase 1 vs Phase 2  |  {os.path.basename(image_path)}",
        fontsize=13, fontweight='bold'
    )

    titles = [
        "Original",
        f"Phase 1 — Mask\nCoverage: {stats1['coverage_%']:.2f}%",
        f"Phase 1 — Overlay\nConf: {stats1['mean_confidence']:.3f}",
        f"Phase 2 — Mask\nCoverage: {stats2['coverage_%']:.2f}%",
        f"Phase 2 — Overlay\nConf: {stats2['mean_confidence']:.3f}",
    ]
    imgs = [orig_img, mask1_orig, overlay1, mask2_orig, overlay2]
    cmaps = [None, 'gray', None, 'gray', None]

    for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
        kw = dict(cmap=cmap, vmin=0, vmax=255) if cmap == 'gray' else {}
        ax.imshow(img, **kw)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  → Saved: {save_path}")
    else:
        plt.show()
    plt.close()

    return stats1, stats2


# ════════════════════════════════════════════════════════════════════════════
#  BATCH FOLDER INFERENCE
# ════════════════════════════════════════════════════════════════════════════

def run_on_folder(folder: str, model: tf.keras.Model,
                  model_label: str, save_dir: str = None,
                  compare_model: tf.keras.Model = None):
    """Run inference on every image in a folder and print a summary table."""
    paths = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(VALID_EXTS)
    ])
    if not paths:
        print(f"[WARN] No images found in {folder}")
        return

    print(f"\n[INFO] Running on {len(paths)} images in '{folder}'")
    all_stats = []

    for i, img_path in enumerate(paths, 1):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        print(f"  [{i:3d}/{len(paths)}]  {os.path.basename(img_path)}", end="  ")

        if compare_model is not None:
            out_path = os.path.join(save_dir, f"{stem}_compare.png") if save_dir else None
            s1, s2 = plot_comparison(img_path, model, compare_model, save_path=out_path)
            print(f"P1 cov={s1['coverage_%']:.2f}%  P2 cov={s2['coverage_%']:.2f}%")
            all_stats.append({"file": stem, **{f"p1_{k}": v for k, v in s1.items()},
                                              **{f"p2_{k}": v for k, v in s2.items()}})
        else:
            out_path = os.path.join(save_dir, f"{stem}_result.png") if save_dir else None
            stats = plot_single(img_path, model, model_label=model_label, save_path=out_path)
            print(f"coverage={stats['coverage_%']:.2f}%  conf={stats['mean_confidence']:.3f}")
            all_stats.append({"file": stem, **stats})

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "─"*50)
    print(f"  SUMMARY — {len(paths)} images")
    if compare_model is not None:
        avg1 = np.mean([s[f"p1_coverage_%"] for s in all_stats])
        avg2 = np.mean([s[f"p2_coverage_%"] for s in all_stats])
        print(f"  Mean corrosion coverage  Phase 1: {avg1:.2f}%   Phase 2: {avg2:.2f}%")
    else:
        avg = np.mean([s["coverage_%"] for s in all_stats])
        print(f"  Mean corrosion coverage: {avg:.2f}%")
    print("─"*50 + "\n")

    return all_stats


# ════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Corrosion segmentation inference")
    p.add_argument("--image",     required=True,
                   help="Path to an image file OR a folder of images")
    p.add_argument("--phase1",    default=DEFAULT_PHASE1_PATH,
                   help="Path to phase-1 .keras model")
    p.add_argument("--phase2",    default=DEFAULT_PHASE2_PATH,
                   help="Path to phase-2 .keras model")
    p.add_argument("--model",     choices=["phase1", "phase2", "both"], default="phase2",
                   help="Which model(s) to use  (default: phase2)")
    p.add_argument("--compare",   action="store_true",
                   help="Show Phase 1 vs Phase 2 side-by-side")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help=f"Sigmoid threshold  (default: {THRESHOLD})")
    p.add_argument("--save_dir",  default=None,
                   help="Directory to save output plots (omit to display instead)")
    return p.parse_args()


def main():
    args = parse_args()
    global THRESHOLD
    THRESHOLD = args.threshold

    # ── Load requested models ────────────────────────────────────────────
    model_p1, model_p2 = None, None
    use_compare = args.compare or args.model == "both"

    if args.model in ("phase1", "both") or use_compare:
        model_p1 = load_model(args.phase1)
    if args.model in ("phase2", "both") or use_compare:
        model_p2 = load_model(args.phase2)

    active_model  = model_p2 if model_p2 is not None else model_p1
    active_label  = "Phase 2 (fine-tuned)" if model_p2 is not None else "Phase 1"

    # ── Run inference ────────────────────────────────────────────────────
    if os.path.isdir(args.image):
        run_on_folder(
            args.image,
            model=active_model,
            model_label=active_label,
            save_dir=args.save_dir,
            compare_model=model_p1 if use_compare else None,
        )
    else:
        if not os.path.isfile(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")

        save_path = None
        if args.save_dir:
            stem = os.path.splitext(os.path.basename(args.image))[0]
            suffix = "_compare.png" if use_compare else "_result.png"
            save_path = os.path.join(args.save_dir, stem + suffix)

        if use_compare:
            plot_comparison(args.image, model_p1, model_p2, save_path=save_path)
        else:
            stats = plot_single(args.image, active_model,
                                model_label=active_label, save_path=save_path)
            print("\n[Result]")
            for k, v in stats.items():
                print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()