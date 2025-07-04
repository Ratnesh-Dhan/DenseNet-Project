import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# --- Paths ---
base_path = r"D:\NML 2nd working directory\corrosion sample piece"
image_dir = os.path.join(base_path, "dataset/images")
annotation_dir = os.path.join(base_path, "dataset/annotations")
aug_image_dir = os.path.join(base_path, "augmented/images")
aug_anno_dir = os.path.join(base_path, "augmented/annotations")

os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_anno_dir, exist_ok=True)

# --- Albumentations Transform ---
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),
    A.Affine(
        translate_percent={"x": 0.05, "y": 0.05},
        scale=(0.9, 1.1),
        rotate=(-15, 15),
        p=0.5
    ),
    A.RandomBrightnessContrast(p=0.2)
])

# --- Augment Loop ---
image_files = sorted(os.listdir(image_dir))

for img_file in tqdm(image_files, total=len(image_files), desc="Augmenting images"):
    name_wo_ext = os.path.splitext(img_file)[0]
    img_path = os.path.join(image_dir, img_file)
    mask_folder = os.path.join(annotation_dir, name_wo_ext)

    if not os.path.exists(mask_folder):
        print(f"⚠️ Annotation folder missing for {img_file}")
        continue

    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Failed to read image: {img_path}")
        continue

    mask_files = sorted(os.listdir(mask_folder))
    masks = []
    mask_names = []

    for mfile in mask_files:
        mpath = os.path.join(mask_folder, mfile)
        mask = cv2.imread(mpath, 0)  # grayscale

        if mask is None:
            print(f"❌ Could not read mask: {mpath}")
            continue

        if mask.shape != image.shape[:2]:
            print(f"⚠️ Shape mismatch in mask: {mpath} | mask: {mask.shape}, image: {image.shape[:2]}")
            continue

        masks.append(mask)
        mask_names.append(os.path.splitext(mfile)[0])

    if not masks:
        print(f"⚠️ No valid masks for image: {img_file}. Skipping...")
        continue

    # Apply multiple augmentations
    for i in range(5):
        try:
            augmented = transform(image=image, masks=masks)
        except Exception as e:
            print(f"❌ Augmentation failed for {img_file} (aug {i}): {e}")
            continue

        aug_image = augmented["image"]
        aug_masks = augmented["masks"]

        # Save augmented image
        aug_img_name = f"{name_wo_ext}_aug{i}.jpg"
        cv2.imwrite(os.path.join(aug_image_dir, aug_img_name), aug_image)

        # Save augmented masks
        aug_mask_folder = os.path.join(aug_anno_dir, f"{name_wo_ext}_aug{i}")
        os.makedirs(aug_mask_folder, exist_ok=True)

        for j, aug_mask in enumerate(aug_masks):
            aug_mask_name = f"{mask_names[j]}_aug{i}.png"
            aug_mask_path = os.path.join(aug_mask_folder, aug_mask_name)
            cv2.imwrite(aug_mask_path, aug_mask)
