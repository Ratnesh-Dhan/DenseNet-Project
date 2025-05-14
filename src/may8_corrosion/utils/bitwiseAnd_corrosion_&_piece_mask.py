import cv2
import os
import shutil

# Define paths (set once)
corrosion_mask_dir = r'D:\NML ML Works\corrosionDataset\masks_corrosion'  # binary corrosion masks (white = corrosion)
merged_piece_mask_dir = r'D:\NML ML Works\corrosion all masks\dataset 2025-04-25 16-40-02\merged_masks'  # binary metal piece masks
output_cleaned_mask_dir = r'D:\NML ML Works\corrosion all masks\dataset 2025-04-25 16-40-02\filtered_corrosion'  # output path
os.makedirs(output_cleaned_mask_dir, exist_ok=True)

# Loop through each corrosion mask
for fname in os.listdir(corrosion_mask_dir):
    if not fname.startswith('corrosion_mask_'):
        continue

    corrosion_path = os.path.join(corrosion_mask_dir, fname)
    base_name = fname.replace('corrosion_mask_', '', 1)
    merged_mask_path = os.path.join(merged_piece_mask_dir, base_name)

    # Read masks
    corrosion_mask = cv2.imread(corrosion_path, cv2.IMREAD_GRAYSCALE)
    if corrosion_mask is None:
        print(f"[ERROR] Could not read corrosion mask: {corrosion_path}")
        continue

    piece_mask = cv2.imread(merged_mask_path, cv2.IMREAD_GRAYSCALE)

    if piece_mask is not None:
        # Resize corrosion mask to match piece mask shape
        if corrosion_mask.shape != piece_mask.shape:
            corrosion_mask = cv2.resize(corrosion_mask, (piece_mask.shape[1], piece_mask.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
            print(f"[RESIZED] {fname} to match piece mask")

        # Threshold both to binary
        _, piece_binary = cv2.threshold(piece_mask, 127, 255, cv2.THRESH_BINARY)
        _, corrosion_binary = cv2.threshold(corrosion_mask, 127, 255, cv2.THRESH_BINARY)

        # Keep only corrosion inside pieces
        cleaned_mask = cv2.bitwise_and(corrosion_binary, piece_binary)

        print(f"[CLEANED] {fname}")
        cv2.imwrite(os.path.join(output_cleaned_mask_dir, fname), cleaned_mask)

    else:
        print(f"[COPIED] No piece mask found for {fname}, copying original.")
        shutil.copyfile(corrosion_path, os.path.join(output_cleaned_mask_dir, fname))
