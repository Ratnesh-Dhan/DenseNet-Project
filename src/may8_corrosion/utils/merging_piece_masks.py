import os
import cv2

# Main dataset folder where each subfolder is named after the image
dataset_dir = r'D:\NML ML Works\corrosion all masks\dataset 2025-04-25 16-40-02\masks_instances'
output_dir = r'D:\NML ML Works\corrosion all masks\dataset 2025-04-25 16-40-02\merged_masks'
# os.makedirs(output_dir, exist_ok=True)

# Iterate over each subfolder (one per image)
for folder_name in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    merged_mask = None

    # Look for all files starting with 'piece_'
    for file_name in os.listdir(folder_path):
        if not file_name.startswith('piece_') or not file_name.endswith('.png'):
            continue

        file_path = os.path.join(folder_path, file_name)
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Failed to read: {file_path}")
            continue

        _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        if merged_mask is None:
            merged_mask = bin_mask
        else:
            merged_mask = cv2.bitwise_or(merged_mask, bin_mask)

    if merged_mask is not None:
        output_path = os.path.join(output_dir, f"{folder_name}.png")
        cv2.imwrite(output_path, merged_mask)
        print(f"Saved merged mask for {folder_name}")
    else:
        print(f"No piece_*.png files found in {folder_name}")
