import cv2
import os

image_path = r"D:\NML ML Works\corrosion all masks\dataset 2025-04-25 16-40-02\img"
output_dir = r"D:\NML ML Works\Testing_mask_binary_resized"
os.makedirs(output_dir, exist_ok=True)

def mask_full_path(file_name):
    mask_path = r'D:\NML ML Works\Testing_mask_binary'
    return cv2.imread(os.path.join(mask_path, f'{file_name}.png'))

files = [os.path.join(image_path, f) for f in os.listdir(image_path)]

for i in files:
    file_name = i.split('\\')[-1]
    image = cv2.imread(i, cv2.IMREAD_COLOR)
    x, y, _ = image.shape
    mask = mask_full_path(file_name=file_name)
    mask = cv2.resize(mask, (y, x))

    # Save the resized mask to a new directory
    save_path = os.path.join(output_dir, f"{file_name}.png")
    cv2.imwrite(save_path, mask)
