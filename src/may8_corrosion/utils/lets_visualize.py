import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

image_path = r"D:\NML ML Works\corrosion all masks\dataset 2025-04-25 16-40-02\img"
output_dir = r"D:\NML ML Works\Testing_mask_binary_resized"

def visualization(filename):
    # Load the original image
    original_image = cv2.imread(os.path.join(image_path, filename))

    # Load the mask image
    mask_image = cv2.imread(os.path.join(output_dir, f'{filename}.png'), cv2.IMREAD_UNCHANGED)

    # Resize the mask image to match the size of the original image
    mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]))

    # Create a copy of the original image
    overlaid_image = original_image.copy()

    # Check the number of channels in the mask image
    if mask_image.shape[2] == 4:
        # The mask image has an alpha channel
        # overlaid_image[mask_image[:, :, 3] > 0] = mask_image[mask_image[:, :, 3] > 0, :3]
        mask_alpha = mask_image[:, :, 3] / 255.0  # Normalize the alpha channel to [0, 1]
        overlaid_image = cv2.addWeighted(original_image, 1 - 0.1*mask_alpha, mask_image[:, :, :3], mask_alpha, 0)
    else:
        # The mask image does not have an alpha channel
        overlaid_image[mask_image[:, :, 0] > 0] = mask_image[mask_image[:, :, 0] > 0, :]

    # Display the overlaid image using Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(overlaid_image)
    plt.title(f'Overlaid Image: {filename}')
    plt.axis('off')
    plt.show()

files = os.listdir(image_path)
for f in files:
    visualization(filename=f)
