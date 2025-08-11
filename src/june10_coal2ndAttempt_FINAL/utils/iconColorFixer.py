import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image with RGBA (including alpha channel)
image_path = r"C:\Users\NDT Lab\Downloads\ai-technology.png"
image = plt.imread(image_path)

# Convert to uint8 if loaded as float
if image.dtype == np.float32 or image.dtype == np.float64:
    image = (image * 255).astype(np.uint8)

# Separate RGB and Alpha channels
rgb = image[:, :, :3]
alpha = image[:, :, 3]

# Find black pixels: RGB == [0, 0, 0]
black_mask = np.all(rgb == [0, 0, 0], axis=-1)

# Change black pixels to white in RGB
rgb[black_mask] = [255, 255, 255]

# Recombine with original alpha channel
result = np.dstack((rgb, alpha))

# Convert to BGRA for OpenCV saving
image_bgra = cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)

# Save
output_path = r"C:\Users\NDT Lab\Downloads\icon_white_fixed.png"
cv2.imwrite(output_path, image_bgra)

print("Saved image with black pixels turned white, transparent background kept.")
