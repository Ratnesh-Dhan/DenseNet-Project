from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Load the image
image_path = "./i003.jpg"
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# Reshape the image to a 2D array of pixels
pixels = image_np.reshape((-1, 3))

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(pixels)
segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]
segmented_image = segmented_pixels.reshape(image_np.shape).astype(np.uint8)

# Plot original and segmented images side by side
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].imshow(image_np)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(segmented_image)
ax[1].set_title("Segmented Image (KMeans, 3 clusters)")
ax[1].axis("off")

plt.tight_layout()
plt.show()
