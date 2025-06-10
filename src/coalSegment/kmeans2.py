import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from skimage import io

# Load image
image_path = "./images/02.jpg"  # Replace with your actual path
image = io.imread(image_path)
pixels = image.reshape(-1, 3)

# Apply KMeans clustering (you can change n_clusters to 5–7 for more detail)
kmeans = KMeans(n_clusters=7, random_state=42)
kmeans.fit(pixels)

# Map each pixel to its cluster's center
segmented_pixels = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
segmented_image = segmented_pixels.reshape(image.shape)

# Show the original and segmented image
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(segmented_image)
ax[1].set_title("Segmented Image (5 Clusters)")
ax[1].axis('off')

plt.tight_layout()
plt.show()
