import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io
import matplotlib.cm as cm

# Load image
image_path = "./image 009.jpg"  # Replace with your actual path
image = io.imread(image_path)
pixels = image.reshape(-1, 3)

# Apply KMeans clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(pixels)
labels = kmeans.labels_.reshape(image.shape[:2])

# Create a distinct colormap
colormap = cm.get_cmap('tab10', n_clusters)  # You can try 'Set3', 'nipy_spectral', etc.

# Map labels to distinct RGB colors
colored_mask = np.zeros_like(image, dtype=np.uint8)
for i in range(n_clusters):
    color = (np.array(colormap(i)[:3]) * 255).astype(np.uint8)
    colored_mask[labels == i] = color

# Show the original and segmented image with vivid cluster colors
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(colored_mask)
ax[1].set_title(f"Segmented Mask (Enhanced Colors, {n_clusters} Clusters)")
ax[1].axis('off')

plt.tight_layout()
plt.show()
