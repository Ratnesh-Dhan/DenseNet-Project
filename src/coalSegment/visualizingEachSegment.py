import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from skimage import io

# Load image
image_path = "./image 009.jpg"  # Replace with your actual path
image = io.imread(image_path)
pixels = image.reshape(-1, 3)

# KMeans Clustering
n_clusters = 5 # for row column output
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(pixels)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_.astype(np.uint8)

# Reshape labels back to 2D
label_image = labels.reshape(image.shape[:2])

# Plot each cluster as a separate image
# fig, axes = plt.subplots(1, 5, figsize=(20, 4))

cols = 3 # type 2
rows = (n_clusters + cols-1) // cols # type 2
fig, axes = plt.subplots(rows, cols, figsize=(15, 5)) # type 2
axes = axes.flatten() # type 2

for i in range(5):
    mask = (label_image == i)
    
    # Create blank image and assign cluster color only where the mask is True
    cluster_image = np.zeros_like(image)
    cluster_image[mask] = cluster_centers[i]
    
    axes[i].imshow(cluster_image)
    axes[i].set_title(f"Cluster {i}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
