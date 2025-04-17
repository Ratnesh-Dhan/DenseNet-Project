import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans

# Load image
image_path = "./image 009.jpg"
image = io.imread(image_path)
pixels = image.reshape(-1, 3)

# Apply KMeans clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(pixels)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_.astype(np.uint8)

# Reshape to image dimensions
segmented_pixels = cluster_centers[labels]
segmented_image = segmented_pixels.reshape(image.shape)

# Count pixels per cluster
total_pixels = len(labels)
percentage = []
colors = []

for i in range(n_clusters):
    count = np.sum(labels == i)
    percent = (count / total_pixels) * 100
    percentage.append(percent)
    colors.append(cluster_centers[i] / 255.0) # Normalize RGB for matplotlib plotting

# Plot segmented image and percentage bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Segmented image
ax1.imshow(segmented_image)
ax1.set_title("Segmented")
ax1.axis("off")

# Bar chart
bars = ax2.bar(range(n_clusters), percentage, color=colors)
ax2.set_xticks(range(n_clusters))
ax2.set_xticklabels([f"Cluster {i+1}" for i in range(n_clusters)])
ax2.set_ylabel("Percentage (%)")
ax2.set_title("Cluster Composition by Color")

# Add percentages on top
for i, bar in enumerate(bars):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2 + 1,
             f"{percentage[i]:.2f}%", ha='center', va='bottom')

plt.tight_layout()
plt.show()