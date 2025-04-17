import numpy as np, cv2
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans
from utils.scaleRemover import scale_remover

def rgb_to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(*color)

# Load image
# image_path = "./images/fromWord2.jpg"
image_path = "./images/cool.jpg"
image = io.imread(image_path)
if image.shape != (1944, 2592, 3):
    image = cv2.resize(image, (2592, 1944))
print("Shape : ",image.shape)
image = scale_remover(image)
pixels = image.reshape(-1, 3)

# Filter out black pixels (0,0,0)
non_black_mask = ~np.all(pixels == [0, 0, 0], axis=1)
non_black_pixels = pixels[non_black_mask]

# Apply KMeans clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

kmeans.fit(non_black_pixels) # from (pixels)

labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_.astype(np.uint8)

# Reshape to image dimensions
# segmented_pixels = cluster_centers[labels]
segmented_pixels = np.zeros_like(pixels)
segmented_pixels[non_black_mask] = cluster_centers[labels]
segmented_image = segmented_pixels.reshape(image.shape)

# Count pixels per cluster
# total_pixels = len(labels)
total_pixels = len(non_black_pixels)
percentage = []
colors = []

for i in range(n_clusters):
    count = np.sum(labels == i)
    percent = (count / total_pixels) * 100
    percentage.append(percent)
    print("Color : ", cluster_centers[i])
    print("Normalized color : ", cluster_centers[i] / 255.0)
    # colors.append(cluster_centers[i] / 255.0)  # Normalize RGB values for matplotlib plotting
    # Use float RGB for bar chart (in 0-1 range) but preserve precision
    rgb = cluster_centers[i]
    rgb_float = [c / 255.0 for c in rgb]
    colors.append(rgb_float)
    # colors.append(rgb_to_hex(cluster_centers[i]))

# Background color 
bg_color = '#9dd9fc'
# Plot segmented image and percentage bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=bg_color)
fig.patch.set_facecolor(bg_color)

# Segmented image
segmented_image = segmented_image.astype(np.float32) / 255.0  # For matplotlib to interpret correctly
ax1.imshow(segmented_image)
ax1.set_title("Segmented")
ax1.axis("off")
ax1.set_facecolor(bg_color)

# Bar chart
bars = ax2.bar(range(n_clusters), percentage, color=colors)
ax2.set_xticks(range(n_clusters))
ax2.set_xticklabels([f"Cluster {i+1}" for i in range(n_clusters)])
ax2.set_ylabel("Percentage (%)")
ax2.set_title("Cluster Composition by Color")
ax2.set_facecolor(bg_color)

# Add percentages on top
for i, bar in enumerate(bars):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2 + 1,
             f"{percentage[i]:.2f}%", ha='center', va='bottom')

plt.tight_layout()
plt.show()