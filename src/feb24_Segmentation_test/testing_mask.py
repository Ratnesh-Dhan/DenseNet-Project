import json
import base64
import zlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Function to visualize segmentation masks on the image
def visualize_segmentation(image_path, json_path):
    # Load the image
    image = Image.open(image_path)
    image = image.convert("RGB")

    # Load the annotation JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Prepare the figure
    plt.figure(figsize=(8, 6))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("masks")
    plt.axis('off')

    # Iterate over each object in the annotation
    for obj in data['objects']:
        if obj['geometryType'] == 'bitmap':
            # Decode and decompress the bitmap data
            bitmap_data = base64.b64decode(obj['bitmap']['data'])
            try:
                decompressed_data = zlib.decompress(bitmap_data)
                bitmap_image = Image.open(io.BytesIO(decompressed_data))
                mask = np.array(bitmap_image.convert('L'))
            except Exception as e:
                print(f"Error processing bitmap for {obj['classTitle']}: {e}")
                continue

            # Get the origin of the mask
            origin = obj['bitmap']['origin']

            # Create an empty mask with the same size as the image
            full_mask = np.zeros((image.height, image.width), dtype=np.uint8)
            h, w = mask.shape
            full_mask[origin[1]:origin[1]+h, origin[0]:origin[0]+w] = mask

            # Overlay the mask with transparency
            plt.imshow(full_mask, alpha=0.6, cmap='jet')

    plt.subplot(1,2,2)
    plt.imshow(image)
    plt.title("original")
    plt.axis("off")
    plt.show()

# Example usage
image_path = '../../Datasets/testDataset/img/2007_007948.jpg'
annotation_path = '../../Datasets/testDataset/ann/2007_007948.jpg.json'
visualize_segmentation(image_path, annotation_path)
