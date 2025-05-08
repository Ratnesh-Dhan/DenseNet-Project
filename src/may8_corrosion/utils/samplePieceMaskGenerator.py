import cv2
import numpy as np

def edge_mask(image_path, output_path):
    # Load image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Optional: dilate to make edges thicker
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Save as binary mask (white edges on black)
    cv2.imwrite(output_path, edges_dilated)

# Example usage:
edge_mask("new.jpg", "mask_edges_104520.png")