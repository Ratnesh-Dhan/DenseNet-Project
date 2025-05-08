import cv2
import numpy as np

def refined_otsu_mask(image_path, output_path):
    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Otsu thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if background is white
    if np.sum(binary == 255) > np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create blank mask
    mask = np.zeros_like(binary)

    # Draw and fill the largest contour only
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Save final mask
    cv2.imwrite(output_path, mask)

# Example usage
refined_otsu_mask("old.jpg", "final_otsu_mask.png")
