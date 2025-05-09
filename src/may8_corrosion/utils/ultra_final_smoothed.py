import cv2
import numpy as np

def perfect_mask(image_path, output_path):
    # Read image and blur
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Otsu thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if white background dominates
    if np.sum(binary == 255) > np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    # Morphological closing to fix edge gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find external contours
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Create empty mask
    mask = np.zeros_like(closed)

    # Fill the largest contour and its holes
    if contours is not None:
        max_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
        for i, h in enumerate(hierarchy[0]):
            if h[3] == -1:  # outer contour
                if i == max_idx:
                    cv2.drawContours(mask, contours, i, 255, cv2.FILLED)
            elif h[3] == max_idx:  # holes inside largest contour
                cv2.drawContours(mask, contours, i, 255, cv2.FILLED)

    # Smooth jagged edges
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Save the result
    cv2.imwrite(output_path, mask)

# Example usage
perfect_mask("old.jpg", "smoothed_mask.png")
