import cv2, os
import numpy as np

def closed_polygon_mask(image_path, output_path):
    # Step 1: Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 2: Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Step 3: Apply Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Invert if background is white
    if np.sum(binary == 255) > np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    # Step 5: Dilation + Morph Closing to close any gaps in the contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Step 6: Find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Create a blank mask and draw the convex hull of the largest contour
    mask = np.zeros_like(binary)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)

    # Step 8: Smooth jagged edges slightly
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, final_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Step 9: Save the final filled PNG mask
    cv2.imwrite(output_path, final_mask)

# Example usage
if __name__ == "__main__":
    dir_path = r"D:\NML ML Works\corrosionDataset\images"
    save_path = r"D:\NML ML Works\corrosionDataset\masks"
    all_files = os.listdir(dir_path)
    for a in all_files:
        image_path = os.path.join(dir_path, a)
        same_image_path = os.path.join(save_path, f'piece_mask_{a}')
        closed_polygon_mask(image_path, same_image_path)
# closed_polygon_mask("old2.jpg", "trial_perfect_closed_mask.png")