import cv2

def create_otsu_mask(image_path, output_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to reduce noise (helps Otsu)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Otsu’s thresholding
    _, binary_mask = cv2.threshold(blurred, 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: Invert if foreground appears black instead of white
    white_ratio = (binary_mask == 255).sum() / binary_mask.size
    if white_ratio > 0.5:
        binary_mask = cv2.bitwise_not(binary_mask)

    # Save binary mask
    cv2.imwrite(output_path, binary_mask)

# Example usage
create_otsu_mask("old.jpg", "otsu_mask_old.png")