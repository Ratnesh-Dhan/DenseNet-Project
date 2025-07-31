import cv2, numpy as np
from matplotlib import pyplot as plt

# --- Assume 'image_bgr', 'dilated_canny', and 'contours' are defined from your main script ---
# (Pasting the full script here for completeness if you run this as a standalone test)

# 1. Load the Image
image_path = "img/ingot3.jpeg"
image_bgr = cv2.imread(image_path)

# 2. Convert to Grayscale
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# 3. Noise Reduction (Gaussian Blur)
blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)

# --- Option 1: Edge Enhancement (Sharpening) ---
sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_image = cv2.filter2D(blurred_image, -1, sharpen_kernel)

# --- Option 2: Adaptive Thresholding ---
adaptive_thresh = cv2.adaptiveThreshold(blurred_image, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

# Canny edge detection for grayScale image (applied to adaptive_thresh)
min_ = 80
max_ = 100
canny = cv2.Canny(adaptive_thresh, min_, max_)

# --- Dilation after Canny ---
kernel_dilate_edges = np.ones((3,3), np.uint8)
dilated_canny = cv2.dilate(canny, kernel_dilate_edges, iterations=1)

# Finding Contours
contours, hierarchy = cv2.findContours(dilated_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# --- Contour Filtering and Final Counting ---
filtered_contours = []
counted_blocks = 0
image_final_display = image_bgr.copy() # Use a fresh copy for final output

# --- !!! TUNE THESE VALUES FOR YOUR SPECIFIC IMAGE !!! ---
min_contour_area = 200    # Minimum area in pixels a contour must have to be counted
max_contour_area = 50000  # Maximum area in pixels a contour can have
# Adjust these based on your image resolution and block size for best accuracy.

print(f"\n--- Analyzing {len(contours)} raw contours ---")
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    
    # Print the area of each contour for debugging
    print(f"Contour {i+1}: Area = {area:.2f} pixels")
    
    # Filter based on area
    if min_contour_area < area < max_contour_area:
        print(f"  -> PASSED filter (Area: {area:.2f})")
        filtered_contours.append(contour)
        counted_blocks += 1

        # Draw the filtered contour and add a count label
        cv2.drawContours(image_final_display, [contour], -1, (0, 255, 0), 1) # Green outline
        
        # Get bounding box for text placement
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(image_final_display, str(counted_blocks), (x, y - 10), # Position text slightly above contour
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1) # Red text, thickness 1 (changed from 2 for visibility)
    else:
        print(f"  -> FAILED filter (Area: {area:.2f} - outside [{min_contour_area}, {max_contour_area}])")


# --- FINAL VISUALIZATION (rest of your plotting code) ---
plt.figure(figsize=(20, 6))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(adaptive_thresh, cmap="gray")
plt.title("Adaptive Thresholding")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(dilated_canny, cmap="gray")
plt.title("Dilated Canny (Input for Contours)")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(image_final_display, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title(f"Final Count: {counted_blocks} Blocks")

plt.tight_layout()
plt.show()

print(f"\nTotal blocks counted after filtering: {counted_blocks}")
print("\n--------------------------------------------------------------")
print("!!! IMPORTANT: Adjust 'min_contour_area' and 'max_contour_area' !!!")
print("!!! These values determine what is considered a 'block'.     !!!")
print("!!! Look at the output and refine them for best accuracy.    !!!")
print("--------------------------------------------------------------")