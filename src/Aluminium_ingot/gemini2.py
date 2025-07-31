import cv2, numpy as np
from matplotlib import pyplot as plt

# 1. Load the Image
image_path = "img/ingot3.jpeg"
image_bgr = cv2.imread(image_path)

if image_bgr is None:
   print(f"Error: Could not load image at {image_path}. Please check the path and file name.")
else:
   # 2. Convert to Grayscale
   image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

   # 3. Noise Reduction (Gaussian Blur)
   blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)

# --- Option 1: Edge Enhancement (Sharpening) ---
# Create a sharpening kernel
sharpen_kernel = np.array([[-1, -1, -1],
                            [-1,  9, -1],
                            [-1, -1, -1]])
sharpened_image = cv2.filter2D(blurred_image, -1, sharpen_kernel)

# --- Option 2: Adaptive Thresholding ---
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C: Threshold value is the weighted sum of neighborhood pixels (Gaussian window) minus a constant C.
# cv2.THRESH_BINARY: Pixel values above the threshold are set to a maximum value (e.g., 255), others to 0.
adaptive_thresh = cv2.adaptiveThreshold(blurred_image, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2) # Block size 11, constant C = 2 (adjust these)


# Canny edge detection for grayScale image
# min_ = 60
# max_ = 90
# canny = cv2.Canny(sharpened_image, min_, max_)

sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
# Convert back to 8-bit unsigned integer and take absolute values
# This is important for visualization and combining correctly.
sobelx_abs = cv2.convertScaleAbs(sobelx)
sobely_abs = cv2.convertScaleAbs(sobely)
# Combine the X and Y gradient magnitudes
# cv2.addWeighted allows you to assign different weights if one direction is more important.
# Here, 0.5 for both means equal contribution.
sobel_combined = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0)

# --- NEW: Dilation after Canny ---
# This helps to connect broken Canny edges.
# kernel_dilate_edges = np.ones((3,3), np.uint8) # Small kernel to connect edges
# dilated_canny = cv2.dilate(canny, kernel_dilate_edges, iterations=1) # Adjust iterations if needed

# Finding Contours
contours, hierarchy = cv2.findContours(sobel_combined.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_with_contours = image_bgr.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 1) # -1 means draw all contours

# --- Contour Filtering and Final Counting ---
filtered_contours = []
counted_blocks = 0
image_final_display = image_bgr.copy() # Use a fresh copy for final output

# --- !!! TUNE THESE VALUES FOR YOUR SPECIFIC IMAGE !!! ---
min_contour_area = 100   # Minimum area in pixels a contour must have to be counted
max_contour_area = 50000  # Maximum area in pixels a contour can have
# Adjust these based on your image resolution and block size for best accuracy.
found_counters = 0
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    
    # Filter based on area
    if min_contour_area < area < max_contour_area:
        filtered_contours.append(contour)
        counted_blocks += 1

        # Draw the filtered contour and add a count label
        cv2.drawContours(image_final_display, [contour], -1, (0, 255, 0), 1) # Green outline
        
        # Get bounding box for text placement
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(image_final_display, str(counted_blocks), (x, y - 10), # Position text slightly above contour
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1) # Red text, thickness 2
        found_counters = found_counters + 1        

# --- Visualization of Enhancement Steps ---
plt.figure(figsize=(18, 6))

plt.subplot(1, 4, 1)
plt.imshow(blurred_image, cmap="gray")
plt.title("Blurred Image (Base) ->")
plt.axis("off")

# plt.subplot(1, 4, 2)
# plt.imshow(sharpened_image, cmap="gray")
# plt.title("Sharpened Image")
# plt.axis("off")

# plt.subplot(1, 4, 2)
# plt.imshow(adaptive_thresh, cmap="gray")
# plt.title("Adaptive Thresholding")
# plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(sharpened_image, cmap="gray")
plt.title("Sharpen image")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(sobel_combined, cmap="gray")
plt.title("Sobel")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(cv2.cvtColor(image_final_display, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title(f"Raw contours found: {found_counters}")

plt.tight_layout()
plt.show()
