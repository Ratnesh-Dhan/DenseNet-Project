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

# Assuming 'blurred_image' from the previous step is available

if 'blurred_image' not in locals():
   print("Please run the previous preprocessing step first to load and blur the image.")
else:
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
   # THRESH_BINARY_INV inverts the result (blocks become white on black background, often better for contour finding)

   # --- Option 3: Morphological Gradient ---
   # The difference between the dilation and erosion of an image.
   # Highlights the borders of objects.
   kernel = np.ones((3, 3), np.uint8)
   dilated = cv2.dilate(blurred_image, kernel, iterations=1)
   eroded = cv2.erode(blurred_image, kernel, iterations=1)
   morph_gradient = cv2.subtract(dilated, eroded)


   # --- Visualization of Enhancement Steps ---
   plt.figure(figsize=(18, 6))

   plt.subplot(1, 4, 1)
   plt.imshow(blurred_image, cmap="gray")
   plt.title("Blurred Image (Base)")
   plt.axis("off")

   plt.subplot(1, 4, 2)
   plt.imshow(sharpened_image, cmap="gray")
   plt.title("Sharpened Image")
   plt.axis("off")

   plt.subplot(1, 4, 3)
   plt.imshow(adaptive_thresh, cmap="gray")
   plt.title("Adaptive Thresholding")
   plt.axis("off")

   plt.subplot(1, 4, 4)
   plt.imshow(morph_gradient, cmap="gray")
   plt.title("Morphological Gradient")
   plt.axis("off")

   plt.tight_layout()
   plt.show()

   print("Visibility enhancement steps completed. Observe the results to decide which method makes the block boundaries most distinct.")
   print("Pay attention to consistent edges and separation between blocks.")
   print("You might need to adjust the parameters for adaptive thresholding (block size and C).")
   print("The sharpened image might enhance noise as well, so observe carefully.")