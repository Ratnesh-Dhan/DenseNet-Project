import cv2
import numpy as np
import os

# Define color range for corrosion (rusty areas)
# These ranges might need minor tweaking based on your dataset
lower_corrosion = np.array([5, 50, 50])   # HSV lower bound (rusty orange/brown)
upper_corrosion = np.array([20, 255, 255]) # HSV upper bound

# Define color range for metal surface (you can adjust if needed)
lower_metal = np.array([0, 0, 150])       # HSV lower bound (light gray/white)
upper_metal = np.array([180, 50, 255])    # HSV upper bound

# Paths
input_folder = r'D:\NML ML Works\cropped corrosion'    # Update this
output_mask_folder = r'D:\NML ML Works\cropped corrosion annotaion\corrosion_mask'
output_piece_folder = r'D:\NML ML Works\cropped corrosion annotaion\sample_piece_mask'

os.makedirs(output_mask_folder, exist_ok=True)
os.makedirs(output_piece_folder, exist_ok=True)

# Loop through images
for img_name in os.listdir(input_folder):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create corrosion mask
    corrosion_mask = cv2.inRange(hsv, lower_corrosion, upper_corrosion)
    
    # Create sample piece mask
    piece_mask = cv2.inRange(hsv, lower_metal, upper_metal)
    
    # Save masks
    corrosion_mask_path = os.path.join(output_mask_folder, f"corrosion_mask_{img_name}")
    piece_mask_path = os.path.join(output_piece_folder, f"piece_mask_{img_name}")

    cv2.imwrite(corrosion_mask_path, corrosion_mask)
    cv2.imwrite(piece_mask_path, piece_mask)
    
    # Optional: Calculate corrosion percentage
    corrosion_area = np.sum(corrosion_mask > 0)
    piece_area = np.sum(piece_mask > 0)
    
    if piece_area > 0:
        corrosion_percentage = (corrosion_area / piece_area) * 100
        print(f"{img_name}: Corrosion = {corrosion_percentage:.2f}%")
    else:
        print(f"{img_name}: No sample piece detected!")

print("Done creating masks!")
