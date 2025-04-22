import os, cv2

base_path = r"C:\Users\NDT Lab\Pictures\dataset\archive\corrosion detect"
# Get all files in the directory
files = os.listdir(os.path.join(base_path, "images"))
files = [os.path.join(base_path, "images", f) for f in files]

for f in files:
    img = cv2.imread(f)
    print(img.shape)