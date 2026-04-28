import cv2
import os

# Input and output directory (can be the same)
input_dir = r"D:\NML ML Works\corrosion all masks\dataset 2025-04-25 16-40-02\img"
output_dir = r"D:\NML ML Works\corrosion all masks\dataset 2025-04-25 16-40-02\img_2nd_png_version"  # or set to a different folder
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
count = 0
total = len(os.listdir(input_dir))
for fname in os.listdir(input_dir):
    count = count + 1
    print(count," / ", total)
    if fname.lower().endswith((".jpg", ".jpeg")):
        jpg_path = os.path.join(input_dir, fname)
        png_name = os.path.splitext(fname)[0] + ".png"
        png_path = os.path.join(output_dir, png_name)

        # Read and save image
        img = cv2.imread(jpg_path)
        if img is not None:
            cv2.imwrite(png_path, img)
            print(f"[CONVERTED] {fname} → {png_name}")
        else:
            print(f"[SKIPPED] Couldn't read: {fname}")
    elif fname.lower().endswith(".png"):
        # For PNG files, just copy to output directory
        src_path = os.path.join(input_dir, fname)
        dst_path = os.path.join(output_dir, fname)
        img = cv2.imread(src_path)
        if img is not None:
            cv2.imwrite(dst_path, img)
            print(f"[COPIED] {fname}")
        else:
            print(f"[SKIPPED] Couldn't read: {fname}")
    else:
        print(f"[SKIPPED] Unsupported format: {fname}")
