from PIL import Image
import os

image_dir = r"D:\NML ML Works\cropped corrosion annotaion\sample_piece_mask"

for filename in os.listdir(image_dir):
    if filename.lower().endswith('.png'):
        filepath = os.path.join(image_dir, filename)
        try:
            with Image.open(filepath) as img:
                # Convert to remove profile metadata
                img.convert("RGB").save(filepath, optimize=True)
                print(f"Cleaned: {filename}")
        except Exception as e:
            print(f"Failed to clean {filename}: {e}")
