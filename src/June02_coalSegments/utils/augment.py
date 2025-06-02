from PIL import Image
import os, random

folder_path = r"D:\NML ML Works\TRAINING-20250602T050431Z-1-001\TRAINING\Inorganic\Cavity filled"
save_folder = r"D:\NML ML Works\TRAINING-20250602T050431Z-1-001\TRAINING\Inorganic\augmented"
files = os.listdir(folder_path)

for file in files:
    image = Image.open(os.path.join(folder_path, file))
    choice = random.choice([True, False])
    if choice:
        new_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        new_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    file_name = file.replace(".png", "_flipped.png")
    new_image.save(os.path.join(save_folder, file_name))
    print(f"Flipped {file} and saved as {file_name}")

