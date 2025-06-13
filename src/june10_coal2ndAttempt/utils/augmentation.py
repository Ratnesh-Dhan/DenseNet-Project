from PIL import Image
import os, random, sys

folder_path = r"D:\NML ML Works\newCoalByDeepBhaiya\16\TRAINING 16\4 Vitrinite"
save_folder = r"D:\NML ML Works\newCoalByDeepBhaiya\16\TRAINING 16\4 Vitrinite"
files = random.sample(os.listdir(folder_path), len(os.listdir(folder_path)))
files = files[:len(files)//2]  # Take only the first half of the files array

for file in files:
    image = Image.open(os.path.join(folder_path, file))
    file_name = file.replace('.png', '')

    # # Fliping left to right
    # new_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # new_image.save(os.path.join(save_folder, f'{file_name}_flipped_LtoR.png'))

    new_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    new_image.save(os.path.join(save_folder, f'{file_name}_flipped_TtoB.png'))

    # # Rotations
    # new_image.rotate(90).save(os.path.join(save_folder, f'{file_name}_rot90.png'))
    # new_image.rotate(180).save(os.path.join(save_folder, f'{file_name}_rot180.png'))
    # new_image.rotate(270).save(os.path.join(save_folder, f'{file_name}_rot270.png'))


sys.exit(0)
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

