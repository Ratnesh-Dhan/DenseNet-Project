from PIL import Image
import os
# Example usage
file_names = os.listdir('../img')
filename = [f for f in file_names if f.endswith('.png.png')]

for f in filename:
    image = Image.open(f"../img/{f}")
    name = f.replace('.png.png', '')
    image.save(f"../img/{name}.jpg", "jpeg")
    os.remove(f"../img/{f}")
