import os
from PIL import Image

base_location = r'/home/zumbie/Codes/NML/DenseNet-Project/Datasets/corrosion'

train_dir = os.path.join(base_location, 'train')
valid_dir = os.path.join(base_location, 'validate')

train_files = os.listdir(train_dir)
valid_files = os.listdir(valid_dir)

ary = []

for folders in train_files:
    ary.append(os.path.join(train_dir, folders))

for folders in valid_files:
    ary.append(os.path.join(valid_dir, folders))

for i in ary:
    files = [f for f in os.listdir(i) if f.endswith('.png')]
    for f in files:
        im_path = os.path.join(i, f)
        img = Image.open(im_path)
        img.save(im_path, icc_profile=None)
        print(f'done {f}')