import os

base_location = r'D:\NML ML Works\corrosion Final dataset'

train_dir = os.path.join(base_location, 'train')
valid_dir = os.path.join(base_location, 'validate')

train_files = os.listdir(train_dir)
valid_files = os.listdir(valid_dir)

ary = []

for folders in train_files:
    ary.append(os.path.join(train_dir, folders))

for folders in valid_files:
    ary.append(os.path.join(valid_dir, folders))

