import os
import shutil
import random

dataset_path = r"/mnt/z/DATASETS/kaggle_semantic_segmentation_CORROSION_dataset"

# train image and mask path
train_img_path = os.path.join(dataset_path, "train/images")
train_mask_path = os.path.join(dataset_path, "train/masks")

# validation image and mask path
val_img_path = os.path.join(dataset_path, "validate/images")
val_mask_path = os.path.join(dataset_path, "validate/masks")

images = os.listdir(train_img_path)

print(images[:20])
random.shuffle(images)
print('\n',' '.join(['*' for i in range(50)]), '\n')
print(images[:20])

for i in range(303):
    shutil.move(os.path.join(train_img_path, images[i]), val_img_path)
    shutil.move(os.path.join(train_mask_path, images[i].split('.')[0]+'.png'), val_mask_path)