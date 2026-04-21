import os 
import random

bmp_path = "/mnt/z/DATASETS/Fringe/bmp"
height_path = "/mnt/z/DATASETS/Fringe/heightmaps"
save_bmp_path = "/mnt/z/DATASETS/Fringe/bmp_save"   
save_height_path = "/mnt/z/DATASETS/Fringe/height_save"
os.makedirs(save_bmp_path,exist_ok=True)
os.makedirs(save_height_path,exist_ok=True)

bmp_files = os.listdir(bmp_path)
total = len(bmp_files)
random.shuffle(bmp_files)

print(total)
print(bmp_files)

for i in range(int(total*0.1)):
    os.rename(os.path.join(bmp_path,bmp_files[i]),os.path.join(save_bmp_path,bmp_files[i]))
    file = bmp_files[i].replace(".bmp","_height_map.npy")
    os.rename(os.path.join(height_path,file),os.path.join(save_height_path,file))