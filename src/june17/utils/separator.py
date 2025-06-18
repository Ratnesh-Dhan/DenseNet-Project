import os
import shutil
from tqdm import tqdm

folder_path = r"D:\NML ML Works\new square corrosion dataset\squares"
base_path = r"D:\NML ML Works\new square corrosion dataset"
files = os.listdir(folder_path)

hs = set([])

for file in files:
    hs.add(file.split('_')[0])

for i in hs:
    paths = os.path.join(base_path, i)
    if not os.path.exists(paths):
        os.makedirs(paths)

for file in tqdm(files):
    preffix = file.split('_')[0]
    copy_folder = os.path.join(base_path, preffix)
    shutil.copyfile(os.path.join(folder_path, file) , os.path.join(copy_folder, file))

# for j in tqdm(files):
#     preffixx = j.split('_')[0]
#     cp_path = os.path.join(base_path, preffixx)
#     shutil.copyfile(os.path.join(folder_path, j), os.path.join(cp_path, j))