import os

path_images = "../../../../Datasets/Traffic_Dataset/images/train/"
path_annot = "../../../../Datasets/Traffic_Dataset/labels/train/"

files = os.listdir(path_annot)

for file in files[:10]:
    with open(os.path.join(path_annot, file), "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line, end="")
    f.close()