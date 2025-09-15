from matplotlib import pyplot as plt
import os
import cv2, numpy as np
from tqdm import tqdm

# upto these %s
upper_x = 78
upper_y = 9
# from these %s
lower_x = 83
lower_y = 96
all = os.listdir(r"D:\NML 2nd working directory\DEEP SOUMYA 14-july-25\final")
all.remove("OTHER EXTRACED FROM DBM5")
print(all)

for folder_name in all:
# folder_name = "DBM4"
    folder_path = rf"D:\NML 2nd working directory\DEEP SOUMYA 14-july-25\final\{folder_name}"
    save_path = rf"D:\NML 2nd working directory\DEEP SOUMYA 14-july-25\final32New\{folder_name}"
    os.makedirs(save_path, exist_ok=True)
    folders = os.listdir(folder_path)

    def upper(img, x1, y1):
        y, x, _ = img.shape
        dx = (x1*x)/100
        dy = (y1*y)/100
        # cv2.rectangle(img, (dx, 0), (x, dy), (128,128,128), 2)
        return dx, dy, y, x

    for file in tqdm(folders, desc="Out of Forty"):
        image =  plt.imread(os.path.join(folder_path, file)).copy()

        # image =  plt.imread(os.path.join(folder_path, "009 (2).jpg")).copy()

        dx, dy, y, x = upper(image, upper_x, upper_y)
        # mask = np.zeros((y+2, x+2), np.uint8)
        # seed_point = (int((dx + x) / 2), int(dy / 2))
        # flooded = image.copy()
        # cv2.floodFill(flooded, mask, seedPoint=seed_point, newVal=(0,0,0))
        flooded = image
        cv2.rectangle(flooded, (int(dx), 0), (int(x), int(dy)), (0,0,0), -1)

        dx, dy, y, x = upper(image, lower_x, lower_y)
        cv2.rectangle(flooded, (int(dx), int(dy)), (int(x), int(y)), (0,0,0), -1)

        cv2.imwrite(os.path.join(save_path, file), cv2.cvtColor(flooded, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(os.path.join(save_path, file), flooded)

