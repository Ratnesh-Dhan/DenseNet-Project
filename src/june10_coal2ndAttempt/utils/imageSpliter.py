from PIL import Image
import numpy as np, cv2, os

directory_path = r"D:\NML ML Works\newCoalByDeepBhaiya\TRAINING 31\2 Inertinite"
save_path = r"D:\NML ML Works\newCoalByDeepBhaiya\TRAINING 15\2 Inertinite"
files = os.listdir(directory_path)
count = 1
total = len(files)
for file in files:
    image = Image.open(os.path.join(directory_path,file))
    file = file.split('.')[0]
    # image.show()
    image = np.array(image)
    x1, y1 = (0,0)
    x2, y2 = (15, 15)
    new_image = image[y1: y2, x1: x2]
    cv2.imwrite(os.path.join(save_path, f"{file}_TL.png"), cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

    x1, y1 = (15, 0)
    x2, y2 = (30, 15)
    new_image = image[y1: y2, x1: x2]
    cv2.imwrite(os.path.join(save_path, f"{file}_TR.png"), cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

    x1, y1 = (15, 15)
    x2, y2 = (30, 30)
    new_image = image[y1: y2, x1: x2]
    cv2.imwrite(os.path.join(save_path, f"{file}_BR.png"), cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

    x1, y1 = (0, 15)
    x2, y2 = (15, 30)
    new_image = image[y1: y2, x1: x2]
    cv2.imwrite(os.path.join(save_path, f"{file}_BL.png"), cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    print(f"{count} have completed out of {total}")
    count = count + 1 