from PIL import Image
import numpy as np, cv2, os

# directory_path = r"D:\NML ML Works\newCoalByDeepBhaiya\TRAINING 31\2 Inertinite"
# save_path = r"D:\NML ML Works\newCoalByDeepBhaiya\TRAINING 15\2 Inertinite"
to_base_path = r"D:\NML ML Works\newCoalByDeepBhaiya\16\TRAINING 16"
from_base_path = r"D:\NML ML Works\newCoalByDeepBhaiya\31\TRAINING 31"

to_files = os.listdir(to_base_path)
print(to_files)
for i in to_files:
    print("inside 1st loop")
    directory_path = os.path.join(from_base_path, i)
    print(directory_path)
    save_path = os.path.join(to_base_path, i)
    files = os.listdir(directory_path)
    print(files[:20])
    # count = 1
    # total = len(files)
    for file in files:
        print("insdie 2nd loop")
        image = Image.open(os.path.join(directory_path,file))
        file = file.split('.')[0]
        # image.show()
        image = np.array(image)
        x1, y1 = (0,0)
        x2, y2 = (16, 16)
        new_image = image[y1: y2, x1: x2]
        cv2.imwrite(os.path.join(save_path, f"{file}_TL.png"), cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

        x1, y1 = (14, 0)
        x2, y2 = (30, 16)
        new_image = image[y1: y2, x1: x2]
        cv2.imwrite(os.path.join(save_path, f"{file}_TR.png"), cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

        x1, y1 = (14, 14)
        x2, y2 = (30, 30)
        new_image = image[y1: y2, x1: x2]
        cv2.imwrite(os.path.join(save_path, f"{file}_BR.png"), cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

        x1, y1 = (0, 14)
        x2, y2 = (16, 30)
        new_image = image[y1: y2, x1: x2]
        cv2.imwrite(os.path.join(save_path, f"{file}_BL.png"), cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        # print(f"{count} have completed out of {total}")
        # count = count + 1 