import cv2, os
from tqdm import tqdm
import sys

folder_name = 'C2'
folderPath = fr"D:\NML 2nd working directory\DEEP SOUMYA 14-july-25\final\{folder_name}"
files = os.listdir(folderPath)
files_full_path = []
for f in files:
    files_full_path.append(os.path.join(folderPath, f))

# for file in files_full_path:
#     image = cv2.imread(file)
#     print(image.shape)


# image = cv2.imread(r"D:\NML 2nd working directory\DEEP SOUMYA 14-july-25\save\SCALE_REMOVED\C20\001.jpg")
# print(image.shape)

# base_path = r"D:\NML 2nd working directory\DEEP SOUMYA 14-july-25"
base_path = fr"D:\NML 2nd working directory\DEEP SOUMYA 14-july-25\final\{folder_name}"
# location_path = os.path.join(base_path, 'final')
# save_path = os.path.join(base_path, 'save')
save_path = fr"D:\NML 2nd working directory\DEEP SOUMYA 14-july-25\save\{folder_name}"
os.makedirs(save_path, exist_ok=True)
all_files = os.listdir(base_path)
for file in tqdm(all_files, total=len(all_files)):
        image = cv2.imread(os.path.join(base_path, file))
        # This is traial
        # Assume 'image' is already loaded
        height, width = image.shape[:2]

        # Rectangle defined as percentages of width and height
        # horizontal distance of rectangle / image width, vertical distance of rectangle / image height
        x1_percent, y1_percent = 0.822, 0.0145
        # horizontal distance of rectangle 2nd point / image width, vertical distance of rectangle 2nd point / image height
        x2_percent, y2_percent = 0.982, 0.0628

        x1_percent, y1_percent = 0.83912, 0.9621
        x2_percent, y2_percent = 0.9965, 0.9938

        # Convert back to absolute coordinates
        x1 = int(x1_percent * width) -6 
        y1 = int(y1_percent * height)
        x2 = int(x2_percent * width) +6
        y2 = int(y2_percent * height)

        # Draw the rectangle
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
        # image = cv2.rectangle(image, (1007, 12), (1204, 52), (0,0,0), -1)
        file_save_name_location = os.path.join(save_path, file)
        cv2.imwrite(file_save_name_location, image)



sys.exit(1)
all_folders = os.listdir(location_path)
print(all_folders)

for folder in all_folders:
    final_location_path = os.path.join(location_path, folder)
    final_save_path = os.path.join(save_path, folder)
    os.makedirs(final_save_path, exist_ok=True)
    all_files = os.listdir(final_location_path)

    for file in tqdm(all_files, total=len(all_files)):
        image = cv2.imread(os.path.join(final_location_path, file))
        image = cv2.rectangle(image, (904, 11), (1085, 55), (0,0,0), -1)
        file_save_name_location = os.path.join(final_save_path, file)
        cv2.imwrite(file_save_name_location, image)
        # cv2.imshow('MODI5D',image)    
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()