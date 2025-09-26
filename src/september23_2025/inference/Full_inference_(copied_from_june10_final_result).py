
import numpy as np
import cv2, os
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import gc

def sliding_window_inference(model, image, window_size=16, stride=16, center_patch_size=16):
    cavity_green = 0
    cavity_filled_blue = 0
    inertinte_red = 0
    mineral_yellow = 0
    vitrinite_purple = 0
    h, w, _ = image.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    half_patch = window_size // 2
    half_center = center_patch_size // 2

    patches = []
    coords = []

    def skip_scale(h, w, x, y)-> bool:
        upper_x = 78
        upper_y = 9
        lower_x = 83
        lower_y = 96
        if round((y/h)*100) < upper_y:
            if round((x/w)*100) >= upper_x:
                return True
        
        elif round((y/h)*100) >= lower_y:
            if round((x/w)*100) >= lower_x:
                return True
        
        return False

    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            patch = image[y:y+window_size, x:x+window_size].astype(np.float32) / 255.0
            flag = skip_scale(h, w, x, y)
            if flag:
                break
            patches.append(patch)
            coords.append((x, y))

    patches = np.array(patches)
    predictions = model.predict(patches, verbose=1)

    for (x, y), prediction in tqdm(zip(coords, predictions), total=len(predictions)):
        pred_class = np.argmax(prediction)
        center_y = y + half_patch
        center_x = x + half_patch

        if pred_class == 0:
            color = (0, 255, 0)  # Cavity GREEN
            cavity_green = cavity_green + 1
        elif pred_class == 1:
            color = (0, 0, 255)  # Cavity filled BLUE
            cavity_filled_blue = cavity_filled_blue + 1
        elif pred_class == 2:
            color = (255, 0, 0)  # Inertinite RED
            inertinte_red = inertinte_red + 1
        elif pred_class == 3:
            color = (255, 255, 0)  # Minerals YELLOW
            mineral_yellow = mineral_yellow + 1
        else:  # pred_class == 4
            color = (128, 0, 128)  # Vitrinite PURPLE
            vitrinite_purple = vitrinite_purple + 1

        cv2.rectangle(
            heatmap,
            (center_x - half_center, center_y - half_center),
            (center_x + half_center, center_y + half_center),
            color,
            thickness=-1
        )

    return heatmap, cavity_green, cavity_filled_blue, inertinte_red, mineral_yellow, vitrinite_purple

EarlyStoppedBestSeptmber24 = "../models/EarlyStoppedBestSeptmber24.keras"
Septmber24 = "../models/Septmber24.keras"
model_ary = [EarlyStoppedBestSeptmber24, Septmber24]

for model_name in model_ary:
    model = tf.keras.models.load_model(model_name) 
    # result_name = "TESTING_ON_REMOVED_SCALE"
    result_name = model_name.split('/')[-1].split('.')[0]
    print(f"Currently running on {result_name} model.")
    result_folder = os.path.join("../results/septmber25", result_name)
    os.makedirs(result_folder, exist_ok=True)

    count = 0

    # For multiple images & multiple folders
    with open(os.path.join(result_folder, "final_output.txt"), 'w') as f:
        # path_location = r"D:\NML 2nd working directory\DEEP SOUMYA 14-july-25\save"
        # path_location = r"D:/NML 2nd working directory/DEEP SOUMYA 14-july-25/final32New"
        path_location = r"/mnt/d/NML 2nd working directory/DEEP SOUMYA 14-july-25/final32New"
        outer_folders = os.listdir(path_location)
        print(outer_folders)
        print(f"Toatal number of subfolders : {len(outer_folders)}")
        # half = []
        # for i in outer_folders:
        #     if i.startswith('DBM'):
        #         half.append(i)
        # print("Half: ", half)
        for outer_folder in outer_folders:
            total_mineral_percentage = 0
            total_images = 0
            folder_path = os.path.join(path_location, outer_folder)
            files = os.listdir(folder_path)
            total_images = len(files)  
            print("Total files : ", total_images)

            for file_name in files:
                img = plt.imread(os.path.join(folder_path, file_name))
                img = np.array(img, copy=True)  # Make it writable
                img = cv2.rectangle(img, (2146, 30), (2572, 162), (0, 0, 0), -1)  # Black rectangle with thickness=-1 for filling
                heatmap, cavity, cavity_filled, inertinite, minerals, vitrinite = sliding_window_inference(model, img)

                total_number = cavity + cavity_filled + inertinite + minerals + vitrinite
                cavity_percentage = round((cavity/total_number)*100, 2)
                cavity_filled_percentage = round((cavity_filled/total_number)*100, 2)
                inertinite_percentage = round((inertinite/total_number)*100, 2)
                minerals_percentage = round((minerals/total_number)*100, 2)
                vitrinite_percentage = round((vitrinite/total_number)*100, 2)

                # Adding mineral % 
                total_mineral_percentage = total_mineral_percentage + minerals_percentage + cavity_filled_percentage
                count = count + 1
                del img, heatmap
                

            # Average ash % .
            print(f'Folder name = {outer_folder}')
            print("Total miniral % = ", total_mineral_percentage)
            average = total_mineral_percentage/total_images
            print("Average mineral % = ", average)
            print("Average ash % = ", average/1.1)

            f.write(f'{outer_folder} Total mineral %: {total_mineral_percentage}\n')
            f.write(f'{outer_folder} Average mineral %: {average}\n')
            f.write(f'{outer_folder} Average ash %: {average/1.1}\n')
            f.write('-' * 40 + '\n')
    tf.keras.backend.clear_session()
    gc.collect()