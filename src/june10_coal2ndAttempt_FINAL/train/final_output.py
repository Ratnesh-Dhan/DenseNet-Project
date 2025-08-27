
import numpy as np
import cv2, os, sys
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

def sliding_window_inference(model, image, window_size=16, stride=8, center_patch_size=16, class_num=5):
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

    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            patch = image[y:y+window_size, x:x+window_size].astype(np.float32) / 255.0
            patches.append(patch)
            coords.append((x, y))

    patches = np.array(patches)
    predictions = model.predict(patches, verbose=1)

    for (x, y), prediction in tqdm(zip(coords, predictions), total=len(predictions)):
        pred_class = np.argmax(prediction)
        center_y = y + half_patch
        center_x = x + half_patch

        if class_num == 3:
            if pred_class == 0:
                color = (0, 0, 0)  # Cavity GREEN
                cavity_green = cavity_green + 1
            elif pred_class == 1:
                color = (255, 255, 0)  # Cavity filled YELLOW
                # color = (0, 0, 0)
                cavity_filled_blue = cavity_filled_blue + 1
            elif pred_class == 2:
                color = (128, 0, 128)  # Inertinite PURPLE
                inertinte_red = inertinte_red + 1
            elif pred_class == 3:
                color = (255, 255, 0)  # Minerals YELLOW
                mineral_yellow = mineral_yellow + 1
            else:  # pred_class == 4
                color = (128, 0, 128)  # Vitrinite PURPLE
                vitrinite_purple = vitrinite_purple + 1

        else:
            if pred_class == 0:
                color = (0, 0, 0)  # Cavity GREEN
                cavity_green = cavity_green + 1
            elif pred_class == 1:
                color = (0, 0, 255)  # Cavity filled BLUE
                # color = (0, 0, 0)  # Cavity filled BLUE
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


# model = tf.keras.models.load_model("../models/CNNmodelJUNE24.keras") # This is best
model = tf.keras.models.load_model("../train_batch/result_of_sheduler_with_min_lr_1e-6/models/Adam/Adam_earlystopped_best_epoch40.keras")

result_name = "Adam_early_stopped_epoch_40"
result_folder = os.path.join("../results", result_name)
os.makedirs(result_folder, exist_ok=True)

# For multiple images & multiple folders
# old txt file is in utils folder
with open(os.path.join(result_folder, "final_output.txt"), 'w') as f:
    path_location = r"D:\NML 2nd working directory\DEEP SOUMYA 14-july-25\final"
    outer_folders = os.listdir(path_location)
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
            heatmap, cavity, cavity_filled, inertinite, minerals, vitrinite = sliding_window_inference(model, img, class_num=5)

            total_number = cavity + cavity_filled + inertinite + minerals + vitrinite
            cavity_percentage = round((cavity/total_number)*100, 2)
            cavity_filled_percentage = round((cavity_filled/total_number)*100, 2)
            inertinite_percentage = round((inertinite/total_number)*100, 2)
            minerals_percentage = round((minerals/total_number)*100, 2)
            vitrinite_percentage = round((vitrinite/total_number)*100, 2)

            # Adding mineral % 
            total_mineral_percentage = total_mineral_percentage + minerals_percentage + cavity_filled_percentage

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