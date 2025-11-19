
import numpy as np
import cv2, os
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import gc

def sliding_window_inference(model, image, window_size=16, stride=16, center_patch_size=16, class_num=5):
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
                color = (0, 255, 0)  # Cavity GREEN
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
# model = tf.keras.models.load_model("../train_batch/result_of_sheduler_with_min_lr_1e-6/models/Nadam/Nadam_earlystopped_best_epoch30.keras")
# adam = "../train_batch/result_of_sheduler_with_min_lr_1e-6/models/Adam/Adam_earlystopped_best_epoch40.keras"
# rmsprop = "../train_batch/result_of_sheduler_with_min_lr_1e-6/models/Adagrad/Adagrad_earlystopped_best_epoch53.keras"
# adagrad = "../train_batch/result_of_sheduler_with_min_lr_1e-6/models/RMSprop/RMSprop_earlystopped_best_epoch35.keras"
# adadelta = "../train_batch/result_of_sheduler_with_min_lr_1e-6/models/Adadelta/Adadelta_earlystopped_best_epoch38.keras"
# nadam = "../train_batch/result_of_sheduler_with_min_lr_1e-6/models/Nadam/Nadam_earlystopped_best_epoch30.keras"
adadelta = "/mnt/d/Codes/DenseNet-Project/src/june10_coal2ndAttempt_FINAL/models/models_nov18/Adadelta/checkpoint_best_weights.keras"
adagrad =  "/mnt/d/Codes/DenseNet-Project/src/june10_coal2ndAttempt_FINAL/models/models_nov18/Adagrad/checkpoint_best_weights.keras"
adam =  "/mnt/d/Codes/DenseNet-Project/src/june10_coal2ndAttempt_FINAL/models/models_nov18/Adam/checkpoint_best_weights.keras"
adamw =  "/mnt/d/Codes/DenseNet-Project/src/june10_coal2ndAttempt_FINAL/models/models_nov18/AdamW/checkpoint_best_weights.keras"
nadam =  "/mnt/d/Codes/DenseNet-Project/src/june10_coal2ndAttempt_FINAL/models/models_nov18/Nadam/checkpoint_best_weights.keras"
rmsprop =  "/mnt/d/Codes/DenseNet-Project/src/june10_coal2ndAttempt_FINAL/models/models_nov18/RMSprop/checkpoint_best_weights.keras"
# CNNmodelJUNE24 = "../models/CNNmodelJUNE24.keras"
model_ary = [adadelta, adagrad, adam, adamw, nadam, rmsprop]
# model_ary = [adam, rmsprop, adagrad, adadelta]

for model_name in model_ary:
    model = tf.keras.models.load_model(model_name) 
    # result_name = "TESTING_ON_REMOVED_SCALE"
    result_name = model_name.split('/')[-1].split('.')[0]
    print(f"Currently running on {result_name} model.")
    result_folder = os.path.join("../results/septmber12", result_name)
    os.makedirs(result_folder, exist_ok=True)

    count = 0
    image_folder = os.path.join(result_folder, "images")
    os.makedirs(image_folder, exist_ok=True)
    # For multiple images & multiple folders
    # old txt file is in utils folder
    with open(os.path.join(result_folder, "final_output_new_2.txt"), 'w') as f:
        # path_location = r"D:\NML 2nd working directory\DEEP SOUMYA 14-july-25\save"
        # path_location = r"D:/NML 2nd working directory/DEEP SOUMYA 14-july-25/final32New"
        path_location = r"/mnt/d/NML 2nd working directory/DEEP SOUMYA 14-july-25/final32New"
        outer_folders = os.listdir(path_location)
        half = []
        for i in outer_folders:
            if i.startswith('DBM'):
                half.append(i)
        print("Half: ", half)
        for outer_folder in half:
            total_mineral_percentage = 0
            total_images = 0
            folder_path = os.path.join(path_location, outer_folder)
            files = []
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
                # saving the image
                # plt.figure(figsize=(24, 12))
                # plt.subplot(1, 2, 1)
                # plt.imshow(img)
                # plt.title("Input Petrography Image", fontsize=16)
                # plt.axis('off')
                # plt.subplot(1, 2, 2)
                # plt.imshow(heatmap)
                # plt.title("Sequential CNN", fontsize=16)
                # plt.axis('off')

                # # Add text below the plots
                # plt.figtext(0.5, 0.12, 
                #     f"Cavity Green: {cavity_percentage} % |  Cavity Filled Blue: {cavity_filled_percentage} %  |  Inertinite Red: {inertinite_percentage} %  |  Minerals Yellow: {minerals_percentage} %  |  Vitrinite Purple: {vitrinite_percentage} %", 
                #     wrap=True, horizontalalignment='center', fontsize=20)
                # plt.savefig(f"/mnt/d/NML 2nd working directory/MY_test_result/hello/{file_name}_{count}_comparison.png" )
                # plt.savefig(f"D:/NML 2nd working directory/MY_test_result/hello/{file_name}_{count}_comparison.png" )
                # plt.close()
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