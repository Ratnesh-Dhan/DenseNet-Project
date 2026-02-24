from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

import numpy as np
import cv2, os
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import gc


def sliding_window_inference(model, image, window_size=16, stride=16, class_num=5):

    h, w, _ = image.shape

    # --- Extract patches on GPU ---
    image_tf = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0
    image_tf = tf.expand_dims(image_tf, axis=0)

    patches = tf.image.extract_patches(
        images=image_tf,
        sizes=[1, window_size, window_size, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )

    patches = tf.reshape(patches, [-1, window_size, window_size, 3])

    # --- Batched inference ---
    batch_size = 1024  # Safe for 3060 Ti (8GB VRAM)
    predictions = []

    for i in range(0, patches.shape[0], batch_size):
        batch = patches[i:i+batch_size]
        preds = model(batch, training=False)
        predictions.append(preds)

    predictions = tf.concat(predictions, axis=0)
    predicted_classes = tf.argmax(predictions, axis=1).numpy()

    # --- Count classes (vectorized) ---
    counts = np.bincount(predicted_classes, minlength=class_num)

    cavity = counts[0] if class_num > 0 else 0
    cavity_filled = counts[1] if class_num > 1 else 0
    inertinite = counts[2] if class_num > 2 else 0
    minerals = counts[3] if class_num > 3 else 0
    vitrinite = counts[4] if class_num > 4 else 0

    return cavity, cavity_filled, inertinite, minerals, vitrinite

adam = "/home/zumbie/Codes/NML/DenseNet-Project/src/june10_coal2ndAttempt_FINAL/train_batch/models_feb23_2026/Adam/checkpoint_best_weights.keras"
adadelta = "/home/zumbie/Codes/NML/DenseNet-Project/src/june10_coal2ndAttempt_FINAL/train_batch/models_feb23_2026/Adadelta/checkpoint_best_weights.keras"
adagrad = "/home/zumbie/Codes/NML/DenseNet-Project/src/june10_coal2ndAttempt_FINAL/train_batch/models_feb23_2026/Adagrad/checkpoint_best_weights.keras"
adamw = "/home/zumbie/Codes/NML/DenseNet-Project/src/june10_coal2ndAttempt_FINAL/train_batch/models_feb23_2026/AdamW/checkpoint_best_weights.keras"
nadam = "/home/zumbie/Codes/NML/DenseNet-Project/src/june10_coal2ndAttempt_FINAL/train_batch/models_feb23_2026/Nadam/checkpoint_best_weights.keras"
rmsprop = "/home/zumbie/Codes/NML/DenseNet-Project/src/june10_coal2ndAttempt_FINAL/train_batch/models_feb23_2026/RMSprop/checkpoint_best_weights.keras"
model_ary = [adam , adadelta, adagrad, adamw, nadam, rmsprop]

for model_name in model_ary:
    model = tf.keras.models.load_model(model_name) 
    # result_name = "TESTING_ON_REMOVED_SCALE"
    # result_name = model_name.split('/')[-1].split('.')[0]
    result_name = model_name.split('/')[-2]
    print(f"Currently running on {result_name} model.")
    # result_folder = os.path.join("../results/septmber12", result_name)
    result_folder = os.path.join("../train_batch/results/feb23_26", result_name)
    os.makedirs(result_folder, exist_ok=True)

    count = 0
    image_folder = os.path.join(result_folder, "images")
    os.makedirs(image_folder, exist_ok=True)
    # For multiple images & multiple folders
    # old txt file is in utils folder
    with open(os.path.join(result_folder, "final_output_new_full.txt"), 'w') as f:
        # path_location = r"D:\NML 2nd working directory\DEEP SOUMYA 14-july-25\save"
        # path_location = r"D:/NML 2nd working directory/DEEP SOUMYA 14-july-25/final32New"
        # path_location = r"/mnt/d/DATASETS/coal2026_Full_Images/"
        path_location = r"/media/zumbie/6CA45A53A45A203E/2026-coal_samples/Himanshu Coal Samples 2026"
        outer_folders = sorted(os.listdir(path_location))
        # half = []
        # for i in outer_folders:
        #     if i.startswith('D'):
        #         half.append(i)
        # print("Half: ", half)
        for outer_folder in outer_folders:
            total_mineral_percentage = 0
            total_images = 0
            folder_path = os.path.join(path_location, outer_folder)
            files = []
            files = os.listdir(folder_path)
            total_images = len(files)  
            print("Total files : ", total_images)
            actual_files = []
            for actual_file in files:
                if actual_file.endswith(".jpg"):
                    actual_files.append(actual_file)
            if len(actual_files) == 0:
                continue
            for file_name in actual_files:
                img = plt.imread(os.path.join(folder_path, file_name))
                img = np.array(img, copy=True)  # Make it writable
                img = cv2.rectangle(img, (2146, 30), (2572, 162), (0, 0, 0), -1)  # Black rectangle with thickness=-1 for filling
                cavity, cavity_filled, inertinite, minerals, vitrinite = sliding_window_inference(model, img, class_num=5)

                total_number = cavity + cavity_filled + inertinite + minerals + vitrinite
                cavity_percentage = round((cavity/total_number)*100, 2)
                cavity_filled_percentage = round((cavity_filled/total_number)*100, 2)
                inertinite_percentage = round((inertinite/total_number)*100, 2)
                minerals_percentage = round((minerals/total_number)*100, 2)
                vitrinite_percentage = round((vitrinite/total_number)*100, 2)

                # Adding mineral % 
                total_mineral_percentage = total_mineral_percentage + minerals_percentage + cavity_filled_percentage
                count = count + 1                

            # Average ash % .
            print(f'Folder name = {outer_folder}')
            print(f"Optimizer used = {model_name.split('/')[-2]}")
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