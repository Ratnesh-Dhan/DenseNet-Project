import numpy as np
import cv2, os, sys
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

# def sliding_window_inference(model, image, window_size=31, stride=15, center_patch_size=15):
#     h, w, _f = image.shape
#     print(h,w)
#     heatmap = np.zeros((h, w, 3), dtype=np.uint8)

#     half_patch = window_size // 2
#     half_center = center_patch_size // 2

#     for y in range(15, h - window_size + 1, stride):
#         print(y)
#         for x in range(15, w - window_size + 1, stride):
#             patch = image[y:y+window_size, x:x+window_size].astype(np.float32) / 255.0
#             # patch = np.expand_dims(patch, axis=(0, -1))  # shape: (1, 31, 31, 1)
#             patch = np.expand_dims(patch, axis=0)  # for RGB 

#             prediction = model.predict(patch, verbose=0)
#             pred_class = np.argmax(prediction[0])

#             center_y = y + half_patch
#             center_x = x + half_patch
#             # color = (0, 255, 0) if pred_class == 0 else (0, 0, 255)  # Organic or Inorganic
#             # print(f'Class found: {pred_class}')
#             if pred_class == 0:
#                 color = (0, 255, 0)  # Cavity
#             elif pred_class == 1:
#                 color = (0, 0, 255)  # Cavity filled
#             elif pred_class == 2:
#                 color = (255, 0, 0)  # Inertinite
#             elif pred_class == 3:
#                 color = (255, 255, 0)  # Minerals
#             else:  # pred_class == 4
#                 color = (128, 0, 128)  # Vitrinite

#             cv2.rectangle(
#                 heatmap,
#                 (center_x - half_center, center_y - half_center),
#                 (center_x + half_center, center_y + half_center),
#                 color,
#                 thickness=-1
#             )
#     return heatmap
def sliding_window_inference(model, image, window_size=31, stride=15, center_patch_size=15):
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

    for y in range(15, h - window_size + 1, stride):
        for x in range(15, w - window_size + 1, stride):
            patch = image[y:y+window_size, x:x+window_size].astype(np.float32) / 255.0
            patches.append(patch)
            coords.append((x, y))

    patches = np.array(patches)
    predictions = model.predict(patches, verbose=1)

    for (x, y), prediction in zip(coords, predictions):
        print("patch = ",x, " : ",y)
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


model = tf.keras.models.load_model("../models/EarlyStoppedBest09June.keras")

# For single image

# file_name = "010"
# file = f"../img/{file_name}.jpg"
# img = plt.imread(f"../img/{file}")
# heatmap, cavity, cavity_filled, inertinite, minerals, vitrinite = sliding_window_inference(model, img)

# cv2.imwrite(f"./results/{file_name}_09_heatmap.png", cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
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
#     f"Cavity Green: {cavity}  |  Cavity Filled Blue: {cavity_filled}  |  Inertinite Red: {inertinite}  |  Minerals Yellow: {minerals}  |  Vitrinite Purple: {vitrinite}", 
#     wrap=True, horizontalalignment='center', fontsize=20)
# plt.tight_layout(rect=[0, 0.03, 1, 1])  # Leave space at bottom for the text

# plt.savefig(f"./results/{file_name}_09_comparison.png" )
# plt.show()


# For multiple images

folder_path = r"C:\Users\NDT Lab\Documents\DATA-20250609T100339Z-1-001\DATA\TESTING"
files = os.listdir(folder_path)
save_path = r"C:\Users\NDT Lab\Documents\DATA-20250609T100339Z-1-001\DATA\Results"
for file_name in files:
    # file = f"../img/{file_name}.jpg"
    img = plt.imread(os.path.join(folder_path, file_name))
    heatmap, cavity, cavity_filled, inertinite, minerals, vitrinite = sliding_window_inference(model, img)

    cv2.imwrite(os.path.join(save_path, f"{file_name}09_heatmap.png"), cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input Petrography Image", fontsize=16)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap)
    plt.title("Sequential CNN", fontsize=16)
    plt.axis('off')

# Add text below the plots
    plt.figtext(0.5, 0.12, 
        f"Cavity Green: {cavity}  |  Cavity Filled Blue: {cavity_filled}  |  Inertinite Red: {inertinite}  |  Minerals Yellow: {minerals}  |  Vitrinite Purple: {vitrinite}", 
        wrap=True, horizontalalignment='center', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Leave space at bottom for the text
    plt.savefig(os.path.join(save_path,f"{file_name}09_comparison.png") )