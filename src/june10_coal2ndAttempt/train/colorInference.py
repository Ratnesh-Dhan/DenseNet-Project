import numpy as np
import cv2, os, sys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


# model = tf.keras.models.load_model("../models/newCnnEpoch25.keras")
model = tf.keras.models.load_model("../models/newCNNjune13Epoch_25.keras")
model = tf.keras.models.load_model("../models/newCNNjune13Epoch_100.keras")

# For single image
file_name = "005"
add_name = "CNN"
# file = f"D:/NML ML Works/Coal_Lebels/{file_name}.jpg"
# file = f"D:/NML ML Works/Deep bhaiya/TESTING2-20250611T124336Z-1-001/TESTING2/{file_name}.jpg"
# file = f"D:/NML ML Works/Coal photomicrographs/{file_name}.jpg"
# file = f"C:/Users/NDT Lab/Documents/DATA-20250609T100339Z-1-001/DATA\TESTING/{}"
file = f"C:/Users/NDT Lab/Documents/DATA-20250609T100339Z-1-001/DATA/TESTING/{file_name}.jpg"
img = plt.imread(file)
img = np.array(img, copy=True)  # Make it writable
img = cv2.rectangle(img, (2146, 30), (2572, 162), (0, 0, 0), -1)  # Black rectangle with thickness=-1 for filling

heatmap, cavity, cavity_filled, inertinite, minerals, vitrinite = sliding_window_inference(model, img, class_num=5)

total_number = cavity + cavity_filled + inertinite + minerals + vitrinite
cavity_percentage = round((cavity/total_number)*100, 2)
cavity_filled_percentage = round((cavity_filled/total_number)*100, 2)
inertinite_percentage = round((inertinite/total_number)*100, 2)
minerals_percentage = round((minerals/total_number)*100, 2)
vitrinite_percentage = round((vitrinite/total_number)*100, 2)

cv2.imwrite(f"./results/{file_name}_{add_name}_heatmap.png", cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
plt.figure(figsize=(24, 12))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Input Petrography Image", fontsize=16)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(heatmap)
plt.title("Sequential CNN", fontsize=16)
plt.axis('off')

# Create color patches
legend_patches = [
    mpatches.Patch(color=(0.0, 1.0, 0.0), label=f'Cavity: {cavity_percentage}%'),
    mpatches.Patch(color=(0.0, 0.0, 1.0), label=f'Cavity Filled: {cavity_filled_percentage}%'),
    mpatches.Patch(color=(1.0, 0.0, 0.0), label=f'Inertinite: {inertinite_percentage}%'),
    mpatches.Patch(color=(1.0, 1.0, 0.0), label=f'Minerals: {minerals_percentage}%'),
    mpatches.Patch(color=(128/255, 0.0, 128/255), label=f'Vitrinite: {vitrinite_percentage}%')
]

plt.legend(
    handles=legend_patches,
    loc='lower center',
    bbox_to_anchor=(0.65, -0.17),  # Adjust to your liking
    ncol=2,
    fontsize=16
)

# # Add text below the plots
# plt.figtext(0.5, 0.12, 
#     f"Cavity Green: {cavity_percentage} % |  Cavity Filled Blue: {cavity_filled_percentage} %  |  Inertinite Red: {inertinite_percentage} %  |  Minerals Yellow: {minerals_percentage} %  |  Vitrinite Purple: {vitrinite_percentage} %", 
#     wrap=True, horizontalalignment='center', fontsize=20)

# Organic and Inorganic text 
plt.figtext(0.5, 0.06,
            f"Organic: {round(inertinite_percentage+vitrinite_percentage, 2)} % | Inorganic: {round(minerals_percentage+cavity_filled_percentage,2)} %",
            wrap=True, horizontalalignment='center', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 1])  # Leave space at bottom for the text

# plt.savefig(f"../results/{file_name}_{add_name}_comparison.png" )
plt.show()

sys.exit(0)

# For multiple images

folder_path = r"D:\NML ML Works\Deep bhaiya\TESTING2-20250611T124336Z-1-001\TESTING2"
files = os.listdir(folder_path)
save_path = r"D:\NML ML Works\Deep bhaiya\TESTING2-20250611T124336Z-1-001\resultsWithColorCNNjune13"
if not os.path.exists(save_path):
    os.makedirs(save_path)
for file_name in files:
    img = plt.imread(os.path.join(folder_path, file_name))
    img = (img * 255).astype(np.uint8) if img.dtype == np.float32 or img.max() <= 1.0 else img  # Ensure correct dtype
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    img = cv2.rectangle(img, (2146, 30), (2572, 162), (0, 0, 0), -1)  # Draw black filled rectangle
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert back to RGB for matplotlib

    heatmap, cavity, cavity_filled, inertinite, minerals, vitrinite = sliding_window_inference(model, img, class_num=5)

    total_number = cavity + cavity_filled + inertinite + minerals + vitrinite
    cavity_percentage = round((cavity/total_number)*100, 2)
    cavity_filled_percentage = round((cavity_filled/total_number)*100, 2)
    inertinite_percentage = round((inertinite/total_number)*100, 2)
    minerals_percentage = round((minerals/total_number)*100, 2)
    vitrinite_percentage = round((vitrinite/total_number)*100, 2)

    # cv2.imwrite(os.path.join(save_path,f"{file_name}_heatmap.png"), cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input Petrography Image", fontsize=16)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap)
    plt.title("Sequential CNN", fontsize=16)
    plt.axis('off')

    
# Create color patches
    legend_patches = [
        mpatches.Patch(color=(0.0, 1.0, 0.0), label=f'Cavity: {cavity_percentage}%'),
        mpatches.Patch(color=(0.0, 0.0, 1.0), label=f'Cavity Filled: {cavity_filled_percentage}%'),
        mpatches.Patch(color=(1.0, 0.0, 0.0), label=f'Inertinite: {inertinite_percentage}%'),
        mpatches.Patch(color=(1.0, 1.0, 0.0), label=f'Minerals: {minerals_percentage}%'),
        mpatches.Patch(color=(128/255, 0.0, 128/255), label=f'Vitrinite: {vitrinite_percentage}%')
    ]

    plt.legend(
        handles=legend_patches,
        loc='lower center',
        bbox_to_anchor=(0.65, -0.17),  # Adjust to your liking
        ncol=2,
        fontsize=16
    )
        # Organic and Inorganic text 
    plt.figtext(0.5, 0.06,
                f"Organic: {round(inertinite_percentage+vitrinite_percentage, 2)} % | Inorganic: {round(minerals_percentage+cavity_filled_percentage,2)} %",
                wrap=True, horizontalalignment='center', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Leave space at bottom for the text

    plt.savefig(os.path.join(save_path, f"{file_name}_with_5_class_comparison.png" ))