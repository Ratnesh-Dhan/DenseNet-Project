import numpy as np
import cv2, os, sys
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

color_map = {
    "Cavity Green (Black)": (0, 0, 0),
    "Cavity Filled (Blue)": (255, 0, 0),
    "Inertinite (Red)": (0, 0, 255),
    "Minerals (Yellow)": (0, 255, 255),
    "Vitrinite (Purple)": (128, 0, 128)
}

def sliding_window_inference(model, image, window_size=16, stride=8, center_patch_size=16):
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


# model = tf.keras.models.load_model("../models/newCnnEpoch25.keras")
# model = tf.keras.models.load_model("../models/newCNNjune13Epoch_100.keras")
# model = tf.keras.models.load_model("../models/EarlyStoppedBest11June.keras") # This is good
# model = tf.keras.models.load_model("../models/modelJUNE11.keras") # This is better
model = tf.keras.models.load_model("../models/CNNmodelJUNE24.keras") # This is best
# model = tf.keras.models.load_model("../models/newCnnEpoch25.keras") # This is not best

# For single image
try:
    file_name = "004"
    add_name = "CNN"
    file = f"D:/NML 2nd working directory/testing/{file_name}.jpg"
    img = plt.imread(file)
    img = np.array(img, copy=True)  # Make it writable
    img = cv2.rectangle(img, (2146, 30), (2572, 162), (0, 0, 0), -1)  # Black rectangle with thickness=-1 for filling

    heatmap, cavity, cavity_filled, inertinite, minerals, vitrinite = sliding_window_inference(model, img)

    total_number = cavity + cavity_filled + inertinite + minerals + vitrinite
    cavity_percentage = round((cavity/total_number)*100, 2)
    cavity_filled_percentage = round((cavity_filled/total_number)*100, 2)
    inertinite_percentage = round((inertinite/total_number)*100, 2)
    minerals_percentage = round((minerals/total_number)*100, 2)
    vitrinite_percentage = round((vitrinite/total_number)*100, 2)

    # cv2.imwrite(f"./results/{file_name}_{add_name}_heatmap.png", cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
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
        f"Cavity Black: {cavity_percentage} % |  Cavity Filled Blue: {cavity_filled_percentage} %  |  Inertinite Red: {inertinite_percentage} %  |  Minerals Yellow: {minerals_percentage} %  |  Vitrinite Purple: {vitrinite_percentage} %", 
        wrap=True, horizontalalignment='center', fontsize=20)

    # Organic and Inorganic text 
    plt.figtext(0.5, 0.06,
                f"Organic: {round(inertinite_percentage+vitrinite_percentage, 2)} % | Inorganic: {round(minerals_percentage+cavity_filled_percentage,2)} %",
                wrap=True, horizontalalignment='center', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Leave space at bottom for the text

    # plt.savefig(f"../results/{file_name}_{add_name}_comparison.png" )
    plt.show()


    # === Configuration ===
    h, w, _ = heatmap.shape
    total_pixels = h * w

    # Define exact BGR colors for each class (used during rectangle drawing)
    color_map = {
        "Cavity Green (Black)": (0, 0, 0),
        "Cavity Filled (Blue)": (0, 0, 255),
        "Inertinite (Red)": (255, 0, 0),
        "Minerals (Yellow)": (255, 255, 0),
        "Vitrinite (Purple)": (128, 0, 128)
    }

    # === Pixel counting ===
    color_pixel_counts = {}
    total_counted = 0
    all_known_mask = np.zeros((h, w), dtype=bool)

    for class_name, bgr_color in color_map.items():
        mask = np.all(heatmap == bgr_color, axis=-1)
        count = np.sum(mask)
        color_pixel_counts[class_name] = count
        total_counted += count
        all_known_mask |= mask  # build up mask of all known pixels

    # Undetected pixels = anything not matching class colors
    undetected_pixels = total_pixels - total_counted
    undetected_percent = round((undetected_pixels / total_pixels) * 100, 2)

    # === Output Summary ===
    print(f"\n=== Pixel Coverage Summary from Heatmap ===")
    for class_name, count in color_pixel_counts.items():
        percent = round((count / total_pixels) * 100, 2)
        print(f"{class_name:<25}: {count:>8} pixels  ({percent:>5}%)")

    print(f"{'Undetected / Background':<25}: {undetected_pixels:>8} pixels  ({undetected_percent:>5}%)")
    print(f"{'Total Pixels in Image':<25}: {total_pixels:>8} pixels")
    print(f"{'Sum of All Counted Pixels':<25}: {total_counted + undetected_pixels:>8} pixels")
    print(f"{'Pixel Match Check':<25}: {'✅ OK' if (total_counted + undetected_pixels) == total_pixels else '❌ ERROR'}")

except Exception as e:
    print(f'Error: {e}')