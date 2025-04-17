import cv2, os, logging
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm  # Make sure to import tqdm at the top of your file

# Set logging level to ERROR to suppress lower-level logs
logging.getLogger('ultralytics').setLevel(logging.ERROR)

model_name = "../../../MyTrained_Models/pcb/best_7_april.pt"
model = YOLO(model_name)

def scan_directory_for_files(directory):
    try:
        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(".jpg")]
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")
        return []

def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

def runOnAllImages(files, model):
    for file in tqdm(files, desc="Processing images"):
        image = plt.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = model(image)
        for result in results:
            classes_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # get the class
                cls = int(box.cls[0])
                # getting class name
                class_name = classes_names[cls]
                if class_name == "Resistor" and box.conf[0] < 0.66:
                    continue
                # MOSFET to Cap
                class_name = "IC" if class_name == "MOSFET" else class_name
                colour = getColours(cls)
                image = cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(image, f'{class_name} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
        output_file = os.path.join("../../../Datasets/pcbDataset/test/results/", os.path.basename(file))  # Define output path
        cv2.imwrite(output_file, image)  # Save the image

# Example usage
directory_path = "../../../Datasets/pcbDataset/test/img/"
files = scan_directory_for_files(directory_path)
runOnAllImages(files, model)