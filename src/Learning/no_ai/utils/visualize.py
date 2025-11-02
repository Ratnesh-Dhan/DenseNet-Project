import cv2, os
import matplotlib.pyplot as plt

# === CONFIG ===
path_annot = "../../../../Datasets/Traffic_Dataset/labels/train/"
path_image = "../../../../Datasets/Traffic_Dataset/images/train/"
files = os.listdir(path_image)

def visualize(file, path_image, path_annot):
    image_path = os.path.join(path_image, file)          # path to the image
    label_path = os.path.join(path_annot, file.split(".")[0] + ".txt")         # file containing YOLO annotations

    # === READ IMAGE ===
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # === READ LABELS ===
    with open(label_path, "r") as f:
        lines = f.readlines()

    # === PARSE & DRAW BOXES ===
    for line in lines:
        cls, x_c, y_c, bw, bh = map(float, line.strip().split())
        cls = int(cls)

        # Convert normalized YOLO coords to pixel coords
        x_c *= w
        y_c *= h
        bw *= w
        bh *= h

        x1 = int(x_c - bw / 2)
        y1 = int(y_c - bh / 2)
        x2 = int(x_c + bw / 2)
        y2 = int(y_c + bh / 2)

        # Draw rectangle and class text
        color = (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"Class {cls}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # === SHOW RESULT ===
    plt.imshow(img)
    plt.axis("off")
    plt.show()


for file in files[:10]:
    visualize(file, path_image, path_annot)     