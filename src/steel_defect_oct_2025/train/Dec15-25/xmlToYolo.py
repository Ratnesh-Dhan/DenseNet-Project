import os
import xml.etree.ElementTree as ET

# MUST match data.yaml exactly
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

SPLIT = "test"  # change to val when needed

ANN_DIR = f"/mnt/d/DATASETS/NEU-DET/annotations/{SPLIT}"
OUT_DIR = f"/mnt/d/DATASETS/NEU-DET/labels/{SPLIT}"

os.makedirs(OUT_DIR, exist_ok=True)

for file in os.listdir(ANN_DIR):
    if not file.endswith(".xml"):
        continue

    xml_path = os.path.join(ANN_DIR, file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in CLASSES:
            print(f"this in not in classes: {cls_name}")
            print(CLASSES)
            continue

        class_id = CLASSES.index(cls_name)

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        # YOLO sanity check
        if not (0 < x_center < 1 and 0 < y_center < 1):
            continue

        yolo_lines.append(
            f"{class_id} "
            f"{x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}"
        )

    out_file = os.path.join(
        OUT_DIR,
        file.replace(".xml", ".txt")
    )

    # even if empty, YOLO wants the file
    with open(out_file, "w") as f:
        f.write("\n".join(yolo_lines))

print("XML → YOLO conversion finished. Train like an adult now.")
