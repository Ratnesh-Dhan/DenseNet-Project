import os, cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

CLASS_MAP = {
    1: "ingot",
    2: "side_face",
    3: "bg"
}

def create_voc_xml(filename, img_shape, boxes, labels, save_path):
    height, width, depth = img_shape

    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box.astype(int)

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = CLASS_MAP[label]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    xml_str = ET.tostring(annotation)
    parsed = minidom.parseString(xml_str)
    
    with open(save_path, "w") as f:
        f.write(parsed.toprettyxml(indent="  "))

if __name__ == "__main__":
    # INITIALIZE THE MODEL
    NUM_CLASSES = 4
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("../fasterrcnn_best.pth", map_location=device))
    model.to(device)
    model.eval()
    
    OUTPUT_ROOT = "./cvat_dataset"
    IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
    ANN_DIR = os.path.join(OUTPUT_ROOT, "annotations")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(ANN_DIR, exist_ok=True)

    SCORE_THRESH = 0.55

    for image in all_images:
        img_path = os.path.join(base_path, "dataset_kanika", image)

        img_bgr = cv2.imread(img_path)
        h, w = img_bgr.shape[:2]

        new_w = 800
        scale = new_w / w
        new_h = int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h))

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = T.ToTensor()(img_rgb).to(device)

        with torch.no_grad():
            output = model([img_tensor])[0]

        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

        filtered_boxes = []
        filtered_labels = []

        for box, score, label in zip(boxes, scores, labels):
            if score >= SCORE_THRESH:
                filtered_boxes.append(box)
                filtered_labels.append(label)

        # save image
        save_img_path = os.path.join(IMAGES_DIR, image)
        cv2.imwrite(save_img_path, img_bgr)

        # save xml
        xml_name = os.path.splitext(image)[0] + ".xml"
        xml_path = os.path.join(ANN_DIR, xml_name)

        create_voc_xml(
            filename=image,
            img_shape=img_bgr.shape,
            boxes=filtered_boxes,
            labels=filtered_labels,
            save_path=xml_path
        )