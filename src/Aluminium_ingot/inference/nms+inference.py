import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import cv2
import os
from matplotlib import pyplot as plt

from torchvision.utils import save_image

NUM_CLASSES = 4  # background + ingot + side_face

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("../fasterrcnn_best.pth", map_location=device))
model.to(device)
model.eval()

# img_path = "/mnt/d/Codes/DenseNet-Project/src/Aluminium_ingot/images/4.jpg"

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0
def nms(boxes, iou_thresh=0.45):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    kept = []

    while boxes:
        best = boxes.pop(0)
        kept.append(best)

        boxes = [
            b for b in boxes
            if iou(best['coordinates'], b['coordinates']) < iou_thresh
        ]

    return kept
# containment ratio is the ratio of the inner area to the outer area (Containmnet Supression)
def containment_ratio(inner, outer):
    x1, y1, x2, y2 = inner
    ox1, oy1, ox2, oy2 = outer

    ix1 = max(x1, ox1)
    iy1 = max(y1, oy1)
    ix2 = min(x2, ox2)
    iy2 = min(y2, oy2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    inner_area = (x2 - x1) * (y2 - y1)

    return inter / inner_area if inner_area > 0 else 0
def area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)
def dominance_suppression(boxes, overlap_thresh=0.4, area_ratio=1.8):
    kept = []

    boxes = sorted(boxes, key=lambda x: x["conf"], reverse=True)

    for a in (boxes):
        suppress = False
        for b in kept:
            cont = containment_ratio(a["coordinates"], b["coordinates"])
            
            if cont > overlap_thresh:
                if area(b["coordinates"]) > area_ratio * area(a["coordinates"]):
                    suppress = False
                    break
        if not suppress:
            kept.append(a)

    return kept



for i in range(1, 2):
# for i in range(1, 6):
    img_path = f"/mnt/d/Codes/DenseNet-Project/src/Aluminium_ingot/images/{i}.jpg"
    img_path = f"/mnt/d/Codes/DenseNet-Project/src/Aluminium_ingot/images/1.jpg"
    img_bgr = cv2.imread(img_path)
    h, w = img_bgr.shape[:2]

    new_w = 800
    scale = new_w / w
    new_h = int(h * scale)

    img_bgr = cv2.resize(img_bgr, (new_w, new_h))

    # img_bgr = cv2.resize(img_bgr, (1200, 1200))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_tensor = T.ToTensor()(img_rgb).to(device)

    with torch.no_grad():
        output = model([img_tensor])[0]

    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()

    SCORE_THRESH = 0.55

    ingots = []
    for box, score, label in zip(boxes, scores, labels):
        if score < SCORE_THRESH:
            continue

        x1, y1, x2, y2 = map(int, box)

        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        # name = "ingot" if label == 1 else "side_face"
        if label == 1:
            name = "ingot"
            ingots.append({'coordinates': [x1, y1, x2, y2],
                                    'conf': score})
        else:
            continue

    # Containment Supression
    filtered = []
    # Dominance Suppression (Bigger boxes suppress smaller boxes)
    # ingots = dominance_suppression(ingots, overlap_thresh=0.4, area_ratio=1.8)
    # for i, a in enumerate(ingots):
    #     keep = True
    #     for j, b in enumerate(ingots):
    #         if i == j:
    #             continue

    #         # a is small, b is big, and a lies inside b
    #         if containment_ratio(a["coordinates"], b["coordinates"]) > 0.8:
    #             if a["conf"] < b["conf"]:
    #                 keep = False
    #                 break

    #     if keep:
    #         filtered.append(a)

    # filtered = nms(filtered, iou_thresh=0.5)
    for ingot in ingots:
        x1, y1, x2, y2 = ingot['coordinates']
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_bgr,
            # f"ingot {ingot['conf']:.2f}",
            f"{ingot['conf']:.2f}",
            (x2-30, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3, (0, 255, 0), 1)

    count = len(ingots)

    save_filename = f"{os.path.basename(img_path).split('.')[0]}_count:_{count}.jpg"
    save_filename = os.path.join("../bbox_results", save_filename)
    print(save_filename)
    # cv2.imwrite("image.jpg" ,img_bgr)
    # cv2.imshow(f"Ingots: {count}", img_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.imshow(img_bgr)
    plt.title(f"Ingot count: {count}")
    plt.axis("off") 
    plt.savefig(save_filename, bbox_inches='tight', dpi=300)
    plt.show()
    print("Ingot count:", count)
