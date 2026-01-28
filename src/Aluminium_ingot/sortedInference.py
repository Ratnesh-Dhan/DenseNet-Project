from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "./model/best.pt"
SOURCE = "./images"          # image path OR folder
IMG_SIZE = 896
CONF = 0.25 #0.25
DEVICE = 0
CLASS_NAMES = {
    0: "ingot",
    1: "side_face",
    # 2: "bg"
}
SAVE = True
OUT_DIR = "bbox_results"
# ---------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

results = model.predict(
    source=SOURCE,
    imgsz=IMG_SIZE,
    conf=CONF,
    # iou=0.6,
    device=DEVICE,
    verbose=False
)

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


for r in results:
    img = cv2.imread(r.path)
    if img is None:
        continue

    side_face = []
    ingots = []

    if r.boxes is not None:
        for box, cls, score in zip(
            r.boxes.xyxy, r.boxes.cls, r.boxes.conf
        ):
            x1, y1, x2, y2 = map(int, box.tolist())
            cls_id = int(cls)
            conf = float(score)

            label = f"{CLASS_NAMES.get(cls_id, cls_id)} {conf:.2f}"
            label_name = label.split(' ')[0]

            if label_name == 'ingot':
                ingots.append({'coordinates': [x1, y1, x2, y2],
                                'conf': conf})
            elif label_name == 'side_face':
                side_face.append({'coordinates': [x1, y1, x2, y2],
                                'conf': conf})
    # Apply class-wise NMS
    ingots = nms(ingots, iou_thresh=0.4)
    side_face = nms(side_face, iou_thresh=0.4)

    print("INGOTS")                       
    for ingot in ingots:
        print(ingot)
    print("\nSIDE FACE")
    for side in side_face:
        print(side)

    no_of_ingots = 0
    no_of_side_faces = 0

    for i in ingots:
        if i['conf'] > 0.42: 
            # draw box and label
            x1, y1, x2, y2 = i['coordinates']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            no_of_ingots = no_of_ingots + 1
            cv2.putText(
                img,
                f'ingot {round(i['conf'], 2)}',
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 255, 0),
                1
            )
    for s in side_face:
        x1, y1, x2, y2 = s['coordinates']
        cv2.rectangle(img, (x1, y1), (x2, y2), (252, 23, 3), 2)
        no_of_side_faces = no_of_side_faces + 1
        cv2.putText(
            img,
            f'side_face {round(s['conf'],2)}',
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (252, 23, 3),
            1
        )

    # show image
    plt.imshow( img)
    plt.title(f"Ingots : {no_of_ingots}; Side Faces : {no_of_side_faces}")
    plt.axis("off")
    plt.savefig(os.path.join('bbox_results', os.path.basename(r.path)), dpi=300, bbox_inches='tight')
    plt.show()