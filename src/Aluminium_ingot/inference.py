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
    # 0: "ingot",
    # 1: "side_face",
    2: "bg"
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
    device=DEVICE,
    verbose=False
)

for r in results:
    img = cv2.imread(r.path)
    if img is None:
        continue

    if r.boxes is not None:
        for box, cls, score in zip(
            r.boxes.xyxy, r.boxes.cls, r.boxes.conf
        ):
            x1, y1, x2, y2 = map(int, box.tolist())
            cls_id = int(cls)
            conf = float(score)

            label = f"{CLASS_NAMES.get(cls_id, cls_id)} {conf:.2f}"
            label_name = label.split(' ')[0]

            # draw box and label
            if label_name == 'ingot':
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    label,
                    (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            elif label_name == 'side_face':
                cv2.rectangle(img, (x1, y1), (x2, y2), (252, 23, 3), 2)
                cv2.putText(
                    img,
                    label,
                    (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (252, 23, 3),
                    2
                )

    # show image
    plt.imshow( img)
    plt.title("YOLO Bounding Boxes")
    plt.axis("off")
    plt.show()

    # save image
    if SAVE:
        out_path = os.path.join(OUT_DIR, os.path.basename(r.path))
        print(out_path)
        cv2.imwrite(out_path, img)

# cv2.destroyAllWindows()
print("✔ Inference + visualization done")
