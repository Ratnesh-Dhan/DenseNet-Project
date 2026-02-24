import cv2
import torch
import os
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ---- LOAD SAM ----
MODEL_TYPE = "vit_b"
CHECKPOINT = f"./model/sam_{MODEL_TYPE}.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
# sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
sam.to(device="cpu")

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    min_mask_region_area=1500
)

# ---- VIDEO ----
cap = cv2.VideoCapture("Aluminum_Ingot.mp4")

cv2.namedWindow("SAM Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SAM Detection", 500, 900)

FRAME_SKIP = 20
frame_id = 0
last_masks = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_small = cv2.resize(frame, (640, 360))
    rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    # ---- RUN SAM EVERY N FRAMES ----
    if frame_id % FRAME_SKIP == 0:
        last_masks = mask_generator.generate(rgb)

    # ---- DRAW USING LAST MASKS ----
    for mask in last_masks:
        seg = mask["segmentation"]

        y, x = np.where(seg)
        if len(x) == 0:
            continue

        x1, x2 = x.min(), x.max()
        y1, y2 = y.min(), y.max()
        sx = frame.shape[1] / 640
        sy = frame.shape[0] / 360

        x1 = int(x1 * sx)
        x2 = int(x2 * sx)
        y1 = int(y1 * sy)
        y2 = int(y2 * sy)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        color = np.array([0, 255, 0], dtype=np.uint8)
        frame[seg] = frame[seg] * 0.6 + color * 0.4

    cv2.imshow("SAM Detection", frame)

    frame_id += 1  # <-- YOU FORGOT THIS
    print(frame_id)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
