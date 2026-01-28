import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ---- LOAD SAM ----
MODEL_TYPE = "vit_b"
CHECKPOINT = f"./model/sam_{MODEL_TYPE}.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device="cpu")  # CPU-safe

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    min_mask_region_area=1500
)

# ---- LOAD IMAGE ----
img = cv2.imread("./images/ingot1.jpg")
if img is None:
    raise RuntimeError("Image not found. Try again.")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---- GENERATE MASKS ----
masks = mask_generator.generate(rgb)

# ---- DRAW RESULTS ----
overlay = img.copy()

for i, mask in enumerate(masks):
    seg = mask["segmentation"]

    # bounding box
    y, x = np.where(seg)
    if len(x) == 0:
        continue

    x1, x2 = x.min(), x.max()
    y1, y2 = y.min(), y.max()

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # overlay mask
    color = np.array([0, 255, 0], dtype=np.uint8)
    overlay[seg] = overlay[seg] * 0.6 + color * 0.4

# ---- SHOW ----
cv2.namedWindow("SAM Segmentation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SAM Segmentation", 1200, 800)
cv2.imshow("SAM Segmentation", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
