import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import cv2
import os
from matplotlib import pyplot as plt
from torchvision.utils import save_image
# from paddleocr import TextRecognition
# PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = True
from manga_ocr import MangaOcr
from PIL import Image

NUM_CLASSES = 3 # background + japanese + english

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("../model/best_model.pth", map_location=device))
model.to(device)
model.eval()

img_path = "../images/18.webp"

img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

img_tensor = T.ToTensor()(img_rgb).to(device)

with torch.no_grad():
    output = model([img_tensor])[0]

boxes = output["boxes"].cpu().numpy()
scores = output["scores"].cpu().numpy()
labels = output["labels"].cpu().numpy()

SCORE_THRESH = 0.55

box_ary = []

for box, score, label in zip(boxes, scores, labels):
    if score < SCORE_THRESH:
        continue

    x1, y1, x2, y2 = map(int, box)
    crop = img_rgb[y1:y2, x1:x2]
    box_ary.append({"crop": crop, "label": label, "score": score, "x1": x1, "y1": y1, "x2": x2, "y2": y2})


manga_ocr = MangaOcr()
for box in box_ary:
    crop = box["crop"]
    label = box["label"]
    score = box["score"]
    x1 = box["x1"]
    y1 = box["y1"]
    x2 = box["x2"]
    y2 = box["y2"]
    image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    # rec_model = TextRecognition(model_name="PP-OCRv5_server_rec")
    # result = rec_model.predict(input=crop)
    
    # for res in result:
    #     print(res['rec_text'])
    #     # In 3.0, the result object stores data in these keys
    text = manga_ocr(image)
    print(text)
    print("--------------------------------")