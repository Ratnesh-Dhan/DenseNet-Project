import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ======================
# CONFIG
# ======================
PTH_PATH = "fasterrcnn_best.pth"
ONNX_PATH = "fasterrcnn.onnx"
NUM_CLASSES = 4  # CHANGE THIS if you had more classes (+ background already included)

DEVICE = "cpu"  # ONNX export MUST be CPU

# ======================
# Rebuild model EXACTLY
# ======================
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=None
)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features, NUM_CLASSES
)

# ======================
# Load weights
# ======================
state_dict = torch.load(PTH_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
model.to(DEVICE)

# ======================
# Wrapper (VERY IMPORTANT)
# ONNX can't handle List[Dict]
# ======================
class FasterRCNNWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        outputs = self.model(images)

        boxes = outputs[0]["boxes"]
        scores = outputs[0]["scores"]
        labels = outputs[0]["labels"]

        return boxes, scores, labels


wrapped_model = FasterRCNNWrapper(model)

# ======================
# Dummy input
# FasterRCNN expects List[Tensor]
# ======================
dummy_image = torch.randn(3, 640, 640)
dummy_input = [dummy_image]

# ======================
# Export ONNX
# ======================
torch.onnx.export(
    wrapped_model,
    args=(dummy_input,),
    f=ONNX_PATH,
    opset_version=11,
    input_names=["images"],
    output_names=["boxes", "scores", "labels"],
    dynamic_axes={
        "images": {0: "batch"},
        "boxes": {0: "num_boxes"},
        "scores": {0: "num_boxes"},
        "labels": {0: "num_boxes"},
    },
)

print("✅ ONNX export successful:", ONNX_PATH)
