from ultralytics import YOLO

# Load the model
model = YOLO('../../../MyTrained_Models/pcbYOLO/last.pt')

# Run inference on the video
results = model('pcbOnLine.mp4', save=True)