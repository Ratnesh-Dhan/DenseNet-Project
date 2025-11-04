# pip install torch torchvision opencv-python matplotlib

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead

# ====================== CONFIG ======================
MODEL_PATH = "/mnt/d/Codes/DenseNet-Project/src/Learning/with_ai/models/ssd_model_fixed.pth"
CLASSES_FILE = "/mnt/d/Codes/DenseNet-Project/Datasets/Traffic_Dataset/classes.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCORE_THRESHOLD = 0.5  # Confidence threshold for predictions
OUTPUT_DIR = "../results/predictions2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== LOAD MODEL ======================
def load_model(model_path, num_classes):
    """Load trained SSD model"""
    model = ssd300_vgg16(weights=None)
    
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = [4, 6, 6, 6, 4, 4]
    
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    return model

# ====================== LOAD CLASSES ======================
def load_classes(classes_file):
    """Load class names from file"""
    with open(classes_file) as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# ====================== PREPROCESS IMAGE ======================
def preprocess_image(image_path):
    """
    Load and preprocess image for SSD inference
    Returns: preprocessed tensor, original image
    """
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()
    h, w = img.shape[:2]
    
    # Convert to tensor and resize to 300x300
    img_tensor = F.to_tensor(img)
    img_tensor = F.resize(img_tensor, [300, 300])
    
    # Normalize with ImageNet stats
    img_tensor = F.normalize(img_tensor, 
                            mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    
    return img_tensor, original_img, (w, h)

# ====================== RUN INFERENCE ======================
def predict(model, image_path, classes, score_threshold=0.5):
    """
    Run inference on a single image
    Returns: predictions dict with boxes, labels, scores
    """
    img_tensor, original_img, (orig_w, orig_h) = preprocess_image(image_path)
    
    # Add batch dimension and move to device
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    
    # Filter by score threshold
    scores = predictions['scores'].cpu().numpy()
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    high_score_idx = scores >= score_threshold
    scores = scores[high_score_idx]
    boxes = boxes[high_score_idx]
    labels = labels[high_score_idx]
    
    # Scale boxes back to original image size
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * (orig_w / 300)
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * (orig_h / 300)
    
    # Convert label indices to class names (subtract 1 for background)
    class_names = [classes[int(label) - 1] if label > 0 else "background" 
                   for label in labels]
    
    return {
        'boxes': boxes,
        'labels': labels,
        'scores': scores,
        'class_names': class_names,
        'original_image': original_img
    }

# ====================== VISUALIZE PREDICTIONS ======================
def visualize_predictions(predictions, save_path=None, show=True):
    """
    Draw bounding boxes on image with labels and scores
    """
    img = predictions['original_image']
    boxes = predictions['boxes']
    class_names = predictions['class_names']
    scores = predictions['scores']
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # Define colors for different classes
    colors = plt.cm.hsv(np.linspace(0, 1, len(set(class_names)) + 1))
    class_to_color = {name: colors[i] for i, name in enumerate(set(class_names))}
    
    # Draw each detection
    for box, class_name, score in zip(boxes, class_names, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        color = class_to_color[class_name]
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label with score
        label_text = f"{class_name}: {score:.2f}"
        ax.text(
            x1, y1 - 5,
            label_text,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
            fontsize=10,
            color='white',
            weight='bold'
        )
    
    ax.axis('off')
    plt.title(f"Detections: {len(boxes)} objects found", fontsize=14, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

# ====================== BATCH INFERENCE ======================
def predict_on_folder(model, folder_path, classes, score_threshold=0.5, max_images=None):
    """
    Run inference on all images in a folder
    """
    # Get all image files
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n🔍 Running inference on {len(image_files)} images...")
    
    all_predictions = []
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        
        try:
            predictions = predict(model, img_path, classes, score_threshold)
            
            # Save visualization
            save_path = os.path.join(OUTPUT_DIR, f"pred_{img_file}")
            visualize_predictions(predictions, save_path=save_path, show=False)
            
            all_predictions.append({
                'filename': img_file,
                'num_detections': len(predictions['boxes']),
                'predictions': predictions
            })
            
            print(f"✅ {img_file}: {len(predictions['boxes'])} objects detected")
            
        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")
    
    return all_predictions

# ====================== PRINT DETECTION SUMMARY ======================
def print_detection_summary(predictions):
    """Print summary statistics of detections"""
    print(f"\n{'='*60}")
    print("DETECTION SUMMARY")
    print(f"{'='*60}")
    
    total_detections = sum(p['num_detections'] for p in predictions)
    avg_detections = total_detections / len(predictions) if predictions else 0
    
    print(f"Total images processed: {len(predictions)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {avg_detections:.2f}")
    
    # Count detections per class
    class_counts = {}
    for pred in predictions:
        for class_name in pred['predictions']['class_names']:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\nDetections by class:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")
    
    print(f"{'='*60}\n")

# ====================== MAIN ======================
if __name__ == "__main__":
    # Load classes
    classes = load_classes(CLASSES_FILE)
    num_classes = len(classes) + 1  # +1 for background
    
    print(f"Loading model from {MODEL_PATH}...")
    print(f"Classes: {classes}")
    print(f"Device: {DEVICE}")
    
    # Load model
    model = load_model(MODEL_PATH, num_classes)
    print("✅ Model loaded successfully!\n")
    
    # ============ OPTION 1: Single Image Inference ============
    print("=" * 60)
    print("OPTION 1: Single Image Inference")
    print("=" * 60)
    
    # Replace with your image path
    single_image_path = "../../../../Datasets/Traffic_Dataset/images/test/image_001.jpg"
    
    if os.path.exists(single_image_path):
        print(f"Running inference on: {single_image_path}")
        predictions = predict(model, single_image_path, classes, SCORE_THRESHOLD)
        
        print(f"\n🎯 Found {len(predictions['boxes'])} objects:")
        for i, (class_name, score) in enumerate(zip(predictions['class_names'], predictions['scores'])):
            print(f"  {i+1}. {class_name} (confidence: {score:.3f})")
        
        # Visualize
        visualize_predictions(predictions, 
                            save_path=os.path.join(OUTPUT_DIR, "single_prediction.jpg"),
                            show=True)
    else:
        print(f"⚠️ Image not found: {single_image_path}")
    
    # ============ OPTION 2: Batch Inference on Folder ============
    print("\n" + "=" * 60)
    print("OPTION 2: Batch Inference on Folder")
    print("=" * 60)
    
    # Replace with your folder path
    # test_folder = "../../../../Datasets/Traffic_Dataset/images/test"
    test_folder = "/mnt/d/Codes/DenseNet-Project/Datasets/Traffic_Dataset/testing/test"
    
    if os.path.exists(test_folder):
        all_predictions = predict_on_folder(
            model, 
            test_folder, 
            classes, 
            SCORE_THRESHOLD,
            max_images=50  # Limit to first 20 images (remove for all)
        )
        
        print_detection_summary(all_predictions)
    else:
        print(f"⚠️ Folder not found: {test_folder}")
    
    print(f"\n✅ All predictions saved to: {OUTPUT_DIR}")


# ====================== ALTERNATIVE: WEBCAM INFERENCE ======================
def run_webcam_inference(model, classes, score_threshold=0.5):
    """
    Run real-time inference on webcam feed
    Press 'q' to quit
    """
    cap = cv2.VideoCapture(0)
    
    print("\n📹 Starting webcam inference...")
    print("Press 'q' to quit")
    
    colors = plt.cm.hsv(np.linspace(0, 1, len(classes) + 1))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        
        # Preprocess
        img_tensor = F.to_tensor(frame_rgb)
        img_tensor = F.resize(img_tensor, [300, 300])
        img_tensor = F.normalize(img_tensor, 
                                mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            predictions = model(img_tensor)[0]
        
        # Filter and scale boxes
        scores = predictions['scores'].cpu().numpy()
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        high_score_idx = scores >= score_threshold
        scores = scores[high_score_idx]
        boxes = boxes[high_score_idx]
        labels = labels[high_score_idx]
        
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * (w / 300)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * (h / 300)
        
        # Draw on frame
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.astype(int)
            class_name = classes[int(label) - 1] if label > 0 else "background"
            color = tuple((colors[int(label)] * 255)[:3].astype(int).tolist())
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('SSD Inference', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Uncomment to run webcam inference:
# run_webcam_inference(model, classes, SCORE_THRESHOLD)