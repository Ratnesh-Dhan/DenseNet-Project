"""
inference.py

Comprehensive inference script for SSD object detection
Supports: single image, batch processing, and webcam inference
"""

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
from tqdm import tqdm

# ====================== CONFIG ======================
# MODEL_PATH = "./models/ssd_model_better_early_stopping_best.pth"  # Use best model
MODEL_PATH = "/mnt/d/Codes/DenseNet-Project/src/steel_defect_oct_2025/train/pytorch/models/ssd_model_better_early_stopping_best.pth"  # Use best model
# CLASSES_FILE = "../../../../Datasets/NEU-DET/classes.txt"
CLASSES_FILE = "/mnt/d/Codes/DenseNet-Project/Datasets/NEU-DET/classes.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCORE_THRESHOLD = 0.5  # Confidence threshold
NMS_THRESHOLD = 0.5    # Non-Maximum Suppression threshold
OUTPUT_DIR = "./results/predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== LOAD MODEL ======================
def load_model(model_path, num_classes, device):
    """Load trained SSD model"""
    print(f"Loading model from {model_path}...")
    
    model = ssd300_vgg16(weights=None)
    
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = [4, 6, 6, 6, 4, 4]
    
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
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
    Returns: preprocessed tensor, original image, original size
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
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

# ====================== NON-MAXIMUM SUPPRESSION ======================
def apply_nms(boxes, scores, labels, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove duplicate detections"""
    if len(boxes) == 0:
        return boxes, scores, labels
    
    # Convert to torch tensors if needed
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)
        labels = torch.tensor(labels)
    
    # Apply NMS per class
    keep_indices = []
    unique_labels = torch.unique(labels)
    
    for label in unique_labels:
        label_mask = labels == label
        label_boxes = boxes[label_mask]
        label_scores = scores[label_mask]
        label_indices = torch.where(label_mask)[0]
        
        # Apply NMS
        keep = torch.ops.torchvision.nms(label_boxes, label_scores, iou_threshold)
        keep_indices.extend(label_indices[keep].tolist())
    
    keep_indices = sorted(keep_indices)
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]

# ====================== RUN INFERENCE ======================
def predict(model, image_path, classes, score_threshold=0.5, nms_threshold=0.5, device='cpu'):
    """
    Run inference on a single image
    
    Args:
        model: Trained SSD model
        image_path: Path to input image
        classes: List of class names
        score_threshold: Minimum confidence score
        nms_threshold: IoU threshold for NMS
        device: Device to run inference on
    
    Returns:
        Dictionary with predictions and original image
    """
    img_tensor, original_img, (orig_w, orig_h) = preprocess_image(image_path)
    
    # Add batch dimension and move to device
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
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
    
    # Apply NMS to remove duplicate detections
    if len(boxes) > 0:
        boxes_tensor = torch.tensor(boxes)
        scores_tensor = torch.tensor(scores)
        labels_tensor = torch.tensor(labels)
        
        boxes_tensor, scores_tensor, labels_tensor = apply_nms(
            boxes_tensor, scores_tensor, labels_tensor, nms_threshold
        )
        
        boxes = boxes_tensor.numpy()
        scores = scores_tensor.numpy()
        labels = labels_tensor.numpy()
    
    # Scale boxes back to original image size
    if len(boxes) > 0:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * (orig_w / 300)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * (orig_h / 300)
    
    # Convert label indices to class names
    class_names = []
    for label in labels:
        if label > 0 and label <= len(classes):
            class_names.append(classes[int(label) - 1])
        else:
            class_names.append(f"Unknown({label})")
    
    return {
        'boxes': boxes,
        'labels': labels,
        'scores': scores,
        'class_names': class_names,
        'original_image': original_img,
        'image_path': image_path
    }

# ====================== VISUALIZE PREDICTIONS ======================
def visualize_predictions(predictions, save_path=None, show=True, font_scale=0.6):
    """
    Draw bounding boxes on image with labels and scores
    
    Args:
        predictions: Dictionary with prediction results
        save_path: Path to save visualization (optional)
        show: Whether to display the image
        font_scale: Font size for labels
    """
    img = predictions['original_image'].copy()
    boxes = predictions['boxes']
    class_names = predictions['class_names']
    scores = predictions['scores']
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(img)
    
    # Define colors for different classes
    np.random.seed(42)  # For consistent colors
    unique_classes = list(set(class_names))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_to_color = {cls: colors[i] for i, cls in enumerate(unique_classes)}
    
    # Draw each detection
    for box, class_name, score in zip(boxes, class_names, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        color = class_to_color[class_name]
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2.5, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label with score
        label_text = f"{class_name}: {score:.2f}"
        
        # Calculate text background size
        text_bbox = ax.text(
            x1, y1 - 5,
            label_text,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8, edgecolor='none'),
            fontsize=9,
            color='white',
            weight='bold',
            verticalalignment='bottom'
        )
    
    ax.axis('off')
    
    # Title with image name and detection count
    img_name = os.path.basename(predictions['image_path'])
    title = f"{img_name}\n{len(boxes)} objects detected"
    plt.title(title, fontsize=12, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        print(f"  💾 Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

# ====================== BATCH INFERENCE ======================
def predict_on_folder(model, folder_path, classes, score_threshold=0.5, 
                      nms_threshold=0.5, device='cpu', max_images=None):
    """
    Run inference on all images in a folder
    
    Args:
        model: Trained model
        folder_path: Path to folder with images
        classes: List of class names
        score_threshold: Confidence threshold
        nms_threshold: NMS IoU threshold
        device: Device to run on
        max_images: Maximum number of images to process (None for all)
    
    Returns:
        List of predictions for each image
    """
    # Get all image files
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n🔍 Running inference on {len(image_files)} images from {folder_path}...")
    
    all_predictions = []
    
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(folder_path, img_file)
        
        try:
            predictions = predict(model, img_path, classes, score_threshold, 
                                nms_threshold, device)
            
            # Save visualization
            save_path = os.path.join(OUTPUT_DIR, f"pred_{img_file}")
            visualize_predictions(predictions, save_path=save_path, show=False)
            
            all_predictions.append({
                'filename': img_file,
                'num_detections': len(predictions['boxes']),
                'predictions': predictions
            })
            
        except Exception as e:
            print(f"\n❌ Error processing {img_file}: {e}")
    
    return all_predictions

# ====================== PRINT DETECTION SUMMARY ======================
def print_detection_summary(predictions):
    """Print summary statistics of detections"""
    print(f"\n{'='*70}")
    print("DETECTION SUMMARY")
    print(f"{'='*70}")
    
    total_detections = sum(p['num_detections'] for p in predictions)
    avg_detections = total_detections / len(predictions) if predictions else 0
    
    print(f"Total images processed: {len(predictions)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {avg_detections:.2f}")
    
    # Count detections per class
    class_counts = {}
    confidence_scores = {}
    
    for pred in predictions:
        for class_name, score in zip(pred['predictions']['class_names'], 
                                     pred['predictions']['scores']):
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if class_name not in confidence_scores:
                confidence_scores[class_name] = []
            confidence_scores[class_name].append(score)
    
    print(f"\nDetections by class:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        avg_conf = np.mean(confidence_scores[class_name])
        print(f"  {class_name:20s}: {count:4d} detections (avg conf: {avg_conf:.3f})")
    
    print(f"{'='*70}\n")

# ====================== WEBCAM INFERENCE ======================
def run_webcam_inference(model, classes, score_threshold=0.5, nms_threshold=0.5, device='cpu'):
    """
    Run real-time inference on webcam feed
    
    Args:
        model: Trained model
        classes: List of class names
        score_threshold: Confidence threshold
        nms_threshold: NMS threshold
        device: Device to run on
    
    Controls:
        - 'q': Quit
        - 's': Save current frame
        - '+': Increase threshold
        - '-': Decrease threshold
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("\n📹 Starting webcam inference...")
    print("Controls:")
    print("  q - Quit")
    print("  s - Save current frame")
    print("  + - Increase threshold")
    print("  - - Decrease threshold")
    
    # Colors for each class
    np.random.seed(42)
    colors = {}
    for i, cls in enumerate(classes):
        colors[cls] = tuple((np.random.rand(3) * 255).astype(int).tolist())
    
    frame_count = 0
    current_threshold = score_threshold
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        
        # Preprocess
        img_tensor = F.to_tensor(frame_rgb)
        img_tensor = F.resize(img_tensor, [300, 300])
        img_tensor = F.normalize(img_tensor, 
                                mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            predictions = model(img_tensor)[0]
        
        # Filter and scale boxes
        scores = predictions['scores'].cpu().numpy()
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        high_score_idx = scores >= current_threshold
        scores = scores[high_score_idx]
        boxes = boxes[high_score_idx]
        labels = labels[high_score_idx]
        
        # Apply NMS
        if len(boxes) > 0:
            boxes_t = torch.tensor(boxes)
            scores_t = torch.tensor(scores)
            labels_t = torch.tensor(labels)
            boxes_t, scores_t, labels_t = apply_nms(boxes_t, scores_t, labels_t, nms_threshold)
            boxes = boxes_t.numpy()
            scores = scores_t.numpy()
            labels = labels_t.numpy()
        
        # Scale boxes
        if len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (w / 300)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (h / 300)
        
        # Draw on frame
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.astype(int)
            class_name = classes[int(label) - 1] if label > 0 and label <= len(classes) else "Unknown"
            
            if class_name not in colors:
                colors[class_name] = (0, 255, 0)
            color = colors[class_name]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{class_name}: {score:.2f}"
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display info
        info_text = f"Detections: {len(boxes)} | Threshold: {current_threshold:.2f} | FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('SSD Object Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = os.path.join(OUTPUT_DIR, f"webcam_frame_{frame_count}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"💾 Saved frame to {save_path}")
            frame_count += 1
        elif key == ord('+') or key == ord('='):
            current_threshold = min(0.95, current_threshold + 0.05)
            print(f"Threshold: {current_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            current_threshold = max(0.05, current_threshold - 0.05)
            print(f"Threshold: {current_threshold:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Webcam inference stopped")

# ====================== MAIN ======================
if __name__ == "__main__":
    # Load classes
    classes = load_classes(CLASSES_FILE)
    num_classes = len(classes) + 1  # +1 for background
    
    print("="*70)
    print("SSD OBJECT DETECTION - INFERENCE")
    print("="*70)
    print(f"Classes: {classes}")
    print(f"Device: {DEVICE}")
    print(f"Score threshold: {SCORE_THRESHOLD}")
    print(f"NMS threshold: {NMS_THRESHOLD}")
    print("="*70)
    
    # Load model
    model = load_model(MODEL_PATH, num_classes, DEVICE)
    
    # ============ OPTION 1: Single Image Inference ============
    print("\n" + "="*70)
    print("OPTION 1: Single Image Inference")
    print("="*70)
    
    # single_image_path = "../../../../Datasets/NEU-DET/images/test/crazing_132.jpg"
    single_image_path = "/mnt/d/Codes/DenseNet-Project/Datasets/NEU-DET/images/test/crazing_132.jpg"
    
    if os.path.exists(single_image_path):
        print(f"Running inference on: {single_image_path}")
        predictions = predict(model, single_image_path, classes, 
                            SCORE_THRESHOLD, NMS_THRESHOLD, DEVICE)
        
        print(f"\n🎯 Found {len(predictions['boxes'])} objects:")
        for i, (class_name, score) in enumerate(zip(predictions['class_names'], 
                                                     predictions['scores'])):
            print(f"  {i+1}. {class_name:20s} (confidence: {score:.3f})")
        
        # Visualize
        save_path = os.path.join(OUTPUT_DIR, "single_prediction.jpg")
        visualize_predictions(predictions, save_path=save_path, show=False)
        print(f"\n✅ Visualization saved to {save_path}")
    else:
        print(f"⚠️ Image not found: {single_image_path}")
    
    # ============ OPTION 2: Batch Inference on Folder ============
    print("\n" + "="*70)
    print("OPTION 2: Batch Inference on Folder")
    print("="*70)
    
    test_folder = "../../../../Datasets/NEU-DET/images/test"
    
    if os.path.exists(test_folder):
        all_predictions = predict_on_folder(
            model, 
            test_folder, 
            classes, 
            SCORE_THRESHOLD,
            NMS_THRESHOLD,
            DEVICE,
            max_images=20  # Process first 20 images (remove for all)
        )
        
        print_detection_summary(all_predictions)
    else:
        print(f"⚠️ Folder not found: {test_folder}")
    
    # ============ OPTION 3: Webcam Inference ============
    print("\n" + "="*70)
    print("OPTION 3: Webcam Inference")
    print("="*70)
    
    run_webcam = input("Run webcam inference? (y/n): ").lower().strip()
    if run_webcam == 'y':
        run_webcam_inference(model, classes, SCORE_THRESHOLD, NMS_THRESHOLD, DEVICE)
    
    print(f"\n✅ All predictions saved to: {OUTPUT_DIR}")
    print("="*70)