"""
Run inference on images using trained detection model
"""
import tensorflow as tf
import numpy as np
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import os

class DetectionInference:
    def __init__(self, model_path, label_map_path, min_score_thresh=0.5):
        """
        Initialize detector.
        
        Args:
            model_path: Path to exported SavedModel directory
            label_map_path: Path to label_map.pbtxt file
            min_score_thresh: Minimum confidence threshold
        """
        self.min_score_thresh = min_score_thresh
        
        # Load model
        print("Loading model...")
        self.detect_fn = tf.saved_model.load(model_path)
        print("Model loaded!")
        
        # Load label map
        self.category_index = label_map_util.create_category_index_from_labelmap(
            label_map_path, use_display_name=True
        )
    
    def detect_objects(self, image_path):
        """
        Run detection on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            detections: Dictionary with detection results
            output_image: Image with bounding boxes drawn
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        input_tensor = tf.convert_to_tensor(image_rgb)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        # Run detection
        detections = self.detect_fn(input_tensor)
        
        # Convert to numpy
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                     for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        # Draw boxes on image
        image_with_detections = image_rgb.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=self.min_score_thresh,
            agnostic_mode=False
        )
        
        # Convert back to BGR for OpenCV
        output_image = cv2.cvtColor(image_with_detections, cv2.COLOR_RGB2BGR)
        
        return detections, output_image
    
    def detect_batch(self, image_dir, output_dir):
        """
        Run detection on all images in a directory.
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save output images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing {len(image_files)} images...")
        
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            
            try:
                detections, output_image = self.detect_objects(img_path)
                
                # Save output
                output_path = os.path.join(output_dir, f"detected_{img_file}")
                cv2.imwrite(output_path, output_image)
                
                # Print results
                print(f"\n{img_file}:")
                for i in range(detections['num_detections']):
                    score = detections['detection_scores'][i]
                    if score >= self.min_score_thresh:
                        class_id = detections['detection_classes'][i]
                        class_name = self.category_index[class_id]['name']
                        print(f"  - {class_name}: {score:.2f}")
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        print(f"\nResults saved to {output_dir}")


def main():
    """Example usage."""
    
    # Configuration
    MODEL_PATH = "exported_model/saved_model"
    LABEL_MAP_PATH = "/mnt/d/Codes/DenseNet-Project/Datasets/NEU-DET/tfrecords/label_map.pbtxt"
    TEST_IMAGE_DIR = "/mnt/d/Codes/DenseNet-Project/Datasets/NEU-DET/images/test"
    OUTPUT_DIR = "detection_results"
    MIN_SCORE_THRESH = 0.5
    
    # Initialize detector
    detector = DetectionInference(MODEL_PATH, LABEL_MAP_PATH, MIN_SCORE_THRESH)
    
    # Run detection on test images
    detector.detect_batch(TEST_IMAGE_DIR, OUTPUT_DIR)
    
    # Or detect single image
    # detections, output_img = detector.detect_objects("path/to/image.jpg")
    # cv2.imwrite("output.jpg", output_img)


if __name__ == "__main__":
    main()