import albumentations as A
import cv2, os, numpy as np
from PIL import Image
from os import path

class_map = ['Corrosion']

transform = A.Compose([
    A.SmallestMaxSize(max_size=400, p=1.0),  # Resize smaller dimension to 400px while maintaining aspect ratio
    A.Rotate(limit=30, p=0.5),               # Random rotation between -30 and +30 degrees with 50% probability
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2)
], bbox_params=A.BboxParams(
    format='yolo',
    min_visibility=0.3,          # discard bboxes that are mostly cropped
    label_fields=['class_labels'],  # Added label field
    check_each_transform=True    # Check validity after each transform
))

def transformer(transform, image, bboxes):
    # Extract class labels from bboxes
    class_labels = [int(bbox[4]) for bbox in bboxes]
    
    # Remove class labels from bboxes for Albumentations
    bbox_coords = [bbox[:4] for bbox in bboxes]
    
    try:
        transformed = transform(image=image, bboxes=bbox_coords, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_labels = transformed['class_labels']
        
        # Recombine coordinates with class labels
        result_bboxes = []
        for i, bbox in enumerate(transformed_bboxes):
            # Ensure coordinates are within valid range [0, 1]
            x, y, w, h = bbox
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.001, min(1.0, w))
            h = max(0.001, min(1.0, h))
            
            result_bboxes.append([x, y, w, h, transformed_labels[i]])
            
        return transformed_image, result_bboxes
    except Exception as e:
        print(f"Error during transformation: {e}")
        return image, bboxes  # Return original if transformation fails

def load_yolo_annotation(txt_path):
    bboxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:  # Ensure we have class + 4 coordinates
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                bboxes.append([x, y, w, h, class_id])
    return bboxes

def main():
    try:
        base_path = r"C:\Users\NDT Lab\Pictures\dataset\archive\corrosion detect\validation"
        original_img = path.join(base_path, "img")
        original_ann = path.join(base_path, "ann")
        modified_img = path.join(base_path, "new_augmented_img")
        modified_ann = path.join(base_path, "new_augmented_ann")
        
        # Create output directories 
        os.makedirs(modified_img, exist_ok=True)
        os.makedirs(modified_ann, exist_ok=True)

        txt_files = os.listdir(original_ann)
        if "classes.txt" in txt_files:
            txt_files.remove("classes.txt")
        else:
            print("No classes.txt file with labels")
            
        img_files = os.listdir(original_img)
        count = 0
        total = len(img_files)
        
        # Match txt files with image files
        for i in range(len(txt_files)):
            txt_name = txt_files[i].rsplit('.', 1)[0]
            # Find matching image file
            matching_img = None
            for img_file in img_files:
                if img_file.rsplit('.', 1)[0] == txt_name:
                    matching_img = img_file
                    break
            
            if matching_img:
                print(f"Processing: {txt_name}")
                bboxes = load_yolo_annotation(os.path.join(original_ann, txt_files[i]))
                
                try:
                    image = np.array(Image.open(os.path.join(original_img, matching_img)))
                    
                    # Skip empty images or annotations
                    if image.size == 0 or len(bboxes) == 0:
                        print(f"Skipping {txt_name} - empty image or no annotations")
                        continue
                        
                    new_image, new_txt = transformer(transform=transform, image=image, bboxes=bboxes)
                    
                    # Skip if no valid bboxes after augmentation
                    if len(new_txt) == 0:
                        print(f"Skipping {txt_name} - no valid bboxes after augmentation")
                        continue
                        
                    # After getting transformed image and annotation, save it
                    aug_filename = f"{txt_name}_aug_{count}"
                    new_img_path = os.path.join(modified_img, f"{aug_filename}.jpg")
                    new_txt_path = os.path.join(modified_ann, f"{aug_filename}.txt")
                    
                    # Save image
                    if isinstance(new_image, np.ndarray):
                        cv2.imwrite(new_img_path, cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))
                    
                    # Save annotation
                    with open(new_txt_path, 'w') as f:
                        for bbox in new_txt:
                            f.write(f"{int(bbox[4])} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                    
                    count += 1
                    print(f"Done {count} / {total}")
                    
                except Exception as e:
                    print(f"Error processing {txt_name}: {str(e)}")
                    continue
            else:
                print(f"No matching image found for {txt_name}")

    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()