import albumentations as A
import cv2, os, numpy as np
from PIL import Image

class_map = ['corrosion', 'Corrosion']

transform = A.Compose([
    A.Resize(height=512, width=512, p=1.0),  # Resize to exactly 512x512
    A.Rotate(limit=30, p=0.5),               # Random rotation between -30 and +30 degrees with 50% probability
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def clip_coordinates(bbox):
    # Clip coordinates to [0, 1] range
    return [max(0, min(1, coord)) for coord in bbox]

def yolo_to_albumentations(bbox):
    # Convert from YOLO format (x_center, y_center, width, height) to albumentations format (x_min, y_min, x_max, y_max)
    x_center, y_center, width, height = bbox[:4]
    x_min = x_center - width/2
    y_min = y_center - height/2
    x_max = x_center + width/2
    y_max = y_center + height/2
    return clip_coordinates([x_min, y_min, x_max, y_max])

def albumentations_to_yolo(bbox):
    # Convert from albumentations format (x_min, y_min, x_max, y_max) to YOLO format (x_center, y_center, width, height)
    x_min, y_min, x_max, y_max = bbox[:4]
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width/2
    y_center = y_min + height/2
    return clip_coordinates([x_center, y_center, width, height])

def transformer(transform, image, bboxes):
    # Convert YOLO format to albumentations format and clip coordinates
    class_labels = [bbox[4] for bbox in bboxes]
    bboxes = [yolo_to_albumentations(bbox[:4]) for bbox in bboxes]
    
    # Ensure all coordinates are within [0, 1] range before transformation
    bboxes = [clip_coordinates(bbox) for bbox in bboxes]
    
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_labels = transformed['class_labels']
    
    # Convert back to YOLO format and clip coordinates
    transformed_bboxes = [albumentations_to_yolo(bbox) for bbox in transformed_bboxes]
    transformed_bboxes = [[max(0, min(1, coord)) for coord in bbox[:4]] + [label] 
                         for bbox, label in zip(transformed_bboxes, transformed_labels)]
    
    return transformed_image, transformed_bboxes

def load_yolo_annotation(txt_path):
    bboxes = []
    with open(txt_path, 'r') as f:
        bboxes = [(list(map(float ,line.strip().split()[1:])) + [int(line.strip().split()[0])]) for line in f]
    return bboxes

def main():
    try:
        base_path = r"C:\Users\NDT Lab\Pictures\dataset\archive\corrosion detect"
        original_img = os.path.join(base_path, "validation", "img")
        original_ann = os.path.join(base_path, "validation", "ann")
        modified_img = os.path.join(base_path, "validation", "new_augmented_img")
        modified_ann = os.path.join(base_path, "validation", "new_augmented_ann")
        # Create output directories 
        os.makedirs(modified_img, exist_ok=True)
        os.makedirs(modified_ann, exist_ok=True)

        txt_files = os.listdir(original_ann)
        if "classes.txt" in txt_files:
            txt_files.remove("classes.txt")
        else:
            print("No classes.txt inside ann folder")
        img_files = os.listdir(original_img)
        count = 0
        total = len(img_files)
        for i in range(len(txt_files)):
            if txt_files[i].rsplit('.', 1)[0] == img_files[i].rsplit('.', 1)[0]:
                print(f"{txt_files[i].rsplit('.', 1)[0]} : {img_files[i].rsplit('.', 1)[0]}")
                bbboxes = load_yolo_annotation(os.path.join(original_ann, txt_files[i]))
                # image = cv2.imread(os.path.join("../img", img_files[i]))
                image = np.array(Image.open(os.path.join(original_img, img_files[i])))
                new_image, new_txt = transformer(transform=transform, image=image, bboxes=bbboxes)
                # After getting transformed image and annotation. Lets save it
                aug_filename = f"{img_files[i].rsplit('.', 1)[0]}_aug_{count}"
                new_img_path = os.path.join(modified_img, f"{aug_filename}.jpg")
                new_txt_path  = os.path.join(modified_ann, f"{aug_filename}.txt")
                cv2.imwrite(new_img_path, new_image)
                with open(new_txt_path, 'w') as f:
                    for bbox in range(len(new_txt)):
                        f.write(f"{int(new_txt[bbox][4])} {new_txt[bbox][0]} {new_txt[bbox][1]} {new_txt[bbox][2]} {new_txt[bbox][3]}\n")

                count += 1
                print(f"Done {count} / {total}")

    except RuntimeError as e: 
        print(f"Inner array {e}")
if __name__ == "__main__":
    main()
