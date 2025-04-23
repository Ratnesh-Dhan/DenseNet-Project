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
], bbox_params=A.BboxParams(format='yolo'))
# ],
# bbox_params=A.BboxParams(
#     format='yolo',
#     label_fields=['category_ids'],
#     min_visibility=0.3,          # discard bboxes that are mostly cropped
#     filter_lost_elements=True    # drop invalid bboxes after augmentation
# ))

"""
    transform = A.Compose([
    A.SmallestMaxSize(max_size=400, p=1.0),  # Resize smaller dimension to 400px while maintaining aspect ratio
    A.RandomCrop(width=int(400*0.8), height=int(400*0.8)),  # Crop 80% of the resized image
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2)],
    bbox_params=A.BboxParams(format='yolo'))
"""

def transformer(transform, image, bboxes):
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    return transformed_image, transformed_bboxes

def load_yolo_annotation(txt_path):
    classes_backup = []
    bboxes = []
    with open(txt_path, 'r') as f:
        bboxes = [(list(map(float ,line.strip().split()[1:])) + [int(line.strip().split()[0])]) for line in f]
        # for line in f:
        #     classes_backup.append(line.strip().split()[0])
        """
        # The below code wont work because if we iterate f for once then the file pointer will reach to the end 
        # and for 2nd line there wont be any lines left to read.
        //
        # classes_backup = [line.strip().split()[0] for line in f]
        # bboxes = [list(map(float, line.strip().split()[1:])) + [class_map[int(line.strip().split()[0])]] for line in f]
        """

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