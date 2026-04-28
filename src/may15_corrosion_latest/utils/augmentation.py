import os
import cv2
import albumentations as A
from albumentations.core.serialization import save, load
from tqdm import tqdm

# Define paths
base_dir = r"/home/zumbie/Codes/NML/DenseNet-Project/Datasets/kaggle_semantic_segmentation_CORROSION_dataset/train"
image_dir = os.path.join(base_dir, 'images')
corrosion_mask_dir = os.path.join(base_dir, 'masks')
# sample_mask_dir = os.path.join(base_dir, 'merged_masks_2nd_png_version')
# Output dirs
aug_img_dir = os.path.join(base_dir, 'augmented/images')
aug_corrosion_mask_dir = os.path.join(base_dir, 'augmented/masks')
# aug_sample_mask_dir = os.path.join(base_dir, 'augmented/sample_piece_mask')

os.makedirs(aug_img_dir, exist_ok=True)
os.makedirs(aug_corrosion_mask_dir, exist_ok=True)
# os.makedirs(aug_sample_mask_dir, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.Affine(translate_percent=0.05, scale=1.1, rotate=20, p=0.7),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    # these 2 are additional augmentations
    # A.Perspective(scale=(0.05, 0.1), p=0.3),
    # A.CoarseDropout(max_holes=5, max_height=32, max_width=32, mask_fill_value=0, p=0.4),
], additional_targets={
    'corrosion_mask': 'mask',
    'sample_mask': 'mask'
})

# Process images
for filename in tqdm(os.listdir(image_dir)):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    img_path = os.path.join(image_dir, filename)
    cor_mask_path = os.path.join(corrosion_mask_dir, filename.replace('.jpg', '.png'))
    # samp_mask_path = os.path.join(sample_mask_dir, filename)
    
    image = cv2.imread(img_path)
    corrosion_mask = cv2.imread(cor_mask_path)
    # corrosion_mask = cv2.imread(cor_mask_path, cv2.IMREAD_GRAYSCALE)
    # sample_mask = cv2.imread(samp_mask_path, cv2.IMREAD_GRAYSCALE)
    
    for i in range(5):  # 5 augmentations per image
        # augmented = transform(image=image, corrosion_mask=corrosion_mask, sample_mask=sample_mask)
        augmented = transform(image=image, corrosion_mask=corrosion_mask)
        
        aug_image = augmented['image']
        aug_corrosion_mask = augmented['corrosion_mask']
        # aug_sample_mask = augmented['sample_mask']
        
        aug_image_name = f"{os.path.splitext(filename)[0]}_aug{i}.jpg"
        aug_filename = f"{os.path.splitext(filename)[0]}_aug{i}.png"
        cv2.imwrite(os.path.join(aug_img_dir, aug_image_name), aug_image)
        cv2.imwrite(os.path.join(aug_corrosion_mask_dir, aug_filename), aug_corrosion_mask)
        # cv2.imwrite(os.path.join(aug_sample_mask_dir, aug_filename), aug_sample_mask)
