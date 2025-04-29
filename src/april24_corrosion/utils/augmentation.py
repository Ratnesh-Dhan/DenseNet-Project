import os
import cv2
import albumentations as A
from albumentations.core.serialization import save, load
from tqdm import tqdm

# Define paths
base_dir = r"D:\NML ML Works\cropped corrosion annotaion"
image_dir = os.path.join(base_dir, 'images')
corrosion_mask_dir = os.path.join(base_dir, 'corrosion_mask')
sample_mask_dir = os.path.join(base_dir, 'sample_piece_mask')
# Output dirs
aug_img_dir = os.path.join(base_dir, 'augmented/images')
aug_corrosion_mask_dir = os.path.join(base_dir, 'augmented/corrosion_mask')
aug_sample_mask_dir = os.path.join(base_dir, 'augmented/sample_piece_mask')

os.makedirs(aug_img_dir, exist_ok=True)
os.makedirs(aug_corrosion_mask_dir, exist_ok=True)
os.makedirs(aug_sample_mask_dir, exist_ok=True)

# Define augmentation pipeline
# transform = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
#     A.Rotate(limit=30, p=0.5),
#     A.GaussNoise(p=0.2),
#     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
# ], additional_targets={
#     'corrosion_mask': 'mask',
#     'sample_mask': 'mask'
# })
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.Affine(translate_percent=0.05, scale=1.1, rotate=20, p=0.7),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
], additional_targets={
    'corrosion_mask': 'mask',
    'sample_mask': 'mask'
})


# Process images
for filename in tqdm(os.listdir(image_dir)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    img_path = os.path.join(image_dir, filename)
    cor_mask_path = os.path.join(corrosion_mask_dir, f'corrosion_mask_{filename}')
    samp_mask_path = os.path.join(sample_mask_dir, f'piece_mask_{filename}')
    
    image = cv2.imread(img_path)
    corrosion_mask = cv2.imread(cor_mask_path, cv2.IMREAD_GRAYSCALE)
    sample_mask = cv2.imread(samp_mask_path, cv2.IMREAD_GRAYSCALE)
    
    for i in range(5):  # 5 augmentations per image
        augmented = transform(image=image, corrosion_mask=corrosion_mask, sample_mask=sample_mask)
        
        aug_image = augmented['image']
        aug_corrosion_mask = augmented['corrosion_mask']
        aug_sample_mask = augmented['sample_mask']
        
        aug_filename = f"{os.path.splitext(filename)[0]}_aug{i}.png"
        cv2.imwrite(os.path.join(aug_img_dir, aug_filename), aug_image)
        cv2.imwrite(os.path.join(aug_corrosion_mask_dir, aug_filename), aug_corrosion_mask)
        cv2.imwrite(os.path.join(aug_sample_mask_dir, aug_filename), aug_sample_mask)
