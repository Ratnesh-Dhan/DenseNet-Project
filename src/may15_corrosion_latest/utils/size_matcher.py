import cv2
import os, sys

piece_size_dir = r'D:\NML ML Works\corrosion all masks\dataset 2025-04-25 16-40-02\merged_masks_2nd_png_version'
image_size_dir = r'D:\NML ML Works\corrosion all masks\dataset 2025-04-25 16-40-02\img_2nd_png_version' 
corrosion_size_dir = r'D:\NML ML Works\corrosion all masks\dataset 2025-04-25 16-40-02\filtered_corrosion_2nd_png_version'

pieces = os.listdir(piece_size_dir)
for name in pieces:
    try:
        piece_image = cv2.imread(os.path.join(piece_size_dir, name))
        img_image = cv2.imread(os.path.join(corrosion_size_dir, name))
        if img_image is not None:
            if piece_image.shape != img_image.shape:
                print(piece_image.shape, " : ", img_image.shape)
                img_image = cv2.resize(img_image, (piece_image.shape[1], piece_image.shape[0]))
                cv2.imwrite(os.path.join(corrosion_size_dir, name), img_image)
                print(f"[RESIZED] {name} to match piece mask")
    except:
        print(name)
sys.exit(0)
if corrosion_mask.shape != piece_mask.shape: # If shape mismatch
    # corrosion_mask = cv2.resize(corrosion_mask, (piece_mask.shape[1], piece_mask.shape[0]),
    #                             interpolation=cv2.INTER_NEAREST)
    corrosion_mask = cv2.resize(corrosion_mask, (piece_mask.shape[1], piece_mask.shape[0]))
    print(f"[RESIZED] {fname} to match piece mask")