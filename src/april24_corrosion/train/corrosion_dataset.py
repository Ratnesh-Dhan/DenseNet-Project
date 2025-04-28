import os
import numpy as np
import skimage.io
import skimage.color
import skimage.transform
from mrcnn.config import Config
from mrcnn import model as modellib, utils

class CorrosionDataset(utils.Dataset):
    def load_corrosion(self, dataset_dir, subset):
        # Add classes
        self.add_class("corrosion_dataset", 1, "sample_piece")
        self.add_class("corrosion_dataset", 2, "corrosion")
        
        # List all images
        image_dir = os.path.join(dataset_dir, "images")
        images = next(os.walk(image_dir))[2]

        for img_name in images:
            image_path = os.path.join(image_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            sample_mask_path = os.path.join(dataset_dir, "masks_sample_piece", base_name + "_mask_sample.png")
            corrosion_mask_path = os.path.join(dataset_dir, "masks_corrosion", base_name + "_mask_corrosion.png")

            if not os.path.exists(sample_mask_path) or not os.path.exists(corrosion_mask_path):
                continue

            self.add_image(
                "corrosion_dataset",
                image_id=img_name,
                path=image_path,
                sample_mask=sample_mask_path,
                corrosion_mask=corrosion_mask_path
            )

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        sample_mask = skimage.io.imread(info['sample_mask'])
        corrosion_mask = skimage.io.imread(info['corrosion_mask'])

        # Convert to binary
        if len(sample_mask.shape) == 3:
            sample_mask = sample_mask[:, :, 0]
        sample_mask = (sample_mask > 0).astype(np.bool_)

        if len(corrosion_mask.shape) == 3:
            corrosion_mask = corrosion_mask[:, :, 0]
        corrosion_mask = (corrosion_mask > 0).astype(np.bool_)

        masks = np.stack([sample_mask, corrosion_mask], axis=-1)
        class_ids = np.array([1, 2])

        return masks, class_ids.astype(np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]
