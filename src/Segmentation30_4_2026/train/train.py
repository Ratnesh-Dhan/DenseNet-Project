import tensorflow as tf
from dataloader import load_image, get_array_with_path
from model import setup

image_path = "/mnt/z/DATASETS/kaggle_semantic_segmentation_CORROSION_dataset/train/images"
mask_path = "/mnt/z/DATASETS/kaggle_semantic_segmentation_CORROSION_dataset/train/masks"
images_paths = get_array_with_path(image_path)
masks_paths = get_array_with_path(mask_path)

val_image_path = "/mnt/z/DATASETS/kaggle_semantic_segmentation_CORROSION_dataset/validate/images"
val_mask_path = "/mnt/z/DATASETS/kaggle_semantic_segmentation_CORROSION_dataset/validate/masks"
val_images_paths = get_array_with_path(val_image_path)
val_masks_paths = get_array_with_path(val_mask_path)

dataset = tf.data.Dataset.from_tensor_slices((images_paths, masks_paths))
dataset = dataset.map(load_image)
dataset = dataset.batch(4).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_images_paths, val_masks_paths))
val_dataset = val_dataset.map(load_image)
val_dataset = val_dataset.batch(4).prefetch(tf.data.AUTOTUNE)

model = setup()

history = model.fit(
    dataset,
    epochs=60,
    validation_data=val_dataset,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
    ],
    verbose=1
)