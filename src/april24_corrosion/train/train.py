import os
import tensorflow as tf
from load_dataset import DataPipeline
from trasferLearningModelHardMode import  build_unet_with_transfer_learning

# --- Build Model ---
model = build_unet_with_transfer_learning(input_shape=(512, 512, 3), num_classes=3)


model.summary()

# --- Paths ---
TRAIN_IMAGE_DIR = r'D:\NML ML Works\corrosion Final dataset\train\images'
TRAIN_CORROSION_MASK_DIR = r'D:\NML ML Works\corrosion Final dataset\train\corrosion_mask'
TRAIN_PIECE_MASK_DIR = r'D:\NML ML Works\corrosion Final dataset\train\sample_piece_mask'

VAL_IMAGE_DIR = r'D:\NML ML Works\corrosion Final dataset\validate\images'  
VAL_CORROSION_MASK_DIR = r'D:\NML ML Works\corrosion Final dataset\validate\corrosion_mask'
VAL_PIECE_MASK_DIR = r'D:\NML ML Works\corrosion Final dataset\validate\sample_piece_mask'

MODEL_SAVE_PATH = os.path.join("./model",'unet_resnet50_multiclass.h5')

data_pipeline_train = DataPipeline(TRAIN_IMAGE_DIR, TRAIN_CORROSION_MASK_DIR, TRAIN_PIECE_MASK_DIR)
data_pipeline_validate = DataPipeline(VAL_IMAGE_DIR, VAL_CORROSION_MASK_DIR, VAL_PIECE_MASK_DIR)
# --- Get Datasets ---
train_dataset = data_pipeline_train.get_dataset(TRAIN_IMAGE_DIR, batch_size=8, augment_data=False, shuffle=True)
val_dataset = data_pipeline_validate.get_dataset(VAL_IMAGE_DIR, batch_size=8, augment_data=False, shuffle=False)

# --- Callbacks ---
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

# --- Train ---
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=60,
    callbacks=callbacks
)