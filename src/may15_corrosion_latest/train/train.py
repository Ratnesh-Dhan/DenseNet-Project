import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"   # FOR LARGE MODELS (reducing fragmentation of memory)
import tensorflow as tf
print("x" for i in range(20))
print(tf.config.list_physical_devices('GPU'))
print("x" for i in range(20))
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")   # FOR FAST TRAINING (40% to 50% memory reduction)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# from model import unet_model
# from new_transfer_model import build_unet_with_resnet50
from corrosion_model_21_04_2026 import build_unet_with_resnet50, iou_metric, improved_loss
import matplotlib.pyplot as plt
from data_loader import get_dataset
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# model = unet_model()
model = build_unet_with_resnet50()

# model.summary()

model_path = "./model"
os.makedirs(model_path, exist_ok=True)
plot_save_path = "./plots"
os.makedirs(plot_save_path, exist_ok=True)

base_dir = r"/mnt/z/DATASETS/kaggle_semantic_segmentation_CORROSION_dataset"
# base_dir = r"D:\NML ML Works\kaggle_semantic_segmentation_CORROSION_dataset"
train_ds = get_dataset(os.path.join(base_dir, "train/images"), os.path.join(base_dir, "train/masks"), batch_size=4)
val_ds = get_dataset(os.path.join(base_dir, "validate/images"), os.path.join(base_dir, "validate/masks"), batch_size=4)

train_ds_finetuning = get_dataset(os.path.join(base_dir, "train/images"), os.path.join(base_dir, "train/masks"), batch_size=2)
val_ds_finetuning = get_dataset(os.path.join(base_dir, "validate/images"), os.path.join(base_dir, "validate/masks"), batch_size=2)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(os.path.join(model_path, 'best_model_transferLearning_test.keras'), monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1)
]

# CALLBACKS FOR FINE TUNING
callbacks_ft = [
    EarlyStopping(monitor='val_recall',mode='max', patience=5, restore_best_weights=True),
    ModelCheckpoint(os.path.join(model_path, 'best_model_ft.keras'),
                    monitor='val_recall', mode='max', save_best_only=True),
    # EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    # ModelCheckpoint(os.path.join(model_path, 'best_model_ft.keras'),
    #                 monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1)
]
#  callbacks=[tf.keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True)], 

history1 = model.fit(train_ds, 
                    validation_data=val_ds, 
                    epochs=60,
                    callbacks=callbacks
                    )

#FINE TUNING
tf.keras.backend.clear_session() # Clear the session to free up memory

model = build_unet_with_resnet50()
model.load_weights(os.path.join(model_path, 'best_model_transferLearning_test.keras'))

for layer in model.layers:
    # if isinstance(layer, tf.keras.Model):
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    # if "conv5" in layer.name or "conv4" in layer.name:
    elif "conv5" in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=improved_loss,
    metrics=[iou_metric, tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
)
history2 = model.fit(
    train_ds_finetuning,
    validation_data=val_ds_finetuning,
    epochs=15,  # fine-tuning phase
    callbacks=callbacks_ft
)
model.save(os.path.join(model_path, "best_model_transferLearning_test.keras"))
print('\nModel saved successfully..!\n')

# PLOTTING 
def plot_history(history, name):
    plt.figure(figsize=(12, 5))

    # Recall
    plt.figure(figsize=(6,5))
    if 'recall' in history.history:
        plt.plot(history.history['recall'], label='Train Recall')
        if 'val_recall' in history.history:
            plt.plot(history.history['val_recall'], label='Val Recall')
        plt.title('Recall over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(os.path.join(plot_save_path, f'{name}_recall.png'))
        plt.close()

    # Loss
    plt.figure(figsize=(6,5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('loss', []), label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # IoU / Metric
    plt.subplot(1, 2, 2)
    if 'iou_metric' in history.history:
        plt.plot(history.history['iou_metric'], label='Train IoU')
        if 'val_iou_metric' in history.history:
            plt.plot(history.history['val_iou_metric'], label='Val IoU')
        plt.title('IoU over epochs')
        plt.ylabel('IoU')
    elif 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Train Acc')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuray'], label="Val Acc")
        plt.title('Accuracy over epochs')
        plt.ylabel('Accuracy')
    else:
        plt.text(0.5,0.5, 'No metric found', ha='center')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_path, f'{name}_training_history.png'))
    plt.close()

plot_history(history1, "before_fine_tuning")
plot_history(history2, "fine_tuned")



# def plot_history(history, name):
#     plt.figure(figsize=(12, 5))

#     # Loss
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Val Loss')
#     plt.title('Loss over epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()

#     # Accuracy (optional if using binary accuracy metric)
#     if 'iou_metric' in history.history:
#         plt.subplot(1, 2, 2)
#         plt.plot(history.history['accuracy'], label='Train Acc')
#         plt.plot(history.history['val_accuracy'], label='Val Acc')
#         plt.title('Accuracy over epochs')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.legend()

#     plt.tight_layout()
#     plt.savefig(f'{name}_training_history.png')
#     plt.show()