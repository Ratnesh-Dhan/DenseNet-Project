from random import shuffle
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import LVGG16_16x16
from model32 import battis_x_battis
import json
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os
import absl.logging
import tensorflow as tf

absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR

# train_dir = r"/mnt/d/NML ML Works/newCoalByDeepBhaiya/16/TRAINING 16"
# validation_dir = r"/mnt/d/NML ML Works/newCoalByDeepBhaiya/16/VALIDATION"
train_dir = r"/mnt/d/NML ML Works/newCoalByDeepBhaiya/31/TRAINING 31"
validation_dir = r"/mnt/d/NML ML Works/newCoalByDeepBhaiya/31/VALIDATION"
model_names = ['adadelta', 'adagrad']

# batch_size = 64 # for sparse categorical
batch_size=64
# img_size=(16,16)
img_size=(30, 30)

for model_name in model_names:
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        validation_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    normalization_layer = keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x,y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x,y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    early_stop=EarlyStopping(
        monitor='val_loss',
        patience = 5,
        restore_best_weights=True,
        # verbose=1
    )

    model_path_with_name = os.path.join("../models","32", model_name)
    os.makedirs(model_path_with_name, exist_ok=True)
    model_checkpoint=ModelCheckpoint(
        filepath=os.path.join(model_path_with_name,f"EarlyStoppedBest{model_name}.keras"),
        monitor = 'val_loss',
        # verbose = 1,
        save_best_only=True,
        mode='auto'
    )

    # model = LVGG16_16x16(num_classes=5, optimizer=model_name)
    model = battis_x_battis(num_classes=5, optimizer=model_name)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=[early_stop, model_checkpoint]
    )

    model.save(os.path.join(model_path_with_name, f'{model_name}.keras'))

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.xlim(0)       # Start x-axis at 0
    plt.ylim(0)       # Start y-axis at 0

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.xlim(0)       # Start x-axis at 0
    plt.ylim(0)       # Start y-axis at 0

    plt.tight_layout()
    results_folder = os.path.join("../results", "batch_train_for_32", model_name)
    os.makedirs(results_folder, exist_ok=True)
    plt.savefig(os.path.join(results_folder, f"{model_name}.png"), bbox_inches="tight")

    # class indices
    class_names = ["0 Cavity", "1 Cavity filled", "2 Inertinite", "3 Minerals", "4 Vitrinite"]
    # 1. Get true labels and predictions for validation set
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images)
        predicted_labels = np.argmax(preds, axis=1)
        y_pred.extend(predicted_labels)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 2. Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # 3. Plot it
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_folder, f"{model_name}_confusion_matrix.png"), dpi=300)
    plt.close()

    # % confusion matrix
    cm_normalized = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]

    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix (in % per class)")
    plt.savefig(os.path.join(results_folder ,f"{model_name}_confusion_matrix_%_wise.png"), dpi=300)
    plt.close()

    # --- Save Classification Report ---
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names, 
        digits=4
    )

    with open(os.path.join(results_folder, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(report)
    
    with open(os.path.join(results_folder, f"{model_name}_history.json"), "w") as f:
        json.dump(history.history, f)


    print("✅ Confusion matrix saved as 'confusion_matrix.png'")
    print("✅ Classification report saved as 'classification_report.txt'")