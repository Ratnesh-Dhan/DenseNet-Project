import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import os 

# --- Load Model ---
model_path = "../models/EarlyStoppedBestSeptmber24.keras"
model = tf.keras.models.load_model(model_path)
validation_dir = r"/mnt/d/NML ML Works/newCoalByDeepBhaiya/16/VALIDATION"
train_dir = r"/mnt/d/NML ML Works/newCoalByDeepBhaiya/16/TRAINING 16"
model_name = os.path.basename(model_path)

# --- Recreate validation dataset ---
batch_size=64
img_size=(16,16)

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
# plt.show()
plt.savefig(f"../results/{model_name}_confusion_matrix_new2.png", dpi=300)
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
plt.ylabel("True lavel")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix (in % per class)")
plt.savefig(f"../results/{model_name}_confusion_matrix_%_wise.png", dpi=300)
plt.close()

# --- Save Classification Report ---
report = classification_report(
    y_true, 
    y_pred, 
    target_names=class_names, 
    digits=4
)

with open(f"../results/{model_name}_classification_report.txt", "w") as f:
    f.write(report)

print("✅ Confusion matrix saved as 'confusion_matrix.png'")
print("✅ Classification report saved as 'classification_report.txt'")
