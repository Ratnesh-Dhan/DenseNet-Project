from tf_dataset_loader import CLASS_MAP, load_dataset, preprocess, CLASS_NAMES
from model import get_model
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Load datasets ===
train_records = load_dataset("/mnt/d/Code/DenseNet-Project/Datasets/NEU-DET/train/annotations", "/mnt/d/Code/DenseNet-Project/Datasets/NEU-DET/train/images")
val_records = load_dataset("/mnt/d/Code/DenseNet-Project/Datasets/NEU-DET/val/annotations", "/mnt/d/Code/DenseNet-Project/Datasets/NEU-DET/val/images")
test_records = load_dataset("/mnt/d/Code/DenseNet-Project/Datasets/NEU-DET/test/annotations", "/mnt/d/Code/DenseNet-Project/Datasets/NEU-DET/test/images")

# === TF Dataset ===
# def make_dataset(records, shuffle=False):
#     ds = tf.data.Dataset.from_generator(
#         lambda: iter(records),
#         output_types=(tf.string, tf.float32, tf.int32)
#     )
#     ds = ds.map(lambda x, y, z: preprocess(x, y, z))
#     if shuffle:
#         ds = ds.shuffle(100)
#     ds = ds.batch(4)
#     return ds
def make_dataset(records, shuffle=False):
    ds = tf.data.Dataset.from_generator(
        lambda: iter(records),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    )
    ds = ds.map(lambda x, y, z: preprocess(x, y, z))
    if shuffle:
        ds = ds.shuffle(100)
    ds = ds.batch(4).prefetch(tf.data.AUTOTUNE)
    return ds
train_ds = make_dataset(train_records, shuffle=True)
val_ds = make_dataset(val_records)
test_ds = make_dataset(test_records)

# === Model ===
model = get_model(CLASS_NAMES, CLASS_MAP)

# === Train ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)
model_name = "efficientdet_neu_det"
save_folder = model_name 
os.makedirs(model_name, exist_ok=True)
model.save(os.path.join(save_folder, f"{model_name}.keras"))

# === Plot training curves ===
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="total_loss")
plt.plot(history.history.get("val_loss", []), label="val_total_loss")
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(save_folder, "training_curves.png"))
# plt.figure(figsize=(8, 6))
# plt.plot(history.history["loss"], label="train_loss")
# plt.plot(history.history["val_loss"], label="val_loss")
# plt.title("Training and Validation Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.savefig(os.path.join(save_folder, "training_curves.png"))

# === Evaluate on test dataset ===
test_loss = model.evaluate(test_ds)
print(f"\nTest Loss: {test_loss}")

# === Gather predictions ===
y_true = []
y_pred = []

for batch in test_ds:
    images = batch["images"]
    true_labels = batch["bounding_boxes"]["classes"].numpy()

    preds = model.predict(images, verbose=0)

    # preds is a dict: { 'boxes': (B, N, 4), 'classes': (B, N), 'confidence': (B, N) }
    pred_labels = preds["classes"]

    # Take the top-1 predicted class per image (most confident detection)
    pred_top = [int(p[0]) if len(p) > 0 else -1 for p in pred_labels]

    # Flatten (assuming one ground truth class per image)
    true_top = [lbl[0] if len(lbl) > 0 else -1 for lbl in true_labels]

    y_true.extend(true_top)
    y_pred.extend(pred_top)

# y_true = []
# y_pred = []

# for batch in test_ds:
#     images = batch["images"]
#     true_labels = batch["bounding_boxes"]["classes"].numpy()

#     preds = model.predict(images, verbose=0)
#     # preds = (boxes, class_probs)
#     class_probs = preds[1]
#     pred_labels = np.argmax(class_probs, axis=-1)

#     # Flatten (one label per image assumption)
#     y_true.extend([lbl[0] if len(lbl) > 0 else -1 for lbl in true_labels])
#     y_pred.extend(pred_labels)

# # Remove -1s (images without annotations)
# mask = np.array(y_true) != -1
# y_true = np.array(y_true)[mask]
# y_pred = np.array(y_pred)[mask]

# === Classification Report ===
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
print("\nClassification Report:")
for cls in CLASS_NAMES:
    print(f"{cls:20s}: {report[cls]['precision']*100:.2f}% / {report[cls]['recall']*100:.2f}% / {report[cls]['f1-score']*100:.2f}%")

with open(os.path.join(save_folder, "Classification_report.txt"), "w") as f:
    f.write("=== Classification Report ===\n\n")
    for cls in CLASS_NAMES:
        precision = report[cls]["precision"] * 100
        recall = report[cls]["recall"] * 100
        f1 = report[cls]["f1-score"] * 100
        f.write(f"{cls:20s}: Precision={precision:.2f}%  Recall={recall:.2f}%  F1={f1:.2f}%\n")
    
    # Write overall averages
    f.write("\nOverall Accuracy: {:.2f}%\n".format(report["accuracy"] * 100))
    f.write("Macro Avg: Precision={:.2f}%  Recall={:.2f}%  F1={:.2f}%\n".format(
        report["macro avg"]["precision"] * 100,
        report["macro avg"]["recall"] * 100,
        report["macro avg"]["f1-score"] * 100
    ))
    f.write("Weighted Avg: Precision={:.2f}%  Recall={:.2f}%  F1={:.2f}%\n".format(
        report["weighted avg"]["precision"] * 100,
        report["weighted avg"]["recall"] * 100,
        report["weighted avg"]["f1-score"] * 100
    ))

print(f"\nClassification report saved to: {save_folder}")

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred, normalize="true") * 100
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Test Confusion Matrix (%)")
plt.savefig(os.path.join(save_folder, "Test confusion_matrix_%.png"))