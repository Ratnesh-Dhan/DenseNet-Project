import glob, os, numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from matplotlib import pyplot as plt
from model import build_ssd_model
from dataset_loader import load_image_and_labels

# CONFIGS
IMG_SIZE = (200,200)
BATCH_SIZE = 8
NUM_CLASSES = 6
MAX_BOXES = 10
EPOCHS = 20
MODEL_NAME = "ssd_model"

TRAIN_ANN = "../../../../Datasets/NEU-DET/train/annotations"
VAL_ANN = "../../../../Datasets/NEU-DET/val/annotations"
TEST_ANN = "../../../../Datasets/NEU-DET/test/annotatioas"
TRAIN_IMG_DIR = "../../../../Datasets/NEU-DET/train/images"
VAL_IMG_DIR = "../../../../Datasets/NEU-DET/val/images"
TEST_IMG_DIR = "../../../../Datasets/NEU-DET/test/images"
RESULTS_DIR = f"../results/{MODEL_NAME}"
os.makedirs(RESULTS_DIR, exist_ok=True)

def tf_load(xml_file, img_dir):
    img, boxes, labels = tf.py_function(
        func=load_image_and_labels,
        inp=[xml_file, img_dir],
        Tout=[tf.float32, tf.float32, tf.int32]
    )
    img.set_shape([*IMG_SIZE, 3])
    boxes.set_shape([None, 4])
    labels.set_shape([None])
    return img, {"bboxes": boxes, "class_probs": labels}

def prepare_dataset(xml_glob, img_dir):
    xml_files = glob.glob(xml_glob)
    ds = tf.data.Dataset.from_tensor_slices(xml_files)
    ds = ds.map(lambda x: tf_load(x, img_dir), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(128).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

print("📦 Preparing datasets...")
train_ds = prepare_dataset(TRAIN_ANN, TRAIN_IMG_DIR)
val_ds = prepare_dataset(VAL_ANN, VAL_IMG_DIR)
test_ds = prepare_dataset(TEST_ANN, TEST_IMG_DIR)

# MODEL
print("⚙️ Building model...")
model = build_ssd_model(num_classes=NUM_CLASSES, max_boxes=MAX_BOXES)
model.summary()

# TRAINING
print("🚀 Training started...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

model.save(os.path.join("../../models", f"{MODEL_NAME}.h5"))
print("✅ Model saved successfully.")

# PLOTING TRAINING CURVES
plt.figure(figsize=(10, 5))
plt.plot(history.history["class_probs_accuracy"], label="Train Accuracy")
plt.plot(history.history["val_class_probs_accuracy"], label="Val Accuracy")
plt.title("Classification Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_graph.png"))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "loss_graph.png"))
plt.close()

# EVALUATION
def evaluate_dataset(model, dataset, split_name):
    print(f"\n🔍 Evaluating on {split_name} dataset...")
    all_preds, all_labels = [], []
    for images, labels in dataset:
        bboxes_pred, class_probs_pred = model.predict(images)
        preds = np.argmax(class_probs_pred, axis=-1)
        gts = labels["class_probs"].numpy()

        for p, gt in zip(preds, gts):
            # Take first box with highest probability
            if len(gt) == 0:
                continue
            gt_label = gt[0]
            pred_label = p[0]
            all_preds.append(pred_label)
            all_labels.append(gt_label)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"{split_name} Confusion Matrix")
    plt.savefig(os.path.join(RESULTS_DIR, f"{split_name.lower()}_confusion_matrix.png"))
    plt.close()

    report = classification_report(all_labels, all_preds, output_dict=True)
    avg_acc = report["accuracy"] * 100
    print(f"✅ {split_name} Accuracy: {avg_acc:.2f}%")
    return avg_acc, report

# EVALUATE ON TRAIN/VAL/TEST
train_acc, train_report = evaluate_dataset(model, train_ds, "Train")
val_acc, val_report = evaluate_dataset(model, val_ds, "Validation")
test_acc, test_report = evaluate_dataset(model, test_ds, "Test")

# WRITING RESULTS
with open(os.path.join(RESULTS_DIR, "accuracy_table.txt"), "w") as f:
    f.write("=== Model Evaluation Summary ===\n\n")
    f.write(f"Train Accuracy: {train_acc:.2f}%\n")
    f.write(f"Validation Accuracy: {val_acc:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
    f.write("=== Detailed Classification Report (Test) ===\n")
    for label, metrics in test_report.items():
        f.write(f"{label}: {metrics}\n")

print("\n📊 All results saved in '../results/' folder.")