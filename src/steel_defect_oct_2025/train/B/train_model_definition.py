# custom_detector.py
from unicodedata import digit
import tensorflow as tf
from dataset_loader import load_image_and_label, CLASS_MAP
from glob import glob
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os

def build_model(num_classes):
    # MobileNetV2 backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(200,200,3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

    bbox_regression = tf.keras.layers.Dense(4, name="bbox")(x)  # normalized [xmin,ymin,xmax,ymax]
    class_prediction = tf.keras.layers.Dense(num_classes, activation="softmax", name="class")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=[bbox_regression, class_prediction])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={
            "bbox": "mse",  # replace with smooth L1 later
            "class": "sparse_categorical_crossentropy"
        },
        metrics={"class": "accuracy"}
    )

    model.summary()
    return model

# === PLOTS ===
def plot_training_curves(history, model_name):
    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history["class_accuracy"], label="train_acc")
    plt.plot(history.history["val_class_accuracy"], label="val_acc")
    plt.title("Classification Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history["bbox_loss"], label="bbox_loss")
    plt.plot(history.history["val_bbox_loss"], label="val_bbox_loss")
    plt.plot(history.history["class_loss"], label="class_loss")
    plt.plot(history.history["val_class_loss"], label="val_class_loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"../results/{model_name}_training_curves.png")

# === CONFUSION MATRIX ===
def plot_confusion_matrix_and_classification_report(model, val_ds, CLASS_NAMES, NUM_CLASSES, model_name):
    y_true = []
    y_pred = []

    for batch in val_ds:
        images, labels = batch
        true_classes = labels["class"].numpy()
        preds = model.predict(images, verbose=0)[1]
        pred_classes = np.argmax(preds, axis=1)
        y_true.extend(true_classes)
        y_pred.extend(pred_classes)

    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm_percentage, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.title("Confusion Matrix (%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    os.makedirs("../results", exist_ok=True)
    plt.savefig(f"../results/{model_name}_confusion_matrix.png")

    with open(f"../results/{model_name}_classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, labels=range(NUM_CLASSES), target_names=CLASS_NAMES, digits=4, zero_division=0))
    print(f"Classification report saved to ../results/{model_name}_classification_report.txt")

if __name__ == "__main__":
    num_classes = len(CLASS_MAP)
    CLASS_NAMES = list(CLASS_MAP.keys())
    NUM_CLASSES = len(CLASS_MAP)
    model_name = "custom_mobilenet_detector"
    epochs = 20

    # === Dataset ===
    xml_files = glob("/mnt/d/Code/DenseNet-Project/Datasets/NEU-DET/train/annotations/*.xml")
    img_dir = "/mnt/d/Code/DenseNet-Project/Datasets/NEU-DET/train/images"
    assert len(xml_files) > 0, "No train XML files found — check path!"
    dataset = tf.data.Dataset.from_tensor_slices(xml_files)
    dataset = dataset.map(lambda x: load_image_and_label(x, img_dir))
    train_dataset = dataset.batch(16).prefetch(tf.data.AUTOTUNE)

    val_xml_files = glob("/mnt/d/Code/DenseNet-Project/Datasets/NEU-DET/val/annotations/*.xml")
    val_img_dir = "/mnt/d/Code/DenseNet-Project/Datasets/NEU-DET/val/images"
    assert len(val_xml_files) > 0, "No val XML files found — check path!"
    val_dataset = tf.data.Dataset.from_tensor_slices(val_xml_files)
    val_dataset = val_dataset.map(lambda x: load_image_and_label(x, val_img_dir))
    val_dataset = val_dataset.batch(16).prefetch(tf.data.AUTOTUNE)

    # === TEST DATASET ===
    test_xml_files = glob("/mnt/d/Code/DenseNet-Project/Datasets/NEU-DET/test/annotations/*.xml")
    test_img_dir = "/mnt/d/Code/DenseNet-Project/Datasets/NEU-DET/test/images"
    assert len(test_xml_files) > 0, "No test XML files found — check path!"
    test_dataset = tf.data.Dataset.from_tensor_slices(test_xml_files)
    test_dataset = test_dataset.map(lambda x: load_image_and_label(x, test_img_dir))
    test_dataset = test_dataset.batch(16).prefetch(tf.data.AUTOTUNE)


    # # Train / validation split
    # train_size = int(0.8 * len(xml_files))
    # train_dataset = dataset.take(train_size)
    # val_dataset = dataset.skip(train_size)
    model = build_model(num_classes)
    # === Training ===
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
    model.save(f"{model_name}.keras")
    plot_confusion_matrix_and_classification_report(model, val_dataset, CLASS_NAMES, NUM_CLASSES, model_name)
    print(f"Confusion matrix saved to ../results/{model_name}_confusion_matrix.png")
    plot_training_curves(history, model_name)
    print(f"Training curves saved to ../results/{model_name}_training_curves.png")
    # === Classification Report ===
    y_true = []
    y_pred = []
    for batch in val_dataset:
        images, labels = batch
        true_classes = labels["class"].numpy()
        preds = model.predict(images, verbose=0)[1]
        pred_classes = np.argmax(preds, axis=1) 
    
    # === Evaluate on Test Set ===
    print("\nEvaluating on test dataset...")
    test_results = model.evaluate(test_dataset)
    print("Test Results:", test_results)

    # Optionally — Confusion Matrix & Classification Report for Test Set
    plot_confusion_matrix_and_classification_report(
        model,
        test_dataset,
        CLASS_NAMES,
        NUM_CLASSES,
        f"{model_name}_test"
    )
    with open(f"../results/{model_name}_test_metrics.txt", "w") as f:
        for name, val in zip(model.metrics_names, test_results):
            f.write(f"{name}: {val}\n")

    print(f"Test results saved to ../results/{model_name}_test_confusion_matrix.png")