import os
import tensorflow as tf
import numpy as np
from dataset_loader import XMLDatasetTF
from efficientdet import build_efficientdet_like
from graph_utils import GraphUtils  # <-- your imported graph class

# ================= CONFIG =================
root_dir = "/path/to/dataset"
classes_file = "/path/to/classes.txt"
num_classes = len(open(classes_file).read().splitlines())
batch_size = 4
epochs = 20
save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)

optimizers = {
    "Adam": tf.keras.optimizers.Adam(1e-4),
    "SGD": tf.keras.optimizers.SGD(1e-3, momentum=0.9),
    "RMSprop": tf.keras.optimizers.RMSprop(1e-4),
}

# ================= DATA =================
train_ds = XMLDatasetTF("train", root_dir, classes_file, batch_size)
val_ds = XMLDatasetTF("val", root_dir, classes_file, batch_size)
test_ds = XMLDatasetTF("test", root_dir, classes_file, batch_size)

graphs = GraphUtils(save_dir)
final_results = []

# ================= TRAIN PER OPTIMIZER =================
for name, opt in optimizers.items():
    print(f"\n===== Training with {name} =====")

    model = build_efficientdet_like(num_classes)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )

    # ===== Save model =====
    model_path = os.path.join(save_dir, f"{name}_efficientdet.keras")
    model.save(model_path)

    # ===== Collect training history =====
    hist = {
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "train_acc": history.history["accuracy"],
        "val_acc": history.history["val_accuracy"],
    }

    graphs.plot_loss_accuracy(hist, name)

    # ===== Evaluate =====
    y_true, y_pred = [], []
    for imgs, labels in test_ds:
        preds = model.predict(imgs)
        preds = np.argmax(preds, axis=-1)
        y_pred.extend(preds.flatten())
        y_true.extend(np.argmax(labels, axis=-1).flatten())

    metrics = graphs.get_metrics(y_true, y_pred)
    metrics["optimizer"] = name
    final_results.append(metrics)

    graphs.plot_confusion_matrix(y_true, y_pred, name, class_names=open(classes_file).read().splitlines())

# ================= FINAL COMPARISON =================
graphs.plot_optimizer_comparison(final_results)

print("\nTraining complete. Results and graphs saved to:", save_dir)
