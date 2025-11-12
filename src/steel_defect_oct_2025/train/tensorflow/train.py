import tensorflow as tf
from dataset_loader import XMLDatasetTF
from efficientdet import build_efficientdet_like
import os

# Config
root_dir = "../../../../Datasets/NEU-DET/"
classes_file = os.path.join(root_dir, "classes.txt")
num_classes = len(open(classes_file).read().splitlines())
batch_size = 4
epochs = 20

# Data
train_ds = XMLDatasetTF("train", root_dir, classes_file, batch_size)
val_ds = XMLDatasetTF("val", root_dir, classes_file, batch_size)
test_ds = XMLDatasetTF("test", root_dir, classes_file, batch_size)

# Model
model = build_efficientdet_like(num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Save and plot (using your imported functions)
model.save("./model/efficientdet_like_trained.keras")
# plot_optimizer_comparison etc can be reused here if you want metrics visualized.
