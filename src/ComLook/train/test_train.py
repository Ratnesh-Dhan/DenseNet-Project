"""
Transfer Learning for Object Detection using TensorFlow 2.x
Model: SSD MobileNetV2 (pretrained on COCO)
"""

import os
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Pretrained Model
# -------------------------------
print("Loading pretrained model...")
MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
detector = hub.load(MODEL_URL)

# -------------------------------
# 2. Load Example Dataset
# -------------------------------
# Here we use TensorFlow Datasets (TFDS) -> VOC dataset
import tensorflow_datasets as tfds

dataset, info = tfds.load("voc/2007", with_info=True)
train_ds, val_ds = dataset["train"], dataset["validation"]

# Preprocess: resize + normalize
IMG_SIZE = 224

def preprocess(sample):
    image = tf.image.resize(sample["image"], (IMG_SIZE, IMG_SIZE)) / 255.0
    bboxes = sample["objects"]["bbox"]   # normalized [ymin, xmin, ymax, xmax]
    labels = tf.one_hot(sample["objects"]["label"], depth=info.features["objects"]["label"].num_classes)
    return image, (bboxes, labels)

train_ds = train_ds.map(preprocess).batch(8).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(preprocess).batch(8).prefetch(tf.data.AUTOTUNE)

NUM_CLASSES = info.features["objects"]["label"].num_classes

# -------------------------------
# 3. Build Transfer Learning Model
# -------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Freeze backbone initially

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

# Detection heads
bbox_regression = tf.keras.layers.Dense(4, activation="sigmoid", name="bboxes")(x)
class_prediction = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="class")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=[bbox_regression, class_prediction])

# -------------------------------
# 4. Compile Model
# -------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
        "bboxes": "mse",   # bounding box regression loss
        "class": "categorical_crossentropy"
    },
    metrics={"class": "accuracy"}
)

# -------------------------------
# 5. Train Model
# -------------------------------
EPOCHS = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# -------------------------------
# 6. Fine-Tuning
# -------------------------------
base_model.trainable = True
for layer in base_model.layers[:100]:  # freeze first 100 layers
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss={
        "bboxes": "mse",
        "class": "categorical_crossentropy"
    },
    metrics={"class": "accuracy"}
)

history_fine = model.fit(train_ds, validation_data=val_ds, epochs=5)

# -------------------------------
# 7. Inference on Test Image
# -------------------------------
def plot_predictions(image, bboxes, labels, class_names):
    plt.imshow(image)
    h, w, _ = image.shape
    for box, label in zip(bboxes, labels):
        ymin, xmin, ymax, xmax = box
        xmin, xmax, ymin, ymax = xmin*w, xmax*w, ymin*h, ymax*h
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                             fill=False, color="red", linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(xmin, ymin, class_names[tf.argmax(label).numpy()],
                 color="yellow", fontsize=8, backgroundcolor="black")
    plt.axis("off")
    plt.show()

# Take one validation image
for img, (bbox, lbl) in val_ds.take(1):
    pred_bboxes, pred_classes = model.predict(img)
    plot_predictions(img[0].numpy(), [pred_bboxes[0]], [pred_classes[0]], info.features["objects"]["label"].names)
