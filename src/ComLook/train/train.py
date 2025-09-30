# Load pretrained model
import tensorflow as tf
import tensorflow_hub as hub

# Pretrained model from TF Hub
detector_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
detector = hub.load(detector_url)

# Dataset preparation
def preprocess_data(example):
    image = example["image"]
    bboxes = example["objects"]["bbox"]   # normalized [ymin, xmin, ymax, xmax]
    labels = example["objects"]["label"]
    return image, (bboxes, labels)

# Modifying the model
def create_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),
                                                include_top=False,
                                                weights="imagenet")

    base_model.trainable = False  # freeze backbone

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Detection head
    bbox_regression = tf.keras.layers.Dense(4, activation="sigmoid", name="bboxes")(x)
    class_prediction = tf.keras.layers.Dense(num_classes, activation="softmax", name="class")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=[bbox_regression, class_prediction])

    # Compile & Train
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                loss={
                    "bboxes": "mse",    # regression loss for bounding boxes
                    "class": "categorical_crossentropy"
                })
    return model

model = create_model(num_classes=2)
history = model.fit(train_dataset, validation_data=val_dataset, epochs=20)

# Fine tuning
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False  # keep lower layers frozen

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), ...)

# Inference
image = load_image("test.jpg")
pred_bboxes, pred_class = model.predict(tf.expand_dims(image, axis=0))

# Draw bounding boxes on image
