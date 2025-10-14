import tensorflow as tf
from tensorflow.keras import layers

def build_ssd_model(num_classes, max_boxes=10, input_shape=(200,200,3)):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)

    # Predict boxes: shape (max_boxes, 4)
    bbox_output = layers.Dense(max_boxes*4)(x)
    bbox_output = layers.Reshape((max_boxes, 4), name="bboxes")(bbox_output)

    # Predict classes: shape (max_boxes, num_classes)
    class_output = layers.Dense(max_boxes*num_classes)(x)
    class_output = layers.Reshape((max_boxes, num_classes))(class_output)
    class_output = layers.Activation("softmax", name="class_probs")(class_output)

    model = tf.keras.Model(inputs=base_model.input, outputs=[bbox_output, class_output])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={"bboxes": "mse", "class_probs": "sparse_categorical_crossentropy"},
        metrics={"class_probs": "accuracy"}
    )

    model.summary()
    return model
