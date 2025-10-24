import tensorflow as tf

def create_custom_model():
    num_classes = 6  # scratches, inclusion, class3...class6

    # Backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(200,200,3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

    # Detection heads
    bbox_regression = tf.keras.layers.Dense(4, name="bbox")(x)  # [xmin, ymin, xmax, ymax] normalized [0,1]
    class_prediction = tf.keras.layers.Dense(num_classes, activation="softmax", name="class")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=[bbox_regression, class_prediction])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={
            "bbox": "mse",   # You can replace with Smooth L1 later
            "class": "sparse_categorical_crossentropy"
        },
        metrics={"class": "accuracy"}
    )

    model.summary()
    return model