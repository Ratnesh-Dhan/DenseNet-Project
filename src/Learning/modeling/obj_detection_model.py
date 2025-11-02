import tensorflow as tf
from tensorflow.keras import layers

def object_detection_model(input_shape=(224,224,3), num_classes=2):
    inputs = layers.Input(input_shape)

    # Backbone CNN
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)

    # Flatten + Dense for bounding box coords
    x = layers.Flatten()(x)

    bbox_branch = layers.Dense(128, activation='relu')(x)
    bbox_branch = layers.Dropout(0.3)(bbox_branch)  # Drop 30% of neurons
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox')(bbox_branch)

    # --- Classification Branch ---
    class_branch = layers.Dense(128, activation='relu')(x)
    class_branch = layers.Dropout(0.5)(class_branch)  # Drop 50% of neurons for stronger regularization
    class_output = layers.Dense(1, activation='sigmoid', name='class')(class_branch)

    model = tf.keras.Model(inputs, outputs=[bbox_output, class_output])

    model.compile(
        optimizer = 'adam',
        loss={
            'bbox': 'mse', # bounding box regression . use 'mae' for bounding box centers
            'class': 'binary_crossentropy' # binary prediction
            # 'class': 'categorical_crossentropy' # class prediction
        },
        loss_weights={'bbox': 2.0, 'class': 1.0}
    )
    return model

model = object_detection_model()

model.summary()