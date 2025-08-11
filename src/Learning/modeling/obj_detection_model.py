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
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox')(x)  # [x_min, y_min, x_max, y_max]
    class_output = layers.Dense(num_classes, activation='softmax', name='class')(x)

    model = tf.keras.Model(inputs, outputs=[bbox_output, class_output])

    model.compile(
        optimizer = 'adam',
        loss={
            'bbox': 'mse', # bounding box regression
            'class': 'categorical_crossentropy' # class prediction
        }
    )
    return model

model = object_detection_model()

model.summary()