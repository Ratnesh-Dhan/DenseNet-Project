import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def battis_x_battis(num_classes=5, optimizer='adam', lr=1e-4):
    inputs = keras.Input(shape=(30,30,3))

    # block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding="same")(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)

    # block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=output)

    # Choose optimizer on the go
    optimizers = {
        "adam": tf.keras.optimizers.Adam(learning_rate=lr),
        "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=lr),
        "adadelta": tf.keras.optimizers.Adadelta(learning_rate=lr),
        "adagrad": tf.keras.optimizers.Adagrad(learning_rate=lr),
        "nadam": tf.keras.optimizers.Nadam(learning_rate=lr),
    }

    if optimizer not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer}. Choose from {list(optimizers.keys())}")


    model.compile(
        optimizer=optimizers[optimizer],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

