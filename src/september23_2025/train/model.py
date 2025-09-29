from tensorflow.keras import layers, models
import tensorflow as tf

def LVGG16_16x16(num_classes=5, optimizer='adam', lr=1e-4):
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(16,16,3)))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))  # 16 → 8

    # Block 2
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))  # 8 → 4

    # Block 3 (stop earlier, otherwise collapse)
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))  # 4 → 2

    # Flatten + FC
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

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
        # loss='categorical_crossentropy',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model



