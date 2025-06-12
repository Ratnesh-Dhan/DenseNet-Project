from tensorflow.keras import layers, models

def create_better_model():
    model = models.Sequential([
        layers.Input(shape=(16, 16, 3)),

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same'), 
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'), 
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),

        # Classifier
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(5, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model