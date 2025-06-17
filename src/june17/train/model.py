from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        layers.Input(shape=(30, 30, 3)),

        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        # layers.Conv2D(256, (3, 3), padding='same'),
        # layers.BatchNormalization(),
        # layers.ReLU(),

        layers.GlobalAveragePooling2D(),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(6, activation='softmax')
    ])

    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


model = create_model()
print(model.summary())