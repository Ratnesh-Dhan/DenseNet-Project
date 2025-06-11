from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        layers.Input(shape=(16, 16, 3)),

        # First Block
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        # Second Block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        # Global Average Pooling instead of multiple MaxPooling
        layers.GlobalAveragePooling2D(),  # Keeps semantic info without too much spatial loss

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(5, activation='softmax')  # 5-class output
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Task: Patch-based classification for 5 classes using sliding window inference
# Patch size: 15×15 RGB
# Desired: Avoid excessive downsampling, retain spatial info, accurate class prediction

