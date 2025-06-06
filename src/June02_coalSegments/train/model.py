from tensorflow.keras import layers, models

def create_model():
    # model = models.Sequential([
    #     layers.Input(shape=(31,31,3)), # RGB input
    #     layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    #     layers.MaxPooling2D((2, 2)),
    #     # Lets add extra 1 more layer
    #     layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    #     layers.MaxPooling2D(2, 2),
    #     layers.Flatten(),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dense(3, activation='softmax') # 2 for 2 classes.
    # ])

    model = models.Sequential([
        layers.Input(shape=(31, 31, 3)),

        layers.Conv2D(16, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model