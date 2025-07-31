from tensorflow import keras

def load_dataset(batch_size):
    train_dir = r"D:\NML ML Works\newCoalByDeepBhaiya\16\TRAINING 16"
    validation_dir = r"D:\NML ML Works\newCoalByDeepBhaiya\16\VALIDATION"

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
    )
    validation_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(16, 16),
        batch_size=batch_size,
        class_mode='sparse',
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(16, 16),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False # Important for prediction-evaluation match
    )

    return train_generator, validation_generator