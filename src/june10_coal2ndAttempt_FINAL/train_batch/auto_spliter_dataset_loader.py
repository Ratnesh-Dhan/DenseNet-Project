from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(batch_size):
    data_dir = r"D:\NML ML Works\newCoalByDeepBhaiya\16_noSplit"

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2, # 80% for training and 20 % for validation
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(16,16),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(16,16),
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        shuffle=True
    )

    return train_generator, validation_generator