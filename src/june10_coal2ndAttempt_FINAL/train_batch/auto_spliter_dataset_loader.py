from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(batch_size):
    # data_dir = r"D:\NML ML Works\newCoalByDeepBhaiya\16_noSplit"
    # data_dir = r"/mnt/d/DATASETS/coal2026/"
    data_dir = r"/media/zumbie/6CA45A53A45A203E/2026-coal_samples/Himanshu Coal Samples 2026/16 size"

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.3, # 80% for training and 20 % for validation
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
        shuffle=False
    )

    return train_generator, validation_generator