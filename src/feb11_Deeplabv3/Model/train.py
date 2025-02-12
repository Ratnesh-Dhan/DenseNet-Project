# from model import model
# from data_loader import train_generator, val_generator, train_images, val_images

# batch_size = 8
# epochs = 10

# history = model.fit(
#     train_generator(),
#     steps_per_epoch=len(train_images) // batch_size,
#     epochs=epochs,
#     validation_data=val_generator(),
#     validation_steps=len(val_images) // batch_size
# )

# # Save the model
# model.save('deeplabv3_plus_model.keras')

# import tensorflow as tf
# from model import model
# from data_loader import train_generator, val_generator, train_images, val_images

# train_dataset = tf.data.Dataset.from_generator(
#     train_generator, 
#     output_types=(tf.float32, tf.float32)
# )

# val_dataset = tf.data.Dataset.from_generator(
#     val_generator, 
#     output_types=(tf.float32, tf.float32)
# )

# batch_size = 8
# train_dataset = train_dataset.batch(batch_size)
# val_dataset = val_dataset.batch(batch_size)

# history = model.fit(
#     train_dataset,
#     epochs=10,
#     validation_data=val_dataset
# )

# # Save the model
# model.save('deeplabv3_plus_model.keras')

import tensorflow as tf
from model import model
from data_loader import train_generator, val_generator, train_images, val_images

batch_size = 8

# Specify output shapes for the dataset
output_signature = (
    tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),  # Image shape
    tf.TensorSpec(shape=(None, 512, 512, 1), dtype=tf.float32)   # Mask shape
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator(batch_size),
    output_signature=output_signature
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: val_generator(batch_size),
    output_signature=output_signature
)

# Calculate steps per epoch
steps_per_epoch = len(train_images) // batch_size
validation_steps = len(val_images) // batch_size

history = model.fit(
    train_dataset,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps
)

# Save the model
model.save('deeplabv3_plus_model.keras')