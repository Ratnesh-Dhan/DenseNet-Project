import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

a = tf.random.normal([8,16,16,3])
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8,3,activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2)
])

model(a)