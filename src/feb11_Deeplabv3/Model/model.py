import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# def efficientnet_b0(input_shape):
#     base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
#     x = base_model.output
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dense(128, activation='relu')(x)
#     x = layers.Dropout(0.2)(x)
#     x = layers.Dense(2, activation='softmax')(x)
#     return keras.Model(inputs=base_model.input, outputs=x)

# def deeplabv3_plus(input_shape):
#     base_model = efficientnet_b0(input_shape)
#     x = base_model.input
#     x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
#     x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
#     x = layers.UpSampling2D(size=(2, 2))(x)
#     x = layers.Conv2D(2, (1, 1), activation='softmax')(x)
#     return keras.Model(inputs=base_model.input, outputs=x)

def efficientnet_b0(input_shape):
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2, activation='softmax')(x)
    return keras.Model(inputs=base_model.input, outputs=x)

def deeplabv3_plus(input_shape):
    base_model = efficientnet_b0(input_shape)
    x = base_model.input
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(2, (1, 1), activation='softmax')(x)
    return keras.Model(inputs=base_model.input, outputs=x)

input_shape = (512, 512, 3)
model = deeplabv3_plus(input_shape)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
