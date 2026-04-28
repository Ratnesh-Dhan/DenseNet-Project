import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model

# Custom Dice Loss (for binary segmentation)
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def build_unet_with_resnet50(input_shape=(512, 512, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Optional: Freeze base_model layers initially
    for layer in base_model.layers:
        layer.trainable = True  # Set False if you want to freeze initially

    encoder_output = base_model.output

    # Decoder
    u6 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(encoder_output)
    u6 = layers.concatenate([u6, base_model.get_layer("conv4_block6_out").output])
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    u7 = layers.concatenate([u7, base_model.get_layer("conv3_block4_out").output])
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    u8 = layers.concatenate([u8, base_model.get_layer("conv2_block3_out").output])
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    u9 = layers.concatenate([u9, base_model.get_layer("conv1_relu").output])
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(c9)

    u10 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c9)
    c10 = layers.Conv2D(32, 3, activation='relu', padding='same')(u10)
    c10 = layers.Conv2D(32, 3, activation='relu', padding='same')(c10)

    # Final binary segmentation output (1 channel + sigmoid)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c10)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Loss: Binary Crossentropy + Dice Loss
    def combined_loss(y_true, y_pred):
        return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) + dice_loss(y_true, y_pred)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=combined_loss,
                  metrics=['accuracy'])

    return model
