import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, Model

# Custom Dice Loss (for binary segmentation)
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0],-1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0],-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    return 1 - tf.reduce_mean((2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) + smooth))

# IoU Metric (better than accuracy)
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-6)

# Conv Block
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def build_unet_with_resnet50(input_shape=(512, 512, 3)):
    inputs = layers.Input(shape=input_shape)

    x = preprocess_input(inputs)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)

    # Optional: Freeze base_model layers initially
    for layer in base_model.layers:
        layer.trainable = False  # Set False if you want to freeze initially

    # Skip connection
    s1 = base_model.get_layer("conv1_relu").output
    s2 = base_model.get_layer("conv2_block3_out").output
    s3 = base_model.get_layer("conv3_block4_out").output
    s4 = base_model.get_layer("conv4_block6_out").output

    b1 = base_model.output # botleneck

    # Decoder
    d1 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(b1)
    d1 = layers.concatenate([d1, s4])
    d1 = conv_block(d1, 512)

    d2 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(d1)
    d2 = layers.concatenate([d2, s3])
    d2 = conv_block(d2, 256)

    d3 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(d2)
    d3 = layers.concatenate([d3, s2])
    d3 = conv_block(d3, 128)

    d4 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(d3)
    d4 = layers.concatenate([d4, s1])
    d4 = conv_block(d4, 64)

    d5 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(d4)
    d5 = conv_block(d5, 32)

    # Final binary segmentation output (1 channel + sigmoid)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d5)

    model = Model(inputs, outputs)

    # Loss: Binary Crossentropy + Dice Loss
    def combined_loss(y_true, y_pred):
        return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) + dice_loss(y_true, y_pred)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=combined_loss,
                  metrics=[iou_metric])

    return model
