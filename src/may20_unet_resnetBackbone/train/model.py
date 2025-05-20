#U-Net with resnet encoder

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

def conv_block(inputs, filters):
    x = layers.Conv2D(filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def build_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Pretrained ResNet50 as encoder
    base_model = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    # Skip connections
    skips = [
        base_model.get_layer("conv1_relu").output,     # 128x128
        base_model.get_layer("conv2_block3_out").output,  # 64x64
        base_model.get_layer("conv3_block4_out").output,  # 32x32
        base_model.get_layer("conv4_block6_out").output,  # 16x16
    ]
    bottleneck = base_model.get_layer("conv5_block3_out").output  # 8x8

    # Decoder
    x = bottleneck
    for skip in reversed(skips):
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, x.shape[-1] // 2)

    x = layers.UpSampling2D((2, 2))(x)  # Final upsample to original size
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)

    return Model(inputs, outputs)
