from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

def build_unet_with_resnet50(input_shape=(256, 256, 3)):
    # 1. Input Layer
    inputs = Input(shape=input_shape)

    # 2. Load ResNet50 encoder with pretrained ImageNet weights
    # include_top=False removes final Dense layers
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # 3. Extract skip connections
    skip1 = resnet.get_layer("input_1").output            # 256x256
    skip2 = resnet.get_layer("conv1_relu").output         # 128x128
    skip3 = resnet.get_layer("conv2_block3_out").output   # 64x64
    skip4 = resnet.get_layer("conv3_block4_out").output   # 32x32
    bottleneck = resnet.get_layer("conv4_block6_out").output  # 16x16

    # 4. Decoder
    up1 = UpSampling2D()(bottleneck)
    up1 = Concatenate()([up1, skip4])
    up1 = Conv2D(256, 3, activation='relu', padding='same')(up1)

    up2 = UpSampling2D()(up1)
    up2 = Concatenate()([up2, skip3])
    up2 = Conv2D(128, 3, activation='relu', padding='same')(up2)

    up3 = UpSampling2D()(up2)
    up3 = Concatenate()([up3, skip2])
    up3 = Conv2D(64, 3, activation='relu', padding='same')(up3)

    up4 = UpSampling2D()(up3)
    up4 = Concatenate()([up4, skip1])
    up4 = Conv2D(32, 3, activation='relu', padding='same')(up4)

    up5 = UpSampling2D()(up4)
    up5 = Conv2D(16, 3, activation='relu', padding='same')(up5)

    # 5. Final output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(up5)

    model = Model(inputs, outputs)
    return model
