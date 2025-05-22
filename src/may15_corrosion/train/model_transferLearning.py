from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate
# from tensorflow.keras.models import Model

# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Concatenate, Activation
# from tensorflow.keras.models import Model

def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same", activation="relu")(input_tensor)
    x = Conv2D(num_filters, (3, 3), padding="same", activation="relu")(x)
    return x

def build_unet_with_resnet50(input_shape=(256, 256, 3), num_classes=1):
    # Load a pre-trained ResNet50 model without the top classification layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the layers of the base model to prevent training
    for layer in base_model.layers:
        layer.trainable = False

    # Encoder (using pre-trained ResNet50)
    encoder_output = base_model.output

    # Decoder (U-Net style: Conv2DTranspose layers with concatenation)
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(encoder_output)
    u6 = layers.concatenate([u6, base_model.get_layer("conv4_block6_out").output])  # Concatenate with ResNet feature map
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, base_model.get_layer("conv3_block4_out").output])  # Concatenate with ResNet feature map
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, base_model.get_layer("conv2_block3_out").output])  # Concatenate with ResNet feature map
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, base_model.get_layer("conv1_relu").output])  # Concatenate with ResNet feature map
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # Replaced this
    # outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    # With this
    u10 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
    c10 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u10)
    c10 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c10)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c10)
    # Upto this 

    # Create the model
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        # // (from_logits=False) This tells the loss function that the model’s output is already softmax-activated
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return model

def build_unet_with_resnet50_old1(input_shape=(256, 256, 3), num_classes=1):
    inputs = Input(shape=input_shape)

    # Load ResNet50 as the encoder (backbone)
    resnet50 = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)

    # Get skip connection layers
    skip1 = resnet50.get_layer("conv1_relu").output      # 128x128x64
    skip2 = resnet50.get_layer("conv2_block3_out").output # 64x64x256
    skip3 = resnet50.get_layer("conv3_block4_out").output # 32x32x512
    skip4 = resnet50.get_layer("conv4_block6_out").output # 16x16x1024

    # Bottleneck
    bottleneck = resnet50.get_layer("conv5_block3_out").output # 8x8x2048

    # Decoder
    up1 = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding="same")(bottleneck)
    up1 = Concatenate()([up1, skip4])
    up1 = conv_block(up1, 1024)

    up2 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(up1)
    up2 = Concatenate()([up2, skip3])
    up2 = conv_block(up2, 512)

    up3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(up2)
    up3 = Concatenate()([up3, skip2])
    up3 = conv_block(up3, 256)

    up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(up3)
    up4 = Concatenate()([up4, skip1])
    up4 = conv_block(up4, 64)

    # Final upsample to match input resolution (128x128 -> 256x256)
    up_final = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(up4)
    up_final = conv_block(up_final, 32)

    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation="sigmoid")(up_final)

    model = Model(inputs, outputs)
    return model


def build_unet_with_resnet50_old(input_shape=(256, 256, 3)):
    # 1. Input Layer
    inputs = Input(shape=input_shape)

    # 2. Load ResNet50 encoder with pretrained ImageNet weights
    # include_top=False removes final Dense layers
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # 3. Extract skip connections
    # skip1 = resnet.get_layer("input_1").output            # 256x256
    skip1 = resnet.input                              # 256x256
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
