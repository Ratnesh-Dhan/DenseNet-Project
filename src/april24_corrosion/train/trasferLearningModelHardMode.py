import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

def build_unet_with_transfer_learning(input_shape=(512, 512, 3), num_classes=3):
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

# Create the model
# model = build_unet_with_transfer_learning(input_shape=(512, 512, 3), num_classes=3)

# Display the model summary
# model.summary()
