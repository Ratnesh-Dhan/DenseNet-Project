from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model

def build_unet_with_transfer_learning(input_shape=(512, 512, 3), num_classes=3):
    # Load a pre-trained ResNet50 model without the top classification layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Encoder (using pre-trained ResNet50)
    encoder_output = base_model.output

    # Decoder (you can add the decoder layers here similar to your original code)
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(encoder_output)
    # Add other decoder layers...

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(u9)

    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Optionally freeze the layers of the encoder
    for layer in base_model.layers:
        layer.trainable = False

    return model
