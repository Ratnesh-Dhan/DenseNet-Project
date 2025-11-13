import tensorflow as tf
from tensorflow.keras import layers, Model

def build_ssd_like_model(num_classes):
    # Backbone: pretrained MobileNetV2 (acts like VGG in your PyTorch code)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(300, 300, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Select intermediate layers as SSD feature maps
    layer_names = [
        'block_6_expand_relu',  # 38x38
        'block_13_expand_relu', # 19x19
        'out_relu'              # 10x10
    ]
    layers_outputs = [base_model.get_layer(name).output for name in layer_names]
    backbone = Model(inputs=base_model.input, outputs=layers_outputs)
    
    # Detection heads
    feature_maps = backbone.output
    cls_outputs, box_outputs = [], []
    
    for fmap in feature_maps:
        # Classification head
        cls = layers.Conv2D(num_classes * 6, 3, padding='same')(fmap)
        cls = layers.Reshape((-1, num_classes))(cls)
        
        # Box regression head
        box = layers.Conv2D(4 * 6, 3, padding='same')(fmap)
        box = layers.Reshape((-1, 4))(box)
        
        cls_outputs.append(cls)
        box_outputs.append(box)
    
    # Concatenate predictions from all feature maps
    cls_preds = layers.Concatenate(axis=1)(cls_outputs)
    box_preds = layers.Concatenate(axis=1)(box_outputs)
    
    outputs = tf.concat([box_preds, cls_preds], axis=-1)
    
    model = Model(inputs=backbone.input, outputs=outputs)
    return model
