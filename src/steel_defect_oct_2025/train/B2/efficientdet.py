import tensorflow as tf
from tensorflow.keras import layers, Model

def build_efficientdet_like(num_classes):
    # EfficientNet backbone
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, input_shape=(512, 512, 3), weights='imagenet'
    )
    
    # Pick intermediate feature maps for BiFPN
    C3 = base.get_layer('block4a_expand_activation').output
    C4 = base.get_layer('block6a_expand_activation').output
    C5 = base.get_layer('top_activation').output
    
    # Simple BiFPN-like fusion
    P3 = layers.Conv2D(64, 1, padding='same')(C3)
    P4 = layers.Conv2D(64, 1, padding='same')(C4)
    P5 = layers.Conv2D(64, 1, padding='same')(C5)
    
    P4 = layers.Add()([P4, layers.UpSampling2D()(P5)])
    P3 = layers.Add()([P3, layers.UpSampling2D()(P4)])
    
    features = [P3, P4, P5]
    
    cls_outputs, box_outputs = [], []
    num_anchors = 9
    
    for f in features:
        # Shared conv
        f = layers.Conv2D(64, 3, padding='same', activation='relu')(f)
        # Class head
        cls = layers.Conv2D(num_classes * num_anchors, 3, padding='same')(f)
        cls = layers.Reshape((-1, num_classes))(cls)
        # Box head
        box = layers.Conv2D(4 * num_anchors, 3, padding='same')(f)
        box = layers.Reshape((-1, 4))(box)
        cls_outputs.append(cls)
        box_outputs.append(box)
    
    cls_preds = layers.Concatenate(axis=1)(cls_outputs)
    box_preds = layers.Concatenate(axis=1)(box_outputs)
    
    outputs = tf.concat([box_preds, cls_preds], axis=-1)
    return Model(inputs=base.input, outputs=outputs)

# example
model = build_efficientdet_like(num_classes=5)
model.summary()
