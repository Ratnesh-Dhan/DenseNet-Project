"""
corrosion_model.py
──────────────────
UNet with ResNet50 encoder for binary corrosion segmentation.
Import this file in both train.py and inference.py.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, Model


# ═════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING LAYER  (serializable — no Lambda)
# ═════════════════════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable(package="corrosion")
class ResNetPreprocess(layers.Layer):
    """
    Applies ResNet50 preprocessing (zero-centres ImageNet channels).
    Registered with Keras so it survives save/load without custom_objects.
    Input : float32 [0, 255]
    Output: float32 preprocessed for ResNet50
    """
    def call(self, x):
        return preprocess_input(x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()


# ═════════════════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable(package="corrosion")
def dice_loss(y_true, y_pred, smooth=1e-6):
    """Soft Dice loss — handles class imbalance by optimising overlap ratio."""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred,                       [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denominator  = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    return 1.0 - tf.reduce_mean((2.0 * intersection + smooth) / (denominator + smooth))


@tf.keras.utils.register_keras_serializable(package="corrosion")
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Binary focal loss — down-weights easy background pixels.
    alpha: weight for the positive (corrosion) class.
    """
    y_true  = tf.cast(y_true, tf.float32)
    bce     = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    alpha_t = y_true * alpha  + (1.0 - y_true) * (1.0 - alpha)
    return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)


@tf.keras.utils.register_keras_serializable(package="corrosion")
def combined_loss(y_true, y_pred):
    """Focal + Dice. No BCE — BCE scale dominates on imbalanced masks."""
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


# ═════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═════════════════════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable(package="corrosion")
def iou_metric(y_true, y_pred, threshold=0.5):
    """Intersection-over-Union at a fixed sigmoid threshold."""
    y_pred_bin   = tf.cast(y_pred > threshold, tf.float32)
    y_true       = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin)
    union        = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) - intersection
    return (intersection + 1e-6) / (union + 1e-6)


# ═════════════════════════════════════════════════════════════════════════════
#  BUILDING BLOCKS
# ═════════════════════════════════════════════════════════════════════════════

def conv_block(x, filters, dropout_rate=0.0):
    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if dropout_rate > 0.0:
        x = layers.SpatialDropout2D(dropout_rate)(x)
    return x


def upsample_block(x, skip, filters, dropout_rate=0.0):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    if skip is not None:
        x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters, dropout_rate=dropout_rate)
    return x


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL
# ═════════════════════════════════════════════════════════════════════════════

def build_unet_with_resnet50(input_shape=(256, 256, 3), compile_model=True):
    """
    UNet with ResNet50 encoder.

    Skip connections at 256×256 input:
        conv1_relu          → 128×128 × 64
        conv2_block3_out    →  64×64  × 256
        conv3_block4_out    →  32×32  × 512
        conv4_block6_out    →  16×16  × 1024
        conv5_block3_out    →   8×8   × 2048  ← bottleneck
    """
    inputs = layers.Input(shape=input_shape, name='input_image')

    # ── Encoder ──────────────────────────────────────────────────────────────
    x          = ResNetPreprocess(name='resnet_preprocess')(inputs)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)

    for layer in base_model.layers:
        layer.trainable = False

    s1         = base_model.get_layer("conv1_relu").output
    s2         = base_model.get_layer("conv2_block3_out").output
    s3         = base_model.get_layer("conv3_block4_out").output
    s4         = base_model.get_layer("conv4_block6_out").output
    bottleneck = base_model.get_layer("conv5_block3_out").output

    # ── Decoder ──────────────────────────────────────────────────────────────
    d1 = upsample_block(bottleneck, s4,   filters=512, dropout_rate=0.3)  # 16×16
    d2 = upsample_block(d1,         s3,   filters=256, dropout_rate=0.2)  # 32×32
    d3 = upsample_block(d2,         s2,   filters=128, dropout_rate=0.1)  # 64×64
    d4 = upsample_block(d3,         s1,   filters=64,  dropout_rate=0.0)  # 128×128
    d5 = upsample_block(d4,         None, filters=32,  dropout_rate=0.0)  # 256×256

    outputs = layers.Conv2D(1, 1, activation='sigmoid',
                            dtype='float32', name='output_mask')(d5)

    model = Model(inputs=inputs, outputs=outputs, name='UNet_ResNet50')

    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=combined_loss,
            metrics=[
                iou_metric,
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.Precision(name='precision'),
            ]
        )

    return model


# ═════════════════════════════════════════════════════════════════════════════
#  SANITY CHECK
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import numpy as np
    model = build_unet_with_resnet50(input_shape=(256, 256, 3))
    model.summary(line_length=100)
    dummy_img  = np.random.randint(0, 255, (2, 256, 256, 3)).astype('float32')
    dummy_mask = (np.random.rand(2, 256, 256, 1) > 0.8).astype('float32')
    pred = model.predict(dummy_img, verbose=0)
    print("Output shape :", pred.shape)
    print("Loss         :", combined_loss(dummy_mask, pred).numpy())
    print("IoU          :", iou_metric(dummy_mask, pred).numpy())
    # save + reload to confirm serialisation works
    model.save("/tmp/test_model.keras")
    reloaded = tf.keras.models.load_model("/tmp/test_model.keras")
    pred2 = reloaded.predict(dummy_img, verbose=0)
    print("Reload OK, max diff:", float(abs(pred - pred2).max()))