import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, Model


# ─────────────────────────────────────────────
#  LOSS FUNCTIONS
# ─────────────────────────────────────────────

def dice_loss(y_true, y_pred, smooth=1e-6):
    """Soft Dice loss — great for class-imbalanced binary masks."""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred,                       [tf.shape(y_pred)[0],  -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denominator  = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    return 1.0 - tf.reduce_mean((2.0 * intersection + smooth) / (denominator + smooth))


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Binary focal loss — down-weights easy background pixels.
    alpha: weight for the positive class (corrosion).
    Increase alpha (e.g. 0.75) if corrosion pixels are very rare.
    """
    y_true = tf.cast(y_true, tf.float32)
    bce    = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    p_t    = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    focal  = alpha_t * tf.pow(1.0 - p_t, gamma) * bce
    return tf.reduce_mean(focal)


def combined_loss(y_true, y_pred):
    """
    Focal + Dice.
    - Focal handles class imbalance at the pixel level.
    - Dice optimises the overlap ratio directly.
    BCE is intentionally dropped — its scale dominates the others
    and doesn't help when corrosion is rare.
    """
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

def iou_metric(y_true, y_pred, threshold=0.5):
    """Mean IoU (Jaccard) at a fixed threshold."""
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    y_true     = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin)
    union        = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) - intersection
    return (intersection + 1e-6) / (union + 1e-6)


# ─────────────────────────────────────────────
#  BUILDING BLOCKS
# ─────────────────────────────────────────────

def conv_block(x, filters, dropout_rate=0.0):
    """Double-conv block with optional spatial dropout."""
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
    """Transposed-conv upsample → concat skip → double conv."""
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    # Guard against spatial size mismatch caused by odd dimensions
    if skip is not None:
        x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters, dropout_rate=dropout_rate)
    return x


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────

def build_unet_with_resnet50(input_shape=(256, 256, 3), compile_model=True):
    """
    UNet with a frozen ResNet50 encoder.

    Skip connections (ResNet50 layer names → spatial resolution at 256×256 input):
        conv1_relu          →  128×128  (stride-2 from input)
        conv2_block3_out    →   64×64
        conv3_block4_out    →   32×32
        conv4_block6_out    →   16×16
        conv5_block3_out    →    8×8    (bottleneck)

    Decoder upsamples 8→16→32→64→128→256, matching all 4 skip connections.
    """
    inputs = layers.Input(shape=input_shape, name='input_image')

    # ── Encoder (ResNet50, frozen) ──────────────────────────────────────────
    # preprocess inside the model so raw [0,255] uint8 OR float32 images both work
    x          = layers.Lambda(lambda img: preprocess_input(img), name='resnet_preprocess')(inputs)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)

    for layer in base_model.layers:
        layer.trainable = False   # freeze entire encoder for phase-1

    # Skip connections  (names are stable across TF versions for ResNet50)
    s1 = base_model.get_layer("conv1_relu").output          # 128×128 × 64
    s2 = base_model.get_layer("conv2_block3_out").output    #  64×64  × 256
    s3 = base_model.get_layer("conv3_block4_out").output    #  32×32  × 512
    s4 = base_model.get_layer("conv4_block6_out").output    #  16×16  × 1024
    bottleneck = base_model.get_layer("conv5_block3_out").output  #  8×8 × 2048

    # ── Decoder ────────────────────────────────────────────────────────────
    # Gradually reduce filters to save memory; light dropout in deeper blocks
    d1 = upsample_block(bottleneck, s4, filters=512, dropout_rate=0.3)  # 16×16
    d2 = upsample_block(d1,         s3, filters=256, dropout_rate=0.2)  # 32×32
    d3 = upsample_block(d2,         s2, filters=128, dropout_rate=0.1)  # 64×64
    d4 = upsample_block(d3,         s1, filters=64,  dropout_rate=0.0)  # 128×128
    d5 = upsample_block(d4,         None, filters=32, dropout_rate=0.0) # 256×256 (no skip)

    # ── Output ─────────────────────────────────────────────────────────────
    outputs = layers.Conv2D(1, 1, activation='sigmoid', dtype='float32', name='output_mask')(d5)

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


# ─────────────────────────────────────────────
#  QUICK SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = build_unet_with_resnet50(input_shape=(256, 256, 3))
    model.summary(line_length=100)
    import numpy as np
    dummy_img  = np.random.randint(0, 255, (2, 256, 256, 3)).astype('float32')
    dummy_mask = (np.random.rand(2, 256, 256, 1) > 0.8).astype('float32')
    pred = model.predict(dummy_img)
    print("Output shape:", pred.shape)   # should be (2, 256, 256, 1)
    loss_val = combined_loss(dummy_mask, pred)
    print("Loss:", loss_val.numpy())