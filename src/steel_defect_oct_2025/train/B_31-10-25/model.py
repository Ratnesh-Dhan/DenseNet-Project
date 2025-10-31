from keras_cv.models import EfficientDet
import tensorflow as tf

def get_model(CLASS_NAMES):
    # B0 = lightest, B7 = largest
    model = EfficientDet.from_preset(
        "efficientdet_d0",
        num_classes=len(CLASS_NAMES)
    )

    # Freeze backbone for transfer learning
    for layer in model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        classification_loss="focal",
        box_loss="smoothl1",
    )
    return model
