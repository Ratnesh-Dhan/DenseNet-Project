#   https://keras.io/examples/vision/yolov8/


# from keras_cv.models import EfficientDet
# import tensorflow as tf

# def get_model(CLASS_NAMES):
#     # B0 = lightest, B7 = largest
#     model = EfficientDet.from_preset(
#         "efficientdet_d0",
#         num_classes=len(CLASS_NAMES)
#     )

#     # Freeze backbone for transfer learning
#     for layer in model.layers[:100]:
#         layer.trainable = False

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(1e-4),
#         classification_loss="focal",
#         box_loss="smoothl1",
#     )
#     return model

import keras_cv
import tensorflow as tf

def get_model(CLASS_NAMES, CLASS_MAP):
    # B0 = lightest, B7 = largest

    # Issue 1: "efficientdet_d0" is not a standard preset name with weights.
    # We use "efficientdet_d0_coco" (trained on COCO dataset) or 
    # "efficientdet_d0_pascalvoc" (trained on Pascal VOC dataset) instead.
    # model = keras_cv.models.EfficientDet.from_preset(
    #     "efficientdet_d0_coco",
    #     num_classes=len(CLASS_NAMES)
    # )
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_s_backbone_coco" , # We will use yolov8 small backbone with coco weights
        num_classes=len(CLASS_NAMES)
    )
    model = keras_cv.models.YOLOV8Detector(
        num_classes=len(CLASS_MAP),
        bounding_box_format="yxyx",
        backbone=backbone,
        # fpn_depth=1,
    )

    # Freezing the backbone for transfer learning is a valid approach.
    # The layers can be accessed via model.backbone.layers if you want
    # more granular control over which parts are frozen.
    for layer in model.layers[:100]:
        layer.trainable = False

    # Issue 2: The `compile` method for KerasCV object detection models 
    # expects a dictionary for the `loss` argument if you want to override 
    # specific losses, or you can omit the argument to use defaults.
    # The `classification_loss` and `box_loss` arguments are not standard.
    
    # We pass the losses as a dictionary to the `loss` argument.
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(1e-4),
    #     loss={
    #         "classification_loss": keras_cv.losses.FocalLoss(from_logits=True),
    #         "box_loss": keras_cv.losses.CIoULoss(bounding_box_format="yxyx"), # format depends on your data
    #     }
    # )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        classification_loss= keras_cv.losses.FocalLoss(from_logits=True),
        box_loss= keras_cv.losses.CIoULoss(bounding_box_format="yxyx"), # format depends on your data
        
    )
    return model