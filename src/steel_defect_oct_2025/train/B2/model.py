import tensorflow as tf
from tensorflow.keras import layers

def custom_ssd_loss(y_true, y_pred):
    """
    y_true and y_pred are dicts with:
        - 'bboxes': [batch, MAX_BOXES, 4]
        - 'class_probs': [batch, MAX_BOXES]
        - 'mask': [batch, MAX_BOXES]
    """
    bboxes_true = y_true["bboxes"]
    class_true = y_true["class_probs"]
    mask = y_true["mask"]  # 1 for real boxes, 0 for padded boxes

    bboxes_pred = y_pred["bboxes"]
    class_pred = y_pred["class_probs"]

    # --- Mask out padded boxes ---
    valid_mask = tf.cast(mask, tf.float32)

    # --- Bounding box regression loss ---
    bbox_loss_per_box = tf.reduce_sum(tf.square(bboxes_true - bboxes_pred), axis=-1)  # [batch, MAX_BOXES]
    bbox_loss = tf.reduce_sum(bbox_loss_per_box * valid_mask, axis=-1) / (tf.reduce_sum(valid_mask, axis=-1) + 1e-6)

    # --- Classification loss ---
    # Replace padded class labels (-1) with 0 temporarily to avoid sparse_categorical_crossentropy errors
    safe_class_true = tf.where(class_true < 0, tf.zeros_like(class_true), class_true)
    class_loss_per_box = tf.keras.losses.sparse_categorical_crossentropy(
        safe_class_true, class_pred, from_logits=False
    )
    class_loss = tf.reduce_sum(class_loss_per_box * valid_mask, axis=-1) / (tf.reduce_sum(valid_mask, axis=-1) + 1e-6)

    # --- Total loss ---
    total_loss = bbox_loss + class_loss
    return tf.reduce_mean(total_loss)

# def custom_ssd_loss(y_true, y_pred):
#     # Unpack targets and predictions
#     bboxes_true = y_true["bboxes"]
#     class_true = y_true["class_probs"]
#     mask = y_true["mask"]  # we’ll add this to dataset later

#     bboxes_pred = y_pred["bboxes"]
#     class_pred = y_pred["class_probs"]

#     # === Bounding box regression loss ===
#     bbox_loss = tf.reduce_sum(tf.square(bboxes_true - bboxes_pred), axis=-1)  # [batch, MAX_BOXES]

#     # === Classification loss ===
#     class_loss = tf.keras.losses.sparse_categorical_crossentropy(class_true, class_pred)  # [batch, MAX_BOXES]

#     # === Apply mask so padded boxes don’t count ===
#     bbox_loss = tf.reduce_sum(bbox_loss * mask, axis=-1) / (tf.reduce_sum(mask, axis=-1) + 1e-6)
#     class_loss = tf.reduce_sum(class_loss * mask, axis=-1) / (tf.reduce_sum(mask, axis=-1) + 1e-6)

#     # Combine losses
#     total_loss = bbox_loss + class_loss
#     return tf.reduce_mean(total_loss)

def masked_accuracy(y_true, y_pred):
    """
    y_true and y_pred are dicts with:
        - 'class_probs': [batch, MAX_BOXES] or [batch, MAX_BOXES, num_classes]
        - 'mask': [batch, MAX_BOXES]
    """
    class_true = y_true["class_probs"]
    mask = y_true["mask"]  # 1 for real boxes, 0 for padded boxes

    # Predicted classes
    class_pred = tf.argmax(y_pred, axis=-1)  # [batch, MAX_BOXES]

    # Compare predictions to ground truth
    correct = tf.cast(tf.equal(class_true, class_pred), tf.float32)  # [batch, MAX_BOXES]

    # Apply mask so padded boxes are ignored
    correct *= mask

    # Compute accuracy
    return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + 1e-6)

def build_ssd_model(num_classes, max_boxes=10, input_shape=(200,200,3)):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)

    # Predict boxes: shape (max_boxes, 4)
    bbox_output = layers.Dense(max_boxes*4)(x)
    bbox_output = layers.Reshape((max_boxes, 4), name="bboxes")(bbox_output)

    # Predict classes: shape (max_boxes, num_classes)
    class_output = layers.Dense(max_boxes*num_classes)(x)
    class_output = layers.Reshape((max_boxes, num_classes))(class_output)
    class_output = layers.Activation("softmax", name="class_probs")(class_output)

    model = tf.keras.Model(inputs=base_model.input, outputs={"bboxes": bbox_output, "class_probs":class_output})

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        # loss={"bboxes": "mse", "class_probs": "sparse_categorical_crossentropy"},
        loss=custom_ssd_loss,
        metrics={"class_probs": masked_accuracy}
    )

    # model.summary()
    return model
