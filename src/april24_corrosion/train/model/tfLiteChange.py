import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model('unet_resnet50_multiclass.h5')

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open('unet_resnet50_tfLite_model.tflite', 'wb') as f:
    f.write(tflite_model)
