import sys
import tensorflow as tf

print(tf.sysconfig.get_build_info())
# sys.exit(0)
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
