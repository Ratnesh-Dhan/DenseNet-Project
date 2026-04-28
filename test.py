import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# This should return 1 and list the 'RTX PRO 4000'
