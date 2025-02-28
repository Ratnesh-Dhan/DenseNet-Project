# this is the url for the docs :- https://www.tensorflow.org/guide/keras/transfer_learning
from tensorflow import keras

layer = keras.layers.Dense(3)
layer.build((None, 4)) # Create the weight
layer.trainable = False # Freeze the layer (adding this line will make layers un trainable)

print(f'Weight: {len(layer.weights)}')
print(f'Trainable Weights: {len(layer.trainable_weights)}')
print(f'Non-Trainable Weights: {len(layer.non_trainable_weights)}')