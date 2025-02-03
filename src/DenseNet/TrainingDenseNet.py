# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:27:20 2025
# We are goin to train Dense-net model

@author: Paddy King
"""

import tensorflow as tf

#SETTING THE NUMBER OF THREADS FOR MORE CPU UTILIZATION (not nessasary step. can be commented out but required for better performance)
tf.config.threading.set_intra_op_parallelism_threads(12) #can be adjusted according to cpu cores
tf.config.threading.set_inter_op_parallelism_threads(12) #can be adjusted according to cpu cores

# We will use the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes.
# There are 50,000 training images and 10,000 test images. 
# The dataset is available in the TensorFlow Keras API, so we can load it directly.
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

#preprocess the data by normalizing the pixel values and one-hot encoding the labels.
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# #ADDING CODE TO IMPROVE EFFICIENCY
# train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
# train_dataset = train_dataset.shuffle(buffer_size-50000).batch(64).prefetch(tf.data.autotune)
# #/ ADDING CODE TO IMPROVE EFFICIENCY

#BUILDING THE MODEL
class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')
        self.conv1 = tf.keras.layers.Conv2D(filters=4 * growth_rate, kernel_size=(1, 1), padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=growth_rate, kernel_size=(3, 3), padding='same', use_bias=False)
    
    def call(self, inputs):
        x = self.bn1(inputs)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        output = tf.keras.layers.concatenate([inputs, x], axis=-1)
        return output

class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = [DenseLayer(growth_rate) for _ in range(num_layers)]
        
    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(TransitionLayer, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', use_bias=False)
        self.pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
    
    def call(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        x = self.conv(x)
        output = self.pool(x)
        return output

class DenseNet(tf.keras.Model):
    def __init__(self, num_blocks=3, num_layers=16, growth_rate=12, num_classes=10):
        super(DenseNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=2 * growth_rate, kernel_size=(3, 3), padding='same', use_bias=False)
        self.blocks = [DenseBlock(num_layers, growth_rate) for _ in range(num_blocks)]
        self.trans_layers = [TransitionLayer(num_layers * growth_rate) for _ in range(num_blocks - 1)]
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i != len(self.blocks) - 1:
                x = self.trans_layers[i](x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        output = self.fc(x)
        return output
    
    
# TRAINING MODEL
model = DenseNet(num_blocks=3, num_layers=16, growth_rate=12, num_classes=10)

model.compile(
  optimizer=tf.keras.optimizers.Adam(), 
  loss=tf.keras.losses.categorical_crossentropy, 
  metrics=['accuracy']
)

model.fit(
  train_images, 
  train_labels, 
  epochs=1, 
  batch_size=64, 
  # batch_size=128, 
  validation_data=(test_images, test_labels)
)

# Save the model
model.save('../../TrainedModel/TestDensenet_cifar10.h5')

# Evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test loss : ", test_loss)
print("Test accuracy : ", test_acc)