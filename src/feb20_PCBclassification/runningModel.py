from PCBDataset import PCBDataset
from tensorflow import keras
from matplotlib import pyplot as plt
import os, numpy as np, sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
pcb = PCBDataset("./")

model = keras.models.load_model("pcb_component_classifier.h5")
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Test if the model has loaded successfully by checking its summary
# model.summary()

def prdictor(path):
    og_image = plt.imread(path)
    image = pcb.preprocess_image(image=og_image)

    # Normalize and add the batch dimension
    image_array = np.expand_dims(image, axis=0)  # Shape becomes (1, 224, 224, 3)
    image_array = image_array.astype('float32') / 255.0  # Normalize the image
    print(image_array.shape)  # Should output: (1, 224, 224, 3)
    prediction = model.predict(image_array)
    print(prediction)
    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability
    class_mapping = {
        0: 'Cap1', 1: 'Cap2', 2: 'Cap3', 3: 'Cap4', 
        4: 'MOSFET', 5: 'Mov', 6: 'Resistor', 7: 'Transformer'
    }
    print(f"Predicted Class: {class_mapping[predicted_class]}, production number: {predicted_class}")

# prdictor("./img/test_image.jpg")
# img_folder = os.path.join(pcb.base_path, 'img')

files = os.listdir("./img/")  # Lists all files and directories
# print(files)
files = [f for f in files if os.path.isfile(os.path.join("./img/", f))]  # Only files
# print(files)

for jpg_file in files:
    print(f"Processing file: {jpg_file}")
    prdictor("./img/"+jpg_file)
