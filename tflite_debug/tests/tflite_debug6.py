import cv2
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import pickle 

""" comparison between the results of the corrected tensorflow tflite model and the ones obtained from the mobile emulator
from this experiment I found out that it is necessary to renormalize the image in the mobile program : currently the processed image in the kotlin code it's not normalize to 1, but is seems like it's processing image with colors from 0 to 255
"""

with open("/Users/matteotex/Documents/VSC/matteotex/animal_pose_estimation/tflite_debug/mobile_tflite_results/human_2_output_normalize.pickle","rb") as f:
# with open("/Users/matteotex/Documents/VSC/matteotex/animal_pose_estimation/tflite_debug/mobile_tflite_results/human_2_output.pickle","rb") as f:
    tflite_output=pickle.load(file=f)


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(
    model_path="/Users/matteotex/Documents/VSC/matteotex/animal_pose_estimation/yolov8n-pose_saved_model/yolov8n-pose_float32.tflite"
)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Read the image
image_path = "/Users/matteotex/Documents/VSC/matteotex/animal_pose_estimation/tflite_debug/images/human2.jpg"
image = cv2.imread(image_path)
image2 = cv2.imread(image_path)

# image = cv2.convertScaleAbs(image_)
image = image.astype("float32")
image = image /255
# Get the input size from the model's input details and resize the image accordingly
input_size = input_details[0]["shape"][1:3]
image = cv2.resize(image, tuple(input_size))

image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Convert the image to a float32 numpy array and add an extra dimension
input_data = np.expand_dims(image.astype(np.float32), axis=0)

# Set the tensor to point to the input data to be used
interpreter.set_tensor(input_details[0]["index"], input_data)

# Run the model
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]["index"])


output_data = output_data[0]
count=0

for r1,r2 in zip(output_data,tflite_output):
    for i1,i2 in zip(r1,r2):
        print(f"computer : {i1} - mobile : {i2}")
        count+=1
        if count>20:
            break
    else:
        break


print("stop here")
