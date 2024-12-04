import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ultralytics.utils.ops import scale_coords
from tflite_output_human1 import tflite_output as o1
from tflite_output_human2 import tflite_output as o2

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(
    model_path="/Users/matteotex/Documents/VSC/matteotex/animal_pose_estimation/yolov8n-pose_saved_model/yolov8n-pose_float32.tflite"
)
interpreter.allocate_tensors()

image = cv2.imread(
    "/Users/matteotex/Documents/VSC/matteotex/animal_pose_estimation/pretrained/images/human2.jpg"
)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_size = input_details[0]["shape"][1:3]
image = cv2.resize(image, tuple(input_size))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to a float32 numpy array and add an extra dimension
input_data = np.expand_dims(image.astype(np.float32), axis=0)

# Set the tensor to point to the input data to be used
interpreter.set_tensor(input_details[0]["index"], input_data)

# Run the model
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]["index"])


output_data_transposed = output_data[0].T

# Print the output shape
print("output_data_transposed", output_data_transposed.shape)

# Select the bbox with the highest confidence

print("Argmax:", np.argmax(output_data_transposed[:, -1]))
bbox = output_data_transposed[np.argmax(output_data_transposed[:, -1])]

print("Bbox shape:", bbox.shape)

# Select the first 51 elements and reshape it into 17x3
keypoints = bbox[5:].reshape((17, 3))
keypoints = scale_coords(input_size, keypoints, image.shape).round()

print("stop here")
