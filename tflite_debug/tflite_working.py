import cv2
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf


def draw_bbox_on_image(image, x, y, w, h):
    # Denormalize the coordinates
    x = int(x * image.shape[1])
    y = int(y * image.shape[0])
    w = int(w * image.shape[1])
    h = int(h * image.shape[0])

    # Calculate the (x1, y1) and (x2, y2) points for the rectangle
    x1, y1 = x - w // 2, y - h // 2
    x2, y2 = x + w // 2, y + h // 2

    # Draw the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def plot_keypoints_on_image(image, keypoints, t):
    # Iterate over the keypoints
    for keypoint in keypoints:
        x, y, visibility = keypoint

        # Check if the visibility is greater than the threshold
        if visibility > t:
            # Denormalize the coordinates
            x = int(x * image.shape[1])
            y = int(y * image.shape[0])

            # Draw the keypoint
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    return image


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(
    model_path="/Users/matteotex/Documents/VSC/matteotex/animal_pose_estimation/tflite_debug/tflite_model/yolov8n-pose_float32.tflite"
)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Read the image
# image_path = "/Users/matteotex/Documents/VSC/matteotex/animal_pose_estimation/tflite_debug/images/human1.jpg"
image_path = "/Users/matteotex/Documents/VSC/matteotex/animal_pose_estimation/tflite_debug/images/human2.jpg"
# image_path = "/Users/matteotex/Documents/VSC/matteotex/animal_pose_estimation/tflite_debug/images/image1.png"
image = cv2.imread(image_path)
image2 = cv2.imread(image_path)

# image = cv2.convertScaleAbs(image_)
image = image.astype("float32")
image = image / 255.0
# Get the input size from the model's input details and resize the image accordingly
input_size = input_details[0]["shape"][1:3]
image = cv2.resize(image, tuple(input_size))
image2 = cv2.resize(image2, tuple(input_size))

image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Convert the image to a float32 numpy array and add an extra dimension
input_data = np.expand_dims(image.astype(np.float32), axis=0)

# Set the tensor to point to the input data to be used
interpreter.set_tensor(input_details[0]["index"], input_data)

# Run the model
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]["index"])


output_data_transposed = output_data[0].T

# Select the top K bboxes
K = 10  # Change this to your desired number of bboxes
BASE = 0
sorted_indices = np.argsort(output_data_transposed[:, 4])[::-1]
top_K_by_confidence = output_data_transposed[sorted_indices[BASE : BASE + K]]
print("top_K_by_confidence", top_K_by_confidence[0])

# Process each bbox
for bbox in top_K_by_confidence:
    # Select the first 51 elements and reshape it into 17x3
    keypoints = bbox[5:].reshape((17, 3))
    xywh = bbox[:4]
    image_1 = draw_bbox_on_image(image_, xywh[0], xywh[1], xywh[2], xywh[3])
    image_1 = plot_keypoints_on_image(image_1, keypoints, 0.7)

    image2 = draw_bbox_on_image(image2, xywh[0], xywh[1], xywh[2], xywh[3])
    image2 = plot_keypoints_on_image(image2, keypoints, 0.7)

# Save the image
# cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# cv2.imshow("windows",image_)
# cv2.waitKey(1) & 0xFF == 27 # Press 'Esc' to exit

output_path = "/Users/matteotex/Documents/VSC/matteotex/animal_pose_estimation/tflite_debug/images/output.png"
cv2.imwrite(output_path, image2)
