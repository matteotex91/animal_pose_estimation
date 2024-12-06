from ultralytics import YOLO

model = YOLO("yolov8l-pose.pt")

model.info()

results = model(
    "/Users/matteotex/Documents/VSC/matteotex/animal_pose_estimation/tflite_debug/images/human1.jpg"
)
print("Tensors in results : ")
print(f"Names : {len(results[0].names)}")
print(f"Boxes : {results[0].boxes.shape}")
print(f"Keypoints : {results[0].keypoints.shape}")

print("stop here")

results[0].show()
