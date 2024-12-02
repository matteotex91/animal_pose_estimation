Useful links

YOLO models:
https://learnopencv.com/mastering-all-yolo-models/
https://learnopencv.com/animal-pose-estimation/

DeepLabCut:
https://www.nature.com/articles/s41592-022-01443-0

Deer:
https://www.mdpi.com/2076-2615/14/18/2640
https://www.mdpi.com/2504-446X/8/10/522

Bears (bearid python):
https://bearresearch.org/
https://onlinelibrary.wiley.com/doi/10.1002/ece3.6840

AnimalPose10K
https://github.com/AlexTheBad/AP-10K

Android applications: 
- Test2 : hello world
- Test3 : broken, activities not declare in the manifest
- Test5 : working example with different activities, buttons, preview of the camera and image acquisition. Moreover, it works correctly with the AppCompat, see settings in the manifest, lib.version.toml and build.gradle.kts
- Test6 : working clean example with camera preview
- Test8 : not working attempt to use the overlay of the guess boxes of the tensorflow model
- YOLOv8PoseApp : working example of image processing with a tflite tensorflow model;
  - The processing is slow, less than 10 Hz. Better to see if it's possible to quantize the model , wether reduce the picture size (less than 640x640)
  - The output is meaningless, understand better the data structure and debug if the input is correct.


