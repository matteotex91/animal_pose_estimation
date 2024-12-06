Useful links

YOLO models:
https://learnopencv.com/mastering-all-yolo-models/
https://learnopencv.com/animal-pose-estimation/
https://github.com/ultralytics/ultralytics/issues/4771 (debug of the keypoints output in a tflite pose model)

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
- Test1 : Working example of drawings overlay to a previewview -> preparation to represent the outcome of the tensorflow model 
- Test2 : hello world
- Test3 : Successfully normalized the input TensorImage colors from 255 to 1, and now the right hand is tagged with a red dot.
- Test5 : working example with different activities, buttons, preview of the camera and image acquisition. Moreover, it works correctly with the AppCompat, see settings in the manifest, lib.version.toml and build.gradle.kts
- Test6 : working clean example with camera preview
- YOLOv8PoseApp : working example of image processing with a tflite tensorflow model;
  - The processing is slow, less than 10 Hz. Better to see if it's possible to quantize the model , wether reduce the picture size (less than 640x640)
  - The output seems meaningless, try to understand better the data structure and debug if the input is correct.
  - Try to quantize the model to int8
- TensorflowModelDebug : working example processing a jpg image. The results are exactly the same provided by the script tflite_debug.py in the pretrained folder of this project. Now it's missing the last part : determine where I can obtain the data I want

YOLO pose model trained with deers
- try to train a new model with small dataset -> 10 images , 1 epoch. Great effort from the macbook air but absolutely zero good results

TODO : 
- investigate over c++ native interface to speed up the elaborations (very complicated)
- Generate a set of deer images with a model in blender : different orientations
- Try dell computer for the model training : CUDA compatibility in linux?
- train model with virtual dataset
