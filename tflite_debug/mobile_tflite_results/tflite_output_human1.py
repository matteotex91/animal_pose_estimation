import pickle

with open("/Users/matteotex/Documents/VSC/matteotex/animal_pose_estimation/tflite_debug/mobile_tflite_results/human_1_output.pickle","rb") as f:
    tflite_output=pickle.load(file=f)