from ultralytics import YOLO
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


# Carica il modello YOLOv8
model = YOLO("yolov8l-pose.pt")  # Sostituisci con il percorso del tuo modello

# Esporta il modello in formato ONNX
model.export(format="onnx", opset=11)


# Carica il file ONNX
model_onnx = onnx.load("yolov8l-pose.onnx")

# Converti in formato TensorFlow
tf_rep = prepare(model_onnx)
tf_rep.export_graph("yolov8l-pose-tf")


# Carica il modello TensorFlow
converter = tf.lite.TFLiteConverter.from_saved_model("yolov8l-pose-tf")

# Ottimizza il modello per dispositivi mobili (opzionale)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Esporta il modello in formato TensorFlow Lite
tflite_model = converter.convert()

# Salva il modello TensorFlow Lite
with open("yolov8l-pose.tflite", "wb") as f:
    f.write(tflite_model)
