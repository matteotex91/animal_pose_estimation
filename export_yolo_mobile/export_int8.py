import tensorflow as tf

# Carica il modello TensorFlow Lite esportato
converter = tf.lite.TFLiteConverter.from_saved_model(
    "/Users/matteotex/Documents/VSC/matteotex/animal_pose_estimation/yolov8n-pose_saved_model/"
)  # Modifica con il path corretto
# converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)  # Se hai un modello Keras

# Abilita la quantizzazione
converter.optimizations = [tf.lite.Optimize.DEFAULT]


# Funzione per generare dati rappresentativi
# def representative_dataset():
#     for _ in range(100):  # Usa un campione di 100 immagini rappresentative
#         # Carica un'immagine campione come array numpy con dimensione corretta (es. 1x320x320x3)
#         img = tf.random.uniform(
#             shape=(1, 320, 320, 3), minval=0, maxval=255, dtype=tf.float32
#         )
#         yield [img]
#
#
# converter.representative_dataset = representative_dataset

# Imposta la quantizzazione su INT8
converter.target_spec.supported_types = [tf.int8]

# Genera il modello quantizzato
quantized_tflite_model = converter.convert()

# Salva il modello quantizzato
with open("yolov8n-pose-int8.tflite", "wb") as f:
    f.write(quantized_tflite_model)

print("Modello quantizzato salvato come yolov8n-pose-int8.tflite")
