import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('barcode_detector.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('barcode_detector_model.tflite', 'wb') as f:
    f.write(tflite_model)
