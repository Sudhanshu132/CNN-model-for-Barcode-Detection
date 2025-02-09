import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity

# Load the Keras model
h5_model_path = "D:/Projects/Chat/barcode_detector.h5"
model = tf.keras.models.load_model(h5_model_path)

# Apply model pruning
pruning_schedule = sparsity.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.80,
                                           begin_step=0, end_step=1000)

pruned_model = sparsity.prune_low_magnitude(model, pruning_schedule=pruning_schedule)

# Compile the pruned model (if necessary)
pruned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Now, we convert the pruned model to TensorFlow Lite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)

# Enable post-training quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # This will perform quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # 8-bit quantization

# If you have representative data for calibration
def representative_dataset_gen():
    for input_data in your_data:  # Replace with a small subset of your data
        yield [input_data]

converter.representative_dataset = representative_dataset_gen

# Convert the pruned and quantized model to TensorFlow Lite format
quantized_tflite_model = converter.convert()

# Save the quantized model
quantized_model_path = "D:/Projects/quantized_barcode_detector_model.tflite"
with open(quantized_model_path, "wb") as f:
    f.write(quantized_tflite_model)

print(f"Quantized and pruned TFLite model saved at: {quantized_model_path}")
