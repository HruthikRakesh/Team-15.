import tensorflow as tf
import os

# Path to your original Keras model (.keras file)
original_model_path = 'model/best_model.keras'

# Path where the new, smaller model will be saved
quantized_model_path = 'model/grape_model_quantized.tflite'

os.makedirs(os.path.dirname(quantized_model_path), exist_ok=True)
print(f"Loading model from: {original_model_path}")

try:
    model = tf.keras.models.load_model(original_model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- THIS IS THE NEW LINE WE ARE ADDING ---
# It ensures the model is compatible with older TFLite runtimes like the one on Render.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Converting model to TensorFlow Lite format with quantization...")
tflite_quantized_model = converter.convert()
print("Conversion successful.")

with open(quantized_model_path, 'wb') as f:
    f.write(tflite_quantized_model)

print(f"Successfully saved a more compatible quantized model to: {quantized_model_path}")