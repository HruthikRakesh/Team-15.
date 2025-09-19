import tensorflow as tf
import os

# Path to your original Keras model (.keras file)
original_model_path = 'model/best_model.keras'

# Path where the new, smaller model will be saved
quantized_model_path = 'model/grape_model_quantized.tflite'

# Create the 'model' directory if it doesn't exist
os.makedirs(os.path.dirname(quantized_model_path), exist_ok=True)

print(f"Loading model from: {original_model_path}")

# Load your original Keras model
try:
    model = tf.keras.models.load_model(original_model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize the TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimization (quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
print("Converting model to TensorFlow Lite format with quantization...")
tflite_quantized_model = converter.convert()
print("Conversion successful.")

# Save the new .tflite model to a file
with open(quantized_model_path, 'wb') as f:
    f.write(tflite_quantized_model)

print(f"Successfully saved quantized model to: {quantized_model_path}")