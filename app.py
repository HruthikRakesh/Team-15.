from flask import Flask, render_template, request, jsonify
import tflite_runtime.interpreter as tflite # Using tflite_runtime
import numpy as np
import os
from PIL import Image
import google.generativeai as genai

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'static'

try:
    genai.configure(api_key="AIzaSyAddyA__5xzj7aKPtSUlIDGqLtzNE4dGt4")
except AttributeError:
    print("Please provide your Gemini API key.")

# Load the TFLite model
TFLITE_MODEL_PATH = 'model/grape_model_quantized.tflite'
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# (The rest of your code for disease info, etc., stays exactly the same)
disease_classes = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]
disease_classes_kn = ["ಕಪ್ಪು ಕೊಳೆತ", "ಎಸ್ಕಾ", "ಆರೋಗ್ಯಕರ", "ಎಲೆ ರೋಗ"]
disease_tips = {
    "Black Rot": "Remove infected leaves and apply fungicide early in the season. Improve air circulation.",
    "ESCA": "Prune infected vines and avoid water stress. No chemical cure available.",
    "Healthy": "Your plant looks healthy! Maintain proper nutrition and watering.",
    "Leaf Blight": "Use protective copper-based sprays and remove affected leaves promptly."
}
disease_tips_kn = {
    "Black Rot": "ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಋತುವಿನ ಆರಂಭದಲ್ಲಿ ಶಿಲೀಂಧ್ರನಾಶಕವನ್ನು ಅನ್ವಯಿಸಿ.",
    "ESCA": "ಸೋಂಕಿತ ಬಳ್ಳಿಗಳನ್ನು ಕತ್ತರಿಸಿ ಮತ್ತು ನೀರಿನ ಒತ್ತಡವನ್ನು ತಪ್ಪಿಸಿ. ಯಾವುದೇ ರಾಸಾಯನಿಕ ಚಿಕಿತ್ಸೆ ಲಭ್ಯವಿಲ್ಲ.",
    "Healthy": "ನಿಮ್ಮ ಸಸ್ಯವು ಆರೋಗ್ಯಕರವಾಗಿ ಕಾಣುತ್ತದೆ! ಸರಿಯಾದ ಪೋಷಣೆ ಮತ್ತು ನೀರುಹಾಕುವುದನ್ನು ನಿರ್ವಹಿಸಿ.",
    "Leaf Blight": "ರಕ್ಷಣಾತ್ಮಕ ತಾಮ್ರ ಆಧಾರಿತ ಸ್ಪ್ರೇಗಳನ್ನು ಬಳಸಿ ಮತ್ತು ಪೀಡಿತ ಎಲೆಗಳನ್ನು ತಕ್ಷಣ ತೆಗೆದುಹಾಕಿ."
}

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        filename = f"uploaded_{np.random.randint(10000)}.jpg"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Use the TFLite interpreter for prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        predicted_label_en = disease_classes[predicted_index]

        lang = request.form.get('language', 'en')
        if lang == 'kn':
            predicted_label = disease_classes_kn[predicted_index]
            tip = disease_tips_kn.get(predicted_label_en, "No tips available")
        else:
            predicted_label = predicted_label_en
            tip = disease_tips.get(predicted_label_en, "No tips available")

        return jsonify({
            'status': 'success',
            'prediction': predicted_label, 'confidence': round(confidence, 2),
            'treatment': tip, 'image_url': image_path, 'model_accuracy': 97.22,
            'all_predictions': {
                'labels': disease_classes, 'kannada_labels': disease_classes_kn,
                'probabilities': prediction.tolist()
            }
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# (Your /chat route stays exactly the same)
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.form.get('message')
        lang = request.form.get('language', 'en')
        if not user_message:
            return jsonify({'response': "Please ask a question."})
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        language_instruction = "Your response must be entirely in English."
        if lang == 'kn':
            language_instruction = "Your response must be entirely in clear, natural-sounding Kannada (ಕನ್ನಡ)."
        prompt = f"""
        You are GrapeCare, a helpful AI assistant for grape farmers.
        Your expertise is strictly limited to grape cultivation, common grape diseases (Black Rot, ESCA, Leaf Blight, Healthy), and their treatments.
        {language_instruction}
        Directly answer the user's question. Do not start with generic phrases like "I am a grape assistant."
        If the user asks about something unrelated to grapes, politely refuse in the requested language.
        User question: "{user_message}"
        """
        api_response = gemini_model.generate_content(prompt)
        return jsonify({'response': api_response.text})
    except Exception as e:
        print(f"Error in chat route: {e}")
        return jsonify({'response': "Sorry, I'm having trouble connecting right now. Please try again later."}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)