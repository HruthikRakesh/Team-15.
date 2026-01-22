🍇 GrapeCare: Your AI-Powered Grape Health Advisor

GrapeCare is a web application I built to help grape farmers and enthusiasts quickly and accurately diagnose grape leaf diseases. Just by uploading a photo of a leaf, the app uses a deep learning model to identify potential issues and provides clear, actionable advice.

To make it even more helpful, I've integrated an intelligent chatbot powered by Google's Gemini API, which can answer specific questions about grape health in either English or Kannada.

✨ What Makes GrapeCare Special?

*Instant AI Diagnosis: At its core is a powerful MobileNetV2 model that can tell if a leaf is Healthy or suffering from Black Rot, ESCA, or Leaf Blight.

*Clear Visual Results: Instead of just numbers, you get a simple "Health Status" pie chart that shows the disease percentage, making the diagnosis easy to understand at a glance.

*An Expert in Your Pocket: The built-in Gemini AI Assistant is more than just a chatbot. It's trained to be a grape care expert, ready to answer your questions.

*Fully Bilingual: The entire interface, from the analysis results to the AI chatbot's responses, works seamlessly in both English and Kannada (ಕನ್ನಡ).

*Modern & User-Friendly: The app is designed to be simple and intuitive, with a clean look, multiple color themes, and a dark mode.

🛠️ Technology Behind the App

Backend - Python, Flask

AI Model - TensorFlow (Keras), MobileNetV2

Chatbot - Google Gemini API (gemini-1.5-flash-latest)

Frontend - HTML, Tailwind CSS, JavaScript, Chart.js

Deployment - Gunicorn, Render

🚀 Running it on Your Machine
Want to run the project yourself? Here’s how:

1. Get the Code
git clone [https://github.com/your-username/grape-care-app.git](https://github.com/your-username/grape-care-app.git)
cd grape-care-app


2. Set Up the Environment
For Windows:
python -m venv venv
venv\Scripts\activate


3. Install the Libraries
pip install -r requirements.txt


4. Add Your Gemini API Key
Open the app.py file.
Find the line genai.configure(api_key="YOUR_API_KEY").
Paste your own secret API key from Google AI Studio in place of "YOUR_API_KEY".
5. Start the App
python app.py


Now, just open your web browser and go to http://127.0.0.1:5000.

☁️ Free Cloud Deployment

This app is ready to be deployed for free on Render.

Push your final code to a GitHub repository.

On Render, create a new "Web Service" and connect it to your repository.

Use these settings during setup:

Build Command: sh build.sh

Start Command: gunicorn app:app

Snapshots :
![images](https://github.com/HruthikRakesh/Team-15./blob/main/(1).jpeg)
![images](https://github.com/HruthikRakesh/Team-15./blob/main/.jpeg)
[![demo](https://github.com/HruthikRakesh/Team-15./blob/main/2026-01-22%20at%208.54.28%20PM.mp4)


👨‍💻 About the Developer

Hruthik Naidu R
Mothish C
Mohith EC


