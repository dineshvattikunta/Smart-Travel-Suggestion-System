from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json
import os

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend connection

# --- CONFIGURATION ---
MODEL_PATH = 'model.pkl'
META_PATH = 'model_meta.json'

# --- LOAD ARTIFACTS ---
print("Initializing Backend...")
model = None
meta_data = {}

if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        with open(META_PATH, 'r') as f:
            meta_data = json.load(f)
        print("✅ Model and Metadata loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
else:
    print("⚠️  WARNING: 'model.pkl' or 'model_meta.json' not found.")
    print("   Please run 'python train_model.py' first.")

# --- ROUTES ---

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({"status": "running", "message": "TravelAI API is active."})

@app.route('/meta', methods=['GET'])
def get_meta():
    """
    Returns the feature metadata so the frontend can dynamically 
    build the input form (dropdowns, number inputs, etc.)
    """
    if not meta_data:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 500
    return jsonify(meta_data)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives JSON data, converts it to a DataFrame, 
    and returns the model's recommendation + confidence score.
    """
    if not model:
        return jsonify({"status": "error", "message": "Model not active"}), 500

    try:
        # 1. Parse Input
        data = request.json
        # print(f"📩 Received prediction request: {data}") # Uncomment to debug incoming data

        # 2. Convert to DataFrame
        # The pipeline expects a DataFrame with specific column names.
        # We create a single-row DataFrame from the input dict.
        df = pd.DataFrame([data])

        # 3. Make Prediction
        prediction = model.predict(df)[0]
        
        # 4. Get Confidence Score
        # predict_proba returns an array of probabilities for each class
        probabilities = model.predict_proba(df)[0]
        confidence = max(probabilities) * 100

        result = {
            "status": "success",
            "prediction": prediction,
            "confidence": round(confidence, 2)
        }
        print(f"✨ Prediction: {prediction} ({confidence:.2f}%)")
        return jsonify(result)

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Run on port 5000
    print("🚀 Server starting on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)