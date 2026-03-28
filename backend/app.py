from flask import Flask, request, jsonify, send_from_directory
import pickle
import json
import os
import numpy as np

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# Load model and metrics
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '../model/model.pkl')
metrics_path = os.path.join(base_dir, '../model/metrics.json')

model = None
metrics = {}

def load_data():
    global model, metrics
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load model from {model_path}. Please run model/train.py first. Error: {e}")

    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        print("Metrics loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load metrics. Error: {e}")

# Call load_data on startup
load_data()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        # Try loading again in case it was created after startup
        load_data()
        if not model:
            return jsonify({"error": "Model is not loaded."}), 500
        
    try:
        data = request.json
        # Expecting Features: LB, ASTV, AC, DL, UC
        required_features = ['LB', 'ASTV', 'AC', 'DL', 'UC']
        
        # Extract features in exact order used during training
        features = []
        for feat in required_features:
            if feat not in data:
                return jsonify({"error": f"Missing feature: {feat}"}), 400
            features.append(float(data[feat]))
            
        # Predict
        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        # sklearn Random Forests use index mapping for probabilities based on classes_
        probabilities = model.predict_proba(input_array)[0]
        
        # 1: Normal, 2: Suspect, 3: Pathological
        class_map = l heart rate and variability fall within safe ranges.",
            2.0: "Resu{
            1.0: "Normal",
            2.0: "Suspect",
            3.0: "Pathological"
        }
        
        msg_map = {
            1.0: "Fetalts are slightly abnormal. Monitoring is advised.",
            3.0: "Immediate medical attention recommended due to high risk patterns."
        }
        
        pred_label = class_map.get(prediction, "Unknown")
        confidence = max(probabilities) * 100
        message = msg_map.get(prediction, "")
        
        return jsonify({
            "prediction": pred_label,
            "confidence": f"{confidence:.1f}%",
            "message": message
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/metrics', methods=['GET'])
def get_metrics():
    if not metrics:
        load_data()
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
