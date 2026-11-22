import os
from pathlib import Path
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

# Check both api/ and root directory for diabetes_model.pkl
model_path = current_dir / 'diabetes_model.pkl'
if not model_path.exists():
    model_path = root_dir / 'diabetes_model.pkl'

model = joblib.load(str(model_path))

app = Flask(__name__, static_folder=str(root_dir), static_url_path='')
CORS(app)

@app.route('/')
def home():
    return send_from_directory(str(root_dir), 'index.html')

@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array([data['features']])
    prediction = int(model.predict(input_data)[0])
    
    # Calculate probabilities
    probabilities = model.predict_proba(input_data)[0]
    diabetes_prob = round(float(probabilities[1]) * 100, 1)
    
    if diabetes_prob >= 60:
        risk_level = "High Risk"
    elif diabetes_prob >= 35:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    return jsonify({
        'prediction': prediction,
        'probability': diabetes_prob,
        'risk_level': risk_level
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
