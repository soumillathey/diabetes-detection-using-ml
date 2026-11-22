import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model = joblib.load(os.path.join(BASE_DIR, 'diabetes_model.pkl'))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')
CORS(app)  # Enable CORS

@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)
    
    # Convert data into numpy array and reshape for the model
    input_data = np.array([data['features']])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return the result as JSON
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting server on http://localhost:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)


