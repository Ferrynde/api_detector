# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Gestion manuelle de CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

model = load_model('fraud_detection_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':'no file'}), 400
    f = request.files['file']
    filepath = os.path.join('uploads', f.filename)
    f.save(filepath)
    
    # CORRECTION: 128x128 au lieu de 224x224
    img = image.load_img(filepath, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    pred = model.predict(x)[0]
    os.remove(filepath)
    return jsonify({'prediction': float(pred[0]), 'fraud': bool(pred[0]>0.5)})

@app.route('/predict', methods=['OPTIONS'])
def predict_options():
    return '', 200
    
if __name__=='__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)