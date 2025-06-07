import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import joblib

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = 64
CATEGORIES = ['plastic', 'paper']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = joblib.load('model.pkl')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.flatten().reshape(1, -1)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Ekstraksi fitur dan prediksi
            features = preprocess_image(filepath)
            pred = model.predict(features)[0]
            prediction = CATEGORIES[pred]
            # Tampilkan feature_importance.png
            feature_importance_path = 'feature_importance.png'
            confusion_matrix_path = 'confusion_matrix.png'
            return render_template('index.html', prediction=prediction, filename=filename, feature_importance=feature_importance_path, confusion_matrix=confusion_matrix_path)
        else:
            return render_template('index.html', error='File type not allowed')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
