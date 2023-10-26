import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load your trained model
model = load_model('uniform_RN50model_best.h5')

# Specify the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Load and preprocess the uploaded image
        image = Image.open(file_path)
        image = image.resize((224, 224))
        image = np.array(image) / 255.0

        # Make a prediction using your model
        result = model.predict(np.expand_dims(image, axis=0))

        # You can customize the response based on your model's output
        if result[0][0] > result[0][1]:
            prediction = 'Uniform'
        else:
            prediction = 'Not Uniform'

        return jsonify({'prediction': prediction})

    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
