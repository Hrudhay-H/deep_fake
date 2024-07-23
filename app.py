from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import os
from PIL import Image
import nbimporter
from main import prepare_image, create_model  # Import functions from your existing script

# Initialize Flask app
app = Flask(__name__)

# Path to upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load pre-trained model
model = create_model()
model.load_weights('model_01.weights.h5')  # Ensure you save and load weights separately

def predict_image(file_path):
    # Prepare image and make prediction
    image = prepare_image(file_path).reshape(-1, 128, 128, 3)
    y_pred = model.predict(image)
    y_pred_class = np.argmax(y_pred, axis=1)[0]

    class_names = ['Fake', 'Real']
    result = {
        'class': class_names[y_pred_class],
        'confidence': float(np.amax(y_pred) * 100)
    }
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction = None
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Make prediction
        prediction = predict_image(file_path)
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
