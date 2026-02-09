import os
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template

# Set TensorFlow to use the legacy Keras API
gf = tf.get_default_graph()

# Initialize Flask App
app = Flask(__name__)

# Load Model Safely
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "vgg.h5")

# Try loading with custom_objects for VGG16 compatibility
try:
    # For TensorFlow 2.16+ with Keras 3
    from keras.models import load_model
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback: try loading with custom objects
    from keras.applications.vgg16 import VGG16
    model = VGG16(weights=None, classes=7, classifier_activation='softmax')

# Class Labels
CLASSES = [
    'Bean',
    'Bitter_Gourd',
    'Bottle_Gourd',
    'Brinjal',
    'Broccoli',
    'Cabbage',
    'Capsicum'
]
# Home Route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')

@app.route('/logout.html')
def logout():
    return render_template('logout.html')


# Prediction Route
# Prediction Route
@app.route('/result', methods=['POST'])
def result():

    if 'image' not in request.files:
        return "No file uploaded"

    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    # Save inside static/uploads
    upload_folder = os.path.join(BASE_DIR, 'static', 'uploads')

    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # file_path = os.path.join(upload_folder, file.filename)
    unique_name = str(uuid.uuid4()) + "_" + file.filename
    file_path = os.path.join(upload_folder, unique_name)
    file.save(file_path)

    # Preprocess image
    img = tf.keras.utils.load_img(file_path, target_size=(150, 150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = CLASSES[predicted_index]

    return render_template(
        'logout.html',
        pred=predicted_label,
        # image_file='uploads/' + file.filename
        image_file='uploads/' + unique_name
    )

# Run App
if __name__ == "__main__":
    app.run(debug=True)