# app.py — tumor_detection/app.py

import os
import sys
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from gradcam import generate_gradcam
from report import generate_report

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model(os.path.join('model', 'tumor_model.h5'))

# ── FIXED: 128x128 to match training ─────────────────────────
IMG_SIZE = (128, 128)

# ── Update this if Step 1 output is different order ──────────
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


def prepare_image(img_path):
    img       = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save uploaded file
        filename      = file.filename
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(original_path)

        # Prepare image
        img_array   = prepare_image(original_path)

        # Predict
        predictions     = model.predict(img_array)[0]
        predicted_index = int(np.argmax(predictions))
        predicted_label = CLASS_NAMES[predicted_index]
        confidence      = round(float(predictions[predicted_index]) * 100, 2)

        # Result label
        if predicted_label == 'notumor':
            result_label = 'No Tumor Detected'
        else:
            result_label = f'Tumor Detected — {predicted_label.capitalize()}'

        # Grad-CAM heatmap
        heatmap_filename = 'heatmap_' + filename
        heatmap_path     = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
        heatmap_path     = generate_gradcam(model, img_array, original_path, heatmap_path)

        # Report
        report_text = generate_report(result_label, confidence)

        return jsonify({
            'label'      : result_label,
            'confidence' : confidence,
            'heatmap_url': '/' + heatmap_path.replace('\\', '/'),
            'report'     : report_text
        })

    except Exception as e:
        # This prints the REAL error in your terminal
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)