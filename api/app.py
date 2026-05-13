import base64
import os
import sys
import tempfile

import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
from gradcam import generate_gradcam
from report import generate_report

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, 'templates'),
    static_folder=os.path.join(ROOT_DIR, 'static')
)

MODEL_PATH = os.path.join(ROOT_DIR, 'model', 'tumor_model.h5')
model = load_model(MODEL_PATH)

IMG_SIZE = (128, 128)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


def prepare_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
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

        temp_dir = tempfile.mkdtemp()
        filename = file.filename
        original_path = os.path.join(temp_dir, filename)
        file.save(original_path)

        img_array = prepare_image(original_path)

        predictions = model.predict(img_array)[0]
        predicted_index = int(np.argmax(predictions))
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = round(float(predictions[predicted_index]) * 100, 2)

        if predicted_label == 'notumor':
            result_label = 'No Tumor Detected'
        else:
            result_label = f'Tumor Detected — {predicted_label.capitalize()}'

        heatmap_path = os.path.join(temp_dir, f'heatmap_{filename}')
        heatmap_path = generate_gradcam(model, img_array, original_path, heatmap_path)

        with open(heatmap_path, 'rb') as f:
            heatmap_bytes = f.read()
        heatmap_b64 = 'data:image/png;base64,' + base64.b64encode(heatmap_bytes).decode('utf-8')

        report_text = generate_report(result_label, confidence)

        return jsonify({
            'label': result_label,
            'confidence': confidence,
            'heatmap_url': heatmap_b64,
            'report': report_text
        })

    except Exception as e:
        print(f'ERROR: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
