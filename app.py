from flask import Flask, request, jsonify
import numpy as np
from model.preprocess import preprocess_image
import tensorflow as tf


MODEL_PATH = os.path.join('model', 'model.h5')


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024 # 8 MB upload limit


# Load model once
model = None


def load_model():
global model
if model is None:
model = tf.keras.models.load_model(MODEL_PATH)
return model


@app.route('/')
def index():
return "Breast Cancer Prediction API"


@app.route('/predict', methods=['POST'])
def predict():
if 'image' not in request.files:
return jsonify({'error': 'no image file provided'}), 400


file = request.files['image']
filename = secure_filename(file.filename)
if filename == '':
return jsonify({'error': 'invalid filename'}), 400


# temporarily save
tmp_path = os.path.join('/tmp', filename)
file.save(tmp_path)


try:
img_arr = preprocess_image(tmp_path)
m = load_model()
preds = m.predict(np.expand_dims(img_arr, axis=0))
# Example: model outputs probability of malignant
prob_malignant = float(preds[0][0])
label = 'malignant' if prob_malignant >= 0.5 else 'benign'
return jsonify({'label': label, 'malignant_probability': prob_malignant})
except Exception as e:
return jsonify({'error': str(e)}), 500
finally:
try:
os.remove(tmp_path)
except Exception:
pass


if __name__ == '__main__':
load_model()
app.run(host='0.0.0.0', port=5000, debug=True)