from flask import Flask, jsonify, request
from engine import init_engine, search


init_engine()
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_names = search(image_bytes=img_bytes)
        return jsonify({'names': image_names, 'topk': len(image_names[0]), 'num_image': len(image_names))})


def run_training():
    pass


def load_new_model():
    pass


def get_models():
    pass


def run_offline_extracting():
    pass