import base64
import io
import os

import cv2
import joblib
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "alphabetic_vs_numeric_rf.joblib")
IMG_SIZE = 64

model = joblib.load(MODEL_PATH)


def extract_features_from_array(img_array: np.ndarray) -> np.ndarray:
    """
    Must match the same feature extraction style used in training.
    Expects a grayscale image array.
    """
    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    pixels = img.flatten()
    mean = np.mean(img)
    std = np.std(img)
    horiz = np.sum(img, axis=1)
    vert = np.sum(img, axis=0)

    features = np.concatenate([pixels, [mean, std], horiz, vert])
    return features.reshape(1, -1)


def preprocess_canvas_image(data_url: str) -> np.ndarray:
    """
    Converts base64 canvas image into grayscale numpy array.
    Assumes the canvas is white background with black drawing.
    """
    if "," not in data_url:
        raise ValueError("Invalid image data")

    encoded = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded)

    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = np.array(image)

    return img


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        img = preprocess_canvas_image(image_data)
        features = extract_features_from_array(img)

        pred = model.predict(features)[0]
        probs = model.predict_proba(features)[0]

        labels = ["alphabetic", "numeric"]
        predicted_label = labels[pred]
        confidence = float(np.max(probs)) * 100.0

        return jsonify({
            "prediction": predicted_label,
            "confidence": round(confidence, 2),
            "alphabetic_probability": round(float(probs[0]) * 100.0, 2),
            "numeric_probability": round(float(probs[1]) * 100.0, 2),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)