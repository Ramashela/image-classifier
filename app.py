"""
Cat vs Dog Image Classifier
-----------------------------
A Flask web app that uses a pre-trained MobileNetV2 model (trained on
ImageNet) to classify uploaded images as either a cat or a dog.

This version uses TensorFlow LITE instead of full TensorFlow. The "lite"
runtime is built for exactly this situation: running a pre-trained model
on a machine with very little RAM (like a free-tier cloud server). Full
TensorFlow needs several hundred MB just to load; tflite-runtime needs a
small fraction of that, which is the difference between fitting inside
Render's free 512MB instance or crashing it.

How it works:
1. On first startup, we download two small files from Google's official
   TensorFlow storage: the MobileNetV2 .tflite model (~14MB) and a text
   file of the 1000 ImageNet class labels. We cache them on disk so this
   only happens once, not on every request.
2. User uploads an image through the web page.
3. We resize/normalize the image and run it through the model.
4. We check the model's predicted label against keyword lists for
   "cat" and "dog" and report the result back to the user.
"""

import os
import io
import urllib.request

import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify

# tflite_runtime is a lightweight package that only contains the inference
# engine, not the rest of TensorFlow. If it's unavailable for some reason,
# fall back to the tflite interpreter bundled inside full tensorflow.
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # pragma: no cover
    from tensorflow.lite import Interpreter  # type: ignore

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model + labels setup (runs once at startup, then cached on disk)
# ---------------------------------------------------------------------------

MODEL_DIR = "model_cache"
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenet_v2.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")

MODEL_URL = "https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1/default/1?lite-format=tflite"
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"

IMG_SIZE = 224


def _download_if_missing(url: str, path: str):
    if not os.path.exists(path):
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
        print(f"Saved {path} ({os.path.getsize(path)} bytes)")


os.makedirs(MODEL_DIR, exist_ok=True)
_download_if_missing(MODEL_URL, MODEL_PATH)
_download_if_missing(LABELS_URL, LABELS_PATH)

print("Loading TFLite model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded successfully!")

with open(LABELS_PATH, "r") as f:
    # File has 1001 lines: index 0 is "background", 1-1000 are real classes
    LABELS = [line.strip() for line in f.readlines()]

# Keywords used to collapse 1000 specific ImageNet classes (breeds, etc.)
# down to a simple cat/dog/unknown verdict.
DOG_KEYWORDS = [
    "retriever", "shepherd", "terrier", "spaniel", "poodle", "bulldog",
    "beagle", "hound", "collie", "husky", "mastiff", "pug", "chihuahua",
    "dalmatian", "dachshund", "rottweiler", "doberman", "corgi", "pointer",
    "setter", "schnauzer", "malamute", "pomeranian", "boxer", "dingo",
    "wolf", "dog", "puppy"
]

CAT_KEYWORDS = [
    "cat", "tabby", "persian", "siamese", "egyptian", "lynx", "cougar",
    "lion", "tiger", "leopard", "jaguar", "kitten", "feline"
]


def preprocess(img: Image.Image) -> np.ndarray:
    """Resize and normalize an image the way MobileNetV2 expects."""
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0  # scale pixels to [-1, 1]
    return np.expand_dims(arr, axis=0)


def classify_image(img: Image.Image):
    """
    Runs the image through the TFLite model and returns a simplified
    result: 'cat', 'dog', or 'unknown', plus a confidence score and the
    raw label the model actually predicted.
    """
    input_data = preprocess(img)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]

    # Softmax-like scores; get the top 5 indices, highest first
    top_indices = np.argsort(output_data)[-5:][::-1]

    for idx in top_indices:
        # Labels file is 0-indexed at "background"; model output index 0
        # also corresponds to "background", so they line up directly.
        label = LABELS[idx] if idx < len(LABELS) else "unknown"
        confidence = float(output_data[idx])
        label_lower = label.lower().replace("_", " ")

        if any(keyword in label_lower for keyword in DOG_KEYWORDS):
            return "dog", confidence, label
        if any(keyword in label_lower for keyword in CAT_KEYWORDS):
            return "cat", confidence, label

    # Nothing matched cat or dog keywords — return best guess anyway
    top_idx = top_indices[0]
    top_label = LABELS[top_idx] if top_idx < len(LABELS) else "unknown"
    top_confidence = float(output_data[top_idx])
    return "unknown", top_confidence, top_label


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    result, confidence, raw_label = classify_image(img)

    return jsonify({
        "result": result,
        "confidence": round(confidence * 100, 2),
        "raw_label": raw_label
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
