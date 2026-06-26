"""
Cat vs Dog Image Classifier
-----------------------------
A Flask web app that uses a pre-trained MobileNetV2 model (trained on
ImageNet) to classify uploaded images as either a cat or a dog.

How it works:
1. User uploads an image through the web page.
2. MobileNetV2 predicts what's in the image (out of 1000 possible ImageNet classes).
3. We check if the predicted label matches a list of "cat" or "dog" related
   ImageNet classes, and report the result back to the user.
"""

from flask import Flask, request, render_template, jsonify
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model once when the server starts (not on every request)
print("Loading MobileNetV2 model... (this may take a moment on first run)")
model = MobileNetV2(weights="imagenet")
print("Model loaded successfully!")

# ImageNet has many specific dog breeds and cat breeds rather than
# generic "dog" / "cat" labels. We map relevant keywords to our two classes.
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


def classify_image(img: Image.Image):
    """
    Takes a PIL Image, runs it through MobileNetV2, and returns
    a simplified result: 'cat', 'dog', or 'unknown', plus a confidence score.
    """
    # Preprocess: resize to what MobileNetV2 expects (224x224) and normalize
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Run prediction
    predictions = model.predict(img_array)
    # Get top 5 guesses to improve our odds of matching a cat/dog keyword
    decoded = decode_predictions(predictions, top=5)[0]

    for (_, label, confidence) in decoded:
        label_lower = label.lower().replace("_", " ")
        if any(keyword in label_lower for keyword in DOG_KEYWORDS):
            return "dog", float(confidence), label
        if any(keyword in label_lower for keyword in CAT_KEYWORDS):
            return "cat", float(confidence), label

    # If nothing matched, return the model's best guess anyway
    top_label = decoded[0][1]
    top_confidence = float(decoded[0][2])
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
    # Render sets the PORT environment variable automatically
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

