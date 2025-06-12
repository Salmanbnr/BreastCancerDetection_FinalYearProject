# app.py
import os
import io
import time
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ================================
# GPU CONFIGURATION
# ================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs found and memory growth enabled.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU.")

# ================================
# CUSTOM LOSS FUNCTION
# ================================
class AdaptiveFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super(AdaptiveFocalLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return tf.reduce_mean(focal_loss)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config

# ================================
# HELPER FUNCTIONS
# ================================
def preprocess_image(image, target_size=(300, 300)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image).astype(np.float32)
    return np.expand_dims(image_array, axis=0)

def get_last_conv_layer(model):
    for layer in model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
    return output

def explain_with_lime(image_np, model):
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        images = np.array(images).astype(np.float32)
        return model.predict(images)

    explanation = explainer.explain_instance(image_np, predict_fn, top_labels=1, hide_color=0, num_samples=1000)
    top_pred = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(top_pred, positive_only=True, num_features=5, hide_rest=False)
    return mask.astype(np.uint8)

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def pil_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ================================
# LOAD MODEL
# ================================
custom_objects = {"AdaptiveFocalLoss": AdaptiveFocalLoss}
model = load_model("best_model.h5", custom_objects=custom_objects)
class_names = ["benign", "malignant", "normal"]

# ================================
# FLASK APP
# ================================
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    try:
        image = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    image_array = preprocess_image(image)
    predictions = model.predict(image_array)
    pred_index = np.argmax(predictions[0])
    predicted_label = class_names[pred_index]
    confidence = float(predictions[0][pred_index])

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        "predicted_label": predicted_label,
        "confidence": confidence,
        "image_base64": img_str
    })

@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json(force=True)
    if "image_base64" not in data:
        return jsonify({"error": "No image data provided"}), 400
    try:
        img_data = base64.b64decode(data["image_base64"])
        image = Image.open(io.BytesIO(img_data))
    except Exception:
        return jsonify({"error": "Invalid image data"}), 400

    image_np = np.array(image.resize((300, 300)))
    image_array = np.expand_dims(image_np.astype(np.float32), axis=0)

    # Grad-CAM
    last_conv_layer_name = get_last_conv_layer(model)
    gradcam_heatmap = make_gradcam_heatmap(image_array, model, last_conv_layer_name)
    gradcam_binary = (cv2.resize(gradcam_heatmap, (300, 300)) > 0.4).astype(np.uint8)

    # LIME
    lime_mask = explain_with_lime(image_np, model)
    lime_binary = (lime_mask > 0).astype(np.uint8)

    # Combine - Highlight overlapping areas (URLAB)
    urlab_mask = cv2.bitwise_and(lime_binary, gradcam_binary)

    # Color overlay for all
    lime_colored = cv2.applyColorMap((lime_binary * 255).astype(np.uint8), cv2.COLORMAP_WINTER)
    gradcam_colored = cv2.applyColorMap((gradcam_binary * 255).astype(np.uint8), cv2.COLORMAP_JET)
    urlab_colored = cv2.applyColorMap((urlab_mask * 255).astype(np.uint8), cv2.COLORMAP_HOT)

    overlay_base = image_np.copy()
    lime_result = cv2.addWeighted(overlay_base, 0.6, lime_colored, 0.4, 0)
    gradcam_result = cv2.addWeighted(overlay_base, 0.6, gradcam_colored, 0.4, 0)
    urlab_result = cv2.addWeighted(overlay_base, 0.5, urlab_colored, 0.5, 0)

    return jsonify({
        "lime": image_to_base64(lime_result),
        "gradcam": image_to_base64(gradcam_result),
        "urlab": image_to_base64(urlab_result)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
