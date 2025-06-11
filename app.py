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
from tensorflow.keras import layers
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
        config = super(AdaptiveFocalLoss, self).get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
        })
        return config


# ================================
# HELPER FUNCTIONS
# ================================
def preprocess_image(image, target_size=(300, 300)):
    """Convert PIL image to a numpy array of shape (1, 300,300,3)."""
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image)
    # If your model was trained on [0,255] images, do not normalize.
    image_array = image_array.astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def get_last_conv_layer(model):
    # Retrieve the name of the last Conv2D layer
    for layer in model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(last_conv_layer_name).output, model.output])
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
    heatmap = tf.maximum(
        heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
    return output


def explain_with_lime(image_np, model, class_names):
    """Generate a LIME explanation for the given image."""
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        # images is a list of np arrays in [0,255]; convert to float32 and run model prediction.
        images = np.array(images).astype(np.float32)
        return model.predict(images)

    # Remove the batch dimension for LIME
    image = image_np.copy()
    explanation = explainer.explain_instance(image,
                                             predict_fn,
                                             top_labels=3,
                                             hide_color=0,
                                             num_samples=1000)
    # Get explanation for the top predicted label
    top_pred = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(top_pred,
                                                positive_only=True,
                                                num_features=5,
                                                hide_rest=False)
    lime_img = mark_boundaries(temp.astype(np.uint8), mask)
    return lime_img


def image_to_base64(image):
    """Convert an OpenCV image (BGR) to a base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    img_bytes = buffer.tobytes()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    return encoded


def pil_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded


# ================================
# LOAD THE SAVED MODEL
# ================================
# Make sure best_model.h5 is in the same directory.
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
    start_time = time.time()
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    try:
        image = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": "Invalid image file"}), 400

    image_array = preprocess_image(image)  # shape: (1,300,300,3)
    predictions = model.predict(image_array)
    pred_index = np.argmax(predictions[0])
    predicted_label = class_names[pred_index]
    confidence = float(predictions[0][pred_index])
    process_time = time.time() - start_time

    # Also encode the original image to base64 (to be re-used in explanation request)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        "predicted_label": predicted_label,
        "confidence": confidence,
        "process_time": process_time,
        "image_base64": img_str
    })


@app.route("/explain", methods=["POST"])
def explain():
    # Expect JSON with field "image_base64"
    data = request.get_json(force=True)
    if "image_base64" not in data:
        return jsonify({"error": "No image data provided"}), 400
    try:
        img_data = base64.b64decode(data["image_base64"])
        image = Image.open(io.BytesIO(img_data))
    except Exception as e:
        return jsonify({"error": "Invalid image data"}), 400

    image_array = preprocess_image(image)  # shape: (1,300,300,3)
    # Get Grad-CAM explanation
    last_conv_layer_name = get_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(image_array, model, last_conv_layer_name)
    # Resize image for overlay
    image_np = np.array(image.resize((300, 300)))
    gradcam_img = overlay_heatmap(image_np, cv2.resize(heatmap, (300, 300)))
    gradcam_b64 = image_to_base64(gradcam_img)

    # Get LIME explanation
    # For LIME we need the original image as numpy array (without batch dimension)
    image_np_for_lime = np.array(image.resize((300, 300)))
    lime_explanation = explain_with_lime(image_np_for_lime, model, class_names)
    # Convert the LIME explanation (RGB) to BGR for OpenCV encoding
    lime_explanation_bgr = cv2.cvtColor(
        (lime_explanation * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    lime_b64 = image_to_base64(lime_explanation_bgr)

    return jsonify({
        "gradcam": gradcam_b64,
        "lime": lime_b64
    })


if __name__ == "__main__":
    # Run in debug mode if desired.
    app.run(host="0.0.0.0", port=5000, debug=True)
