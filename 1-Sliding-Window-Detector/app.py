import os
import json
import base64
import threading
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
import torchvision.models as models

# Import modules from our project
from model import MNIST_ResNet50, train_mnist_model, MODEL_PATH, STATUS_PATH
from detector import run_detection_mnist, run_detection_resnet50

app = Flask(__name__, template_folder="templates", static_folder="static")

# Configurations
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Flask backend running on device: {device}")

# Lazy-loaded models
mnist_model = None
resnet50_model = None
resnet50_categories = None

def get_mnist_model():
    global mnist_model
    if mnist_model is None:
        if os.path.exists(MODEL_PATH):
            mnist_model = MNIST_ResNet50().to(device)
            mnist_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            mnist_model.eval()
        else:
            return None
    return mnist_model

def get_resnet50_model():
    global resnet50_model, resnet50_categories
    if resnet50_model is None:
        weights = models.ResNet50_Weights.DEFAULT
        resnet50_categories = weights.meta["categories"]
        resnet50_model = models.resnet50(weights=weights).to(device)
        resnet50_model.eval()
    return resnet50_model, resnet50_categories

# Initial check for model training status
if not os.path.exists(STATUS_PATH):
    with open(STATUS_PATH, "w") as f:
        json.dump({
            "status": "idle",
            "progress": 0,
            "message": "Ready to train custom CNN."
        }, f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/train_status", methods=["GET"])
def train_status():
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, "r") as f:
            status = json.load(f)
        # Check if the weight file actually exists if state is completed
        if status["status"] == "completed" and not os.path.exists(MODEL_PATH):
            status["status"] = "idle"
            status["message"] = "Model weights missing. Please retrain."
    else:
        status = {"status": "idle", "progress": 0, "message": "Ready to train."}
    
    # Also include whether the model file is available
    status["model_available"] = os.path.exists(MODEL_PATH)
    return jsonify(status)

@app.route("/api/train", methods=["POST"])
def train():
    global mnist_model
    with open(STATUS_PATH, "r") as f:
        status = json.load(f)
        
    if status["status"] == "training":
        return jsonify({"error": "Training is already in progress."}), 400
        
    # Clear the cached model weights so they are reloaded upon next inference
    mnist_model = None
    
    # Start training in a background thread so the request returns immediately
    thread = threading.Thread(target=train_mnist_model, kwargs={"epochs": 2, "batch_size": 128})
    thread.daemon = True
    thread.start()
    
    return jsonify({"message": "Training started in background."})

@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided"}), 400
        
    model_type = data.get("model_type", "digit") # "digit", "mobilenet", or "resnet50"
    stride = int(data.get("stride", 16))
    win_size = int(data.get("window_size", 80))
    min_conf = float(data.get("min_conf", 0.70))
    
    prompt = data.get("prompt", "")
    method = "sliding_window"
    if prompt:
        prompt_lower = prompt.lower()
        if "selective" in prompt_lower:
            method = "selective_search"
            
    # Decode base64 image
    img_data = data["image"]
    if "," in img_data:
        img_data = img_data.split(",")[1]
        
    try:
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse base64 image: {str(e)}"}), 400
        
    window_size = (win_size, win_size)
    
    if model_type == "digit":
        model = get_mnist_model()
        if model is None:
            return jsonify({
                "error": "Digit CNN model is not trained yet. Please train the model first.",
                "code": "MODEL_NOT_TRAINED"
            }), 400
            
        all_steps, final_detections = run_detection_mnist(
            model=model,
            cv_image=img,
            step_size=stride,
            window_size=window_size,
            min_conf=min_conf,
            device=device,
            method=method
        )
    elif model_type in ["mobilenet", "resnet50"]:
        model, categories = get_resnet50_model()
        all_steps, final_detections = run_detection_resnet50(
            model=model,
            categories=categories,
            cv_image=img,
            step_size=stride,
            window_size=window_size,
            min_conf=min_conf,
            device=device,
            method=method
        )
    else:
        return jsonify({"error": "Invalid model type"}), 400
        
    return jsonify({
        "all_steps": all_steps,
        "final_detections": final_detections,
        "method": method
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
