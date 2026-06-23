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
from model import FastRCNN, train_fast_rcnn_model, get_model_path, STATUS_PATH
from fast_rcnn import run_fast_rcnn_mnist, run_fast_rcnn_resnet50

app = Flask(__name__, template_folder="templates", static_folder="static")

# Configurations
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Flask backend running on device: {device}")

# Lazy-loaded models
digit_model = None
resnet50_model = None
resnet50_categories = None

def get_digit_model():
    global digit_model
    if digit_model is None:
        model_path = get_model_path()
        print(f"[DEBUG] get_digit_model() - model_path resolves to: {model_path}")
        if os.path.exists(model_path):
            print(f"[DEBUG] Loading digit model weights from: {model_path}")
            digit_model = FastRCNN(num_classes=11).to(device)
            digit_model.load_state_dict(torch.load(model_path, map_location=device))
            digit_model.eval()
            print(f"[DEBUG] Digit model loaded successfully.")
        else:
            print(f"[WARNING] Model weights file not found at: {model_path}")
            return None
    return digit_model

def get_resnet50_model():
    global resnet50_model, resnet50_categories
    if resnet50_model is None:
        # Load weights and category metadata
        weights = models.ResNet50_Weights.DEFAULT
        resnet50_categories = weights.meta["categories"]
        resnet50_model = models.resnet50(weights=weights).to(device)
        resnet50_model.eval()
    return resnet50_model, resnet50_categories

# Initialize status if not present
if not os.path.exists(STATUS_PATH):
    from model import update_status
    update_status("idle", 0, "Ready to train custom Fast R-CNN detector.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/train_status", methods=["GET"])
def train_status():
    model_path = get_model_path()
    model_exists = os.path.exists(model_path)
    
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, "r") as f:
            status = json.load(f)
        # If the status file claims completed but the weight file is missing, reset it
        if status["status"] == "completed" and not model_exists:
            status["status"] = "idle"
            status["message"] = "Model weights missing. Please train the model."
    else:
        status = {"status": "idle", "progress": 0, "message": "Ready to train."}
        
    # Regardless of training_status.json, if weights exist and we are not currently training,
    # treat the status as completed/ready so the frontend enables and uses the model.
    if model_exists and status.get("status") != "training":
        status["status"] = "completed"
        status["progress"] = 100
        if status.get("message") in ["Ready to train.", "Ready to train custom Fast R-CNN detector."] or "missing" in status.get("message", "").lower():
            status["message"] = "Model weights found. Ready to use."
        
    status["model_available"] = model_exists
    return jsonify(status)

@app.route("/api/train", methods=["POST"])
def train():
    global digit_model
    
    # Read status to check if already running
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, "r") as f:
            status = json.load(f)
        if status["status"] == "training":
            return jsonify({"error": "Training is already in progress."}), 400
            
    # Clear active digit model to force reload post-training
    digit_model = None
    
    # Start training in background daemon thread
    thread = threading.Thread(target=train_fast_rcnn_model, kwargs={"epochs": 3, "batch_size": 8})
    thread.daemon = True
    thread.start()
    
    return jsonify({"message": "Training started in background."})

@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided"}), 400
        
    model_type = data.get("model_type", "digit")  # "digit" or "resnet50"
    min_conf = float(data.get("min_conf", 0.70))
    
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
        return jsonify({"error": f"Failed to parse image: {str(e)}"}), 400
        
    if model_type == "digit":
        model = get_digit_model()
        if model is None:
            return jsonify({
                "error": "Digit detector model is not trained yet. Please train the model first.",
                "code": "MODEL_NOT_TRAINED"
            }), 400
            
        all_steps, final_detections, heatmap_b64 = run_fast_rcnn_mnist(
            model=model,
            cv_image=img,
            min_conf=min_conf,
            device=device
        )
    elif model_type == "resnet50":
        model, categories = get_resnet50_model()
        all_steps, final_detections, heatmap_b64 = run_fast_rcnn_resnet50(
            model=model,
            categories=categories,
            cv_image=img,
            min_conf=min_conf,
            device=device
        )
    else:
        return jsonify({"error": "Invalid model type"}), 400
        
    return jsonify({
        "all_steps": all_steps,
        "final_detections": final_detections,
        "heatmap": heatmap_b64
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5004, debug=True)
