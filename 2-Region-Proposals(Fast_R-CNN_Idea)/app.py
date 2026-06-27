import os
import json
import base64
import time
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template

from model import FastRCNN
from detector import run_fast_rcnn_inference

app = Flask(__name__, template_folder="templates", static_folder="static")

# Configurations
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Fast R-CNN Flask backend running on device: {device}")

WEIGHTS_PATH = os.path.join("checkpoints", "best_model (2).pth")
model = None

def get_model():
    global model
    if model is None:
        if os.path.exists(WEIGHTS_PATH):
            model = FastRCNN(num_classes=11).to(device)
            # Load weights
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
            model.eval()
            print("Successfully loaded Fast R-CNN model weights.")
        else:
            print(f"Warning: Model weights file not found at {WEIGHTS_PATH}")
            return None
    return model

@app.route("/")
def index():
    weights_status = os.path.exists(WEIGHTS_PATH)
    return render_template("index.html", weights_status=weights_status)

@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided"}), 400
        
    min_conf = float(data.get("min_conf", 0.70))
    iou_thresh = float(data.get("iou_threshold", 0.30))
    
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
        
    # Get model
    fast_rcnn_model = get_model()
    if fast_rcnn_model is None:
        return jsonify({
            "error": "Fast R-CNN model weights missing. Please ensure checkpoints/best_model (2).pth is in place.",
            "code": "WEIGHTS_MISSING"
        }), 400
        
    # Run detection pipeline and measure time
    start_time = time.time()
    norm_img, all_proposals, final_detections = run_fast_rcnn_inference(
        model=fast_rcnn_model,
        cv_image=img,
        min_conf=min_conf,
        iou_threshold=iou_thresh,
        device=device
    )
    latency_ms = (time.time() - start_time) * 1000
    
    # Convert normalized image to base64 for visualization
    # The normalized image is grayscale; convert to BGR for PNG output compatibility
    norm_img_color = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode('.png', norm_img_color)
    norm_img_base64 = base64.b64encode(buffer).decode('utf-8')
    normalized_image_url = f"data:image/png;base64,{norm_img_base64}"
    
    # Compare with standard R-CNN (sliding window / crop-based R-CNN) speed:
    # If there are N proposals, standard R-CNN would run N forward passes.
    num_proposals = len(all_proposals)
    # Estimate standard R-CNN latency as N * single forward pass + CPU cropping overhead
    est_rcnn_latency_ms = num_proposals * 3.5 + 40.0
    
    return jsonify({
        "all_proposals": all_proposals,
        "final_detections": final_detections,
        "normalized_image": normalized_image_url,
        "metrics": {
            "num_proposals": num_proposals,
            "fast_rcnn_latency_ms": round(latency_ms, 2),
            "est_rcnn_latency_ms": round(est_rcnn_latency_ms, 2),
            "speedup_ratio": round(est_rcnn_latency_ms / (latency_ms + 1e-5), 1)
        }
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=True)
