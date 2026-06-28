import os
import json
import base64
import time
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template

from model import FasterRCNN
from detector import run_faster_rcnn_inference

app = Flask(__name__, template_folder="templates", static_folder="static")

# Configurations
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Faster R-CNN Flask backend running on device: {device}")

WEIGHTS_PATH = os.path.join("checkpoints", "best_model.pth")
model = None
is_mock_weights = False

def get_model():
    global model, is_mock_weights
    if model is None:
        model = FasterRCNN(num_classes=11).to(device)
        if os.path.exists(WEIGHTS_PATH):
            try:
                # Load weights with strict=False to support partial weights or missing RPN keys
                missing_keys, unexpected_keys = model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device), strict=False)
                model.eval()
                is_mock_weights = False
                print("Successfully loaded Faster R-CNN model weights.")
                if missing_keys:
                    print(f"Note: Loaded with missing keys (will be randomly initialized): {missing_keys}")
                if unexpected_keys:
                    print(f"Note: Loaded with unexpected keys (will be ignored): {unexpected_keys}")
            except Exception as e:
                print(f"Error loading state dict from {WEIGHTS_PATH}: {str(e)}")
                print("Initializing with random weights as fallback...")
                is_mock_weights = True
        else:
            print(f"Warning: Model weights file not found at {WEIGHTS_PATH}")
            print("Initializing with random weights for demo compatibility...")
            is_mock_weights = True
            
        model.eval()
    return model

@app.route("/")
def index():
    weights_status = os.path.exists(WEIGHTS_PATH)
    # Ensure model is initialized
    get_model()
    return render_template("index.html", weights_status=weights_status, is_mock=is_mock_weights)

@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided"}), 400
        
    min_conf = float(data.get("min_conf", 0.50))
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
    faster_rcnn_model = get_model()
        
    # Run detection pipeline and measure latency
    start_time = time.time()
    norm_img, all_proposals, final_detections = run_faster_rcnn_inference(
        model=faster_rcnn_model,
        cv_image=img,
        min_conf=min_conf,
        iou_threshold=iou_thresh,
        device=device
    )
    latency_ms = (time.time() - start_time) * 1000
    
    # Convert normalized image to base64 for visualization
    # Grayscale to BGR for PNG display compatibility
    norm_img_color = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode('.png', norm_img_color)
    norm_img_base64 = base64.b64encode(buffer).decode('utf-8')
    normalized_image_url = f"data:image/png;base64,{norm_img_base64}"
    
    # Standard crop-based R-CNN speed simulation comparison:
    num_proposals = len(all_proposals)
    # Estimate standard R-CNN latency: N * single crop forward pass + crop overhead
    est_rcnn_latency_ms = num_proposals * 3.5 + 40.0
    
    # Estimate Fast R-CNN latency (which runs RPN but does not include RPN's overhead)
    # Let's say Fast R-CNN latency is slightly lower than Faster R-CNN because it doesn't run RPN.
    # Faster R-CNN runs both RPN and detector in one single model and saves Selective Search proposal extraction time.
    # In Fast R-CNN, Selective Search takes ~1.5 to 2 seconds. So Faster R-CNN provides a huge speedup over Fast R-CNN!
    ss_latency_ms = 1800.0  # Selective Search average time
    est_fast_rcnn_total_latency_ms = latency_ms + ss_latency_ms
    
    return jsonify({
        "all_proposals": all_proposals,
        "final_detections": final_detections,
        "normalized_image": normalized_image_url,
        "is_mock": is_mock_weights,
        "metrics": {
            "num_proposals": num_proposals,
            "faster_rcnn_latency_ms": round(latency_ms, 2),
            "est_rcnn_latency_ms": round(est_rcnn_latency_ms, 2),
            "est_fast_rcnn_total_latency_ms": round(est_fast_rcnn_total_latency_ms, 2),
            "speedup_ratio": round(est_fast_rcnn_total_latency_ms / (latency_ms + 1e-5), 1)
        }
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5004, debug=True)
