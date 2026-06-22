import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def sliding_window(image_width, image_height, step_size, window_size):
    """
    Generate window coordinates (x, y, w, h) across the image.
    """
    win_w, win_h = window_size
    for y in range(0, image_height - win_h + 1, step_size):
        for x in range(0, image_width - win_w + 1, step_size):
            yield (x, y, win_w, win_h)

def selective_search(cv_image, min_size=30, max_size=200):
    """
    Generate candidate bounding boxes (x, y, w, h) using OpenCV MSER and multi-scale contours.
    This serves as a high-quality, dependency-free selective search/region proposal generator.
    """
    h, w = cv_image.shape[:2]
    candidates = set()
    
    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # 1. Use MSER (Maximally Stable Extremal Regions)
    mser = cv2.MSER_create(min_area=min_size*min_size, max_area=max_size*max_size)
    regions, _ = mser.detectRegions(gray)
    for p in regions:
        # Get bounding box
        x, y, ww, hh = cv2.boundingRect(p)
        if min_size <= ww <= max_size and min_size <= hh <= max_size:
            candidates.add((x, y, ww, hh))
            
    # 2. Use Multi-scale thresholding and contours
    thresholds = [50, 100, 150, 200]
    for thresh_val in thresholds:
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            if min_size <= ww <= max_size and min_size <= hh <= max_size:
                candidates.add((x, y, ww, hh))
                
        _, thresh_normal = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        contours_normal, _ = cv2.findContours(thresh_normal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_normal:
            x, y, ww, hh = cv2.boundingRect(c)
            if min_size <= ww <= max_size and min_size <= hh <= max_size:
                candidates.add((x, y, ww, hh))

    # 3. Filter out redundant/highly overlapping boxes using NMS/IoU
    sorted_candidates = sorted(list(candidates), key=lambda box: box[2] * box[3])
    
    unique_candidates = []
    for box in sorted_candidates:
        x, y, ww, hh = box
        boxA = [x, y, x + ww, y + hh]
        
        is_duplicate = False
        for u_box in unique_candidates:
            ux, uy, uww, uhh = u_box
            boxB = [ux, uy, ux + uww, uy + uhh]
            
            # If IoU is very high, skip it
            if compute_iou(boxA, boxB) > 0.85:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_candidates.append(box)
            
    return unique_candidates

def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    Format: [x1, y1, x2, y2]
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of intersection rectangle
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    
    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Compute the union area
    unionArea = boxAArea + boxBArea - interArea
    
    if unionArea == 0:
        return 0.0
        
    return interArea / unionArea

def non_max_suppression(detections, iou_threshold=0.3):
    """
    Perform Non-Maximum Suppression (NMS) on detections.
    detections format: list of dicts with keys:
      {"box": [x, y, w, h], "class": str, "score": float}
    """
    if len(detections) == 0:
        return []
        
    # Group detections by class (per-class NMS)
    class_groups = {}
    for det in detections:
        cls = det["class"]
        if cls not in class_groups:
            class_groups[cls] = []
        class_groups[cls].append(det)
        
    keep_detections = []
    
    for cls, group in class_groups.items():
        # Sort group by score descending
        group = sorted(group, key=lambda x: x["score"], reverse=True)
        
        while len(group) > 0:
            # Keep the highest scoring detection
            best_det = group.pop(0)
            keep_detections.append(best_det)
            
            # Compare remaining boxes in group with best_det
            boxA = [
                best_det["box"][0],
                best_det["box"][1],
                best_det["box"][0] + best_det["box"][2],
                best_det["box"][1] + best_det["box"][3]
            ]
            
            remaining = []
            for det in group:
                boxB = [
                    det["box"][0],
                    det["box"][1],
                    det["box"][0] + det["box"][2],
                    det["box"][1] + det["box"][3]
                ]
                # If IoU is below threshold, keep the box for future evaluation
                if compute_iou(boxA, boxB) < iou_threshold:
                    remaining.append(det)
            group = remaining
            
    return keep_detections

def run_detection_mnist(model, cv_image, step_size, window_size, min_conf, device, method="sliding_window"):
    """
    Run detection using the custom MNIST digit model with batched inference.
    """
    h, w = cv_image.shape[:2]
    win_w, win_h = window_size
    
    # Convert image to grayscale for MNIST
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # In MNIST, digits are white on a black background.
    # Our canvas will draw black on white, so we invert or normalize.
    avg_color = np.mean(gray)
    if avg_color > 127:
        gray_processed = 255 - gray
    else:
        gray_processed = gray.copy()
        
    all_steps = []
    detections = []
    
    model.eval()
    
    # Generate windows
    if method == "selective_search":
        windows = selective_search(cv_image, min_size=30, max_size=200)
    else:
        windows = list(sliding_window(w, h, step_size, window_size))
    
    # Identify which windows need model inference
    active_crops = [] # list of (idx, crop_resized)
    
    for idx, (x, y, ww, hh) in enumerate(windows):
        crop = gray_processed[y:y+hh, x:x+ww]
        
        # Check active pixel percentage
        _, thresh = cv2.threshold(crop, 30, 255, cv2.THRESH_BINARY)
        active_pct = np.sum(thresh == 255) / (ww * hh)
        
        if active_pct < 0.02:
            all_steps.append({
                "box": [x, y, ww, hh],
                "class": "Background",
                "score": 1.0,
                "is_detection": False
            })
        else:
            all_steps.append(None)
            
            # Preprocess crop for MNIST: resize to 28x28
            crop_resized = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
            active_crops.append((idx, crop_resized))
            
    # Run batched model inference
    if len(active_crops) > 0:
        crops_tensors = []
        for _, crop_resized in active_crops:
            crop_tensor = torch.tensor(crop_resized, dtype=torch.float32) / 255.0
            crop_tensor = (crop_tensor - 0.1307) / 0.3081
            crops_tensors.append(crop_tensor.unsqueeze(0)) # 1 x 28 x 28
            
        crops_batch = torch.stack(crops_tensors).to(device) # Shape: K x 1 x 28 x 28
        
        # Run inference in batches of 128
        batch_size = 128
        all_probs = []
        with torch.no_grad():
            for i in range(0, crops_batch.size(0), batch_size):
                batch = crops_batch[i:i+batch_size]
                outputs = model(batch)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs)
        all_probs = torch.cat(all_probs, dim=0) # Shape: K x 10
        
        # Map predictions back to windows
        for i, (idx, _) in enumerate(active_crops):
            x, y, ww, hh = windows[idx]
            probabilities = all_probs[i]
            conf, pred_class_idx = torch.max(probabilities, dim=0)
            score = conf.item()
            pred_class = str(pred_class_idx.item())
            
            is_detection = score >= min_conf
            
            step_info = {
                "box": [x, y, ww, hh],
                "class": pred_class,
                "score": round(score, 4),
                "is_detection": is_detection
            }
            all_steps[idx] = step_info
            
            if is_detection:
                detections.append({
                    "box": [x, y, ww, hh],
                    "class": pred_class,
                    "score": round(score, 4)
                })
                
    # Run Non-Maximum Suppression to filter overlapping boxes
    final_detections = non_max_suppression(detections, iou_threshold=0.3)
    
    return all_steps, final_detections

def run_detection_resnet50(model, categories, cv_image, step_size, window_size, min_conf, device, method="sliding_window"):
    """
    Run detection using pre-trained ResNet50 with batched inference.
    """
    h, w = cv_image.shape[:2]
    win_w, win_h = window_size
    
    all_steps = []
    detections = []
    
    model.eval()
    
    # Generate windows
    if method == "selective_search":
        windows = selective_search(cv_image, min_size=30, max_size=200)
    else:
        windows = list(sliding_window(w, h, step_size, window_size))
    
    # Preprocess all crops
    crops_tensors = []
    for (x, y, ww, hh) in windows:
        crop = cv_image[y:y+hh, x:x+ww]
        
        # Preprocess crop: resize to 224x224, convert to RGB, normalize
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        crop_tensor = torch.tensor(crop_resized, dtype=torch.float32) / 255.0
        # Permute to channels-first (3 x 224 x 224)
        crop_tensor = crop_tensor.permute(2, 0, 1)
        
        # Mean and std for ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        crop_tensor = (crop_tensor - mean) / std
        crops_tensors.append(crop_tensor)
        
    if len(crops_tensors) > 0:
        crops_batch = torch.stack(crops_tensors).to(device) # Shape: W x 3 x 224 x 224
        
        # Run inference in batches of 64
        batch_size = 64
        all_probs = []
        with torch.no_grad():
            for i in range(0, crops_batch.size(0), batch_size):
                batch = crops_batch[i:i+batch_size]
                outputs = model(batch)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs)
        all_probs = torch.cat(all_probs, dim=0) # Shape: W x 1000
        
        # Map predictions back
        for idx, (x, y, ww, hh) in enumerate(windows):
            probabilities = all_probs[idx]
            conf, pred_class_idx = torch.max(probabilities, dim=0)
            score = conf.item()
            pred_class = categories[pred_class_idx.item()]
            
            is_detection = score >= min_conf
            
            step_info = {
                "box": [x, y, ww, hh],
                "class": pred_class,
                "score": round(score, 4),
                "is_detection": is_detection
            }
            all_steps.append(step_info)
            
            if is_detection:
                detections.append({
                    "box": [x, y, ww, hh],
                    "class": pred_class,
                    "score": round(score, 4)
                })
                
    # Run Non-Maximum Suppression to filter overlapping boxes
    final_detections = non_max_suppression(detections, iou_threshold=0.3)
    
    return all_steps, final_detections
