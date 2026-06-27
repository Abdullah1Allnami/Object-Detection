import cv2
import numpy as np
import torch
import torch.nn.functional as F

def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    Format: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    
    if unionArea == 0.0:
        return 0.0
        
    return interArea / unionArea

def normalize_image_digits(cv_image):
    """
    Find contours of digits, scale them to fit in a 20x20 box,
    center them inside a 28x28 grid, and place them back on a 256x256 canvas.
    This solves the scale mismatch between user drawing and training data.
    """
    h, w = cv_image.shape[:2]
    
    # Ensure grayscale
    if len(cv_image.shape) == 3:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_image.copy()
        
    # Standardize image: white digits on black background
    if np.mean(gray) > 127:
        gray_processed = 255 - gray
    else:
        gray_processed = gray.copy()
        
    # Find contours
    _, thresh = cv2.threshold(gray_processed, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    normalized_canvas = np.zeros_like(gray_processed)
    digit_centers = []
    
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        # Filter out tiny noise, but allow thin digits like '1' (width >= 4, height >= 6)
        if ww >= 4 and hh >= 6:
            digit_crop = gray_processed[y:y+hh, x:x+ww]
            aspect = ww / hh
            if ww > hh:
                n_w = 20
                n_h = int(20 / aspect)
            else:
                n_h = 20
                n_w = int(20 * aspect)
                
            n_w = max(1, n_w)
            n_h = max(1, n_h)
            
            digit_resized = cv2.resize(digit_crop, (n_w, n_h), interpolation=cv2.INTER_AREA)
            
            # Center in standard 28x28 MNIST bounding box
            digit_28 = np.zeros((28, 28), dtype=np.uint8)
            dx = (28 - n_w) // 2
            dy = (28 - n_h) // 2
            digit_28[dy:dy+n_h, dx:dx+n_w] = digit_resized
            
            # Target center coordinates
            cx = x + ww // 2
            cy = y + hh // 2
            
            # Compute top-left corner coordinates on the final canvas
            cx1 = max(0, cx - 14)
            cy1 = max(0, cy - 14)
            cx2 = min(w, cx1 + 28)
            cy2 = min(h, cy1 + 28)
            
            block_w = cx2 - cx1
            block_h = cy2 - cy1
            
            normalized_canvas[cy1:cy2, cx1:cx2] = np.maximum(
                normalized_canvas[cy1:cy2, cx1:cx2],
                digit_28[0:block_h, 0:block_w]
            )
            digit_centers.append((cx, cy))
            
    # Fallback to original processed image if no valid digits found
    if len(digit_centers) == 0:
        return gray_processed, []
        
    return normalized_canvas, digit_centers

def generate_region_proposals_centered(centers, width=256, height=256):
    """
    Generate standard 28x28 region proposals centered around the detected digits.
    Includes slight scales and shifts to allow Fast R-CNN classification & regression heads to align the boxes.
    """
    proposals = []
    
    # If no centers found, return default grid proposals
    if not centers:
        for y in range(30, height - 50, 40):
            for x in range(30, width - 50, 40):
                proposals.append([x, y, x + 40, y + 40])
        return proposals
        
    for cx, cy in centers:
        # Evaluate three scales around standard 28x28 size
        sizes = [24, 28, 32]
        # Evaluate five minor shift displacements
        shifts = [(0, 0), (-3, -3), (3, 3), (-2, 2), (2, -2)]
        for s in sizes:
            half = s // 2
            for dx, dy in shifts:
                ncx = cx + dx
                ncy = cy + dy
                x1 = max(0, ncx - half)
                y1 = max(0, ncy - half)
                x2 = min(width - 1, x1 + s)
                y2 = min(height - 1, y1 + s)
                proposals.append([x1, y1, x2, y2])
                
    # Deduplicate proposals
    unique_proposals = []
    for prop in proposals:
        is_dup = False
        for uprop in unique_proposals:
            if compute_iou(prop, uprop) > 0.90:
                is_dup = True
                break
        if not is_dup:
            unique_proposals.append(prop)
            
    return unique_proposals

def decode_bbox_regression(proposals, bbox_offsets):
    """
    Apply predicted regression offsets to the original region proposals.
    proposals: tensor of shape (N, 4) containing [x1, y1, x2, y2]
    bbox_offsets: tensor of shape (N, 4) containing [tx, ty, tw, th]
    Returns: tensor of shape (N, 4) containing decoded [x1_pred, y1_pred, x2_pred, y2_pred]
    """
    px1 = proposals[:, 0]
    py1 = proposals[:, 1]
    px2 = proposals[:, 2]
    py2 = proposals[:, 3]
    
    pw = px2 - px1
    ph = py2 - py1
    px_ctr = px1 + pw / 2.0
    py_ctr = py1 + ph / 2.0
    
    tx = bbox_offsets[:, 0]
    ty = bbox_offsets[:, 1]
    tw = bbox_offsets[:, 2]
    th = bbox_offsets[:, 3]
    
    x_ctr_pred = px_ctr + tx * pw
    y_ctr_pred = py_ctr + ty * ph
    w_pred = pw * torch.exp(tw)
    h_pred = ph * torch.exp(th)
    
    x1_pred = x_ctr_pred - w_pred / 2.0
    y1_pred = y_ctr_pred - h_pred / 2.0
    x2_pred = x_ctr_pred + w_pred / 2.0
    y2_pred = y_ctr_pred + h_pred / 2.0
    
    # Clip coordinates to image borders [0, 256]
    x1_pred = torch.clamp(x1_pred, 0.0, 256.0)
    y1_pred = torch.clamp(y1_pred, 0.0, 256.0)
    x2_pred = torch.clamp(x2_pred, 0.0, 256.0)
    y2_pred = torch.clamp(y2_pred, 0.0, 256.0)
    
    return torch.stack([x1_pred, y1_pred, x2_pred, y2_pred], dim=1)

def non_max_suppression_class_agnostic(detections, iou_threshold=0.3):
    """
    Perform class-agnostic Non-Maximum Suppression (NMS) on detections.
    Prevents double-detections of different classes on the same digit location.
    """
    if len(detections) == 0:
        return []
        
    # Sort detections by score descending
    sorted_dets = sorted(detections, key=lambda x: x["score"], reverse=True)
    
    keep_detections = []
    
    while len(sorted_dets) > 0:
        best_det = sorted_dets.pop(0)
        keep_detections.append(best_det)
        
        boxA = best_det["box"]
        remaining = []
        for det in sorted_dets:
            boxB = det["box"]
            # Class-agnostic NMS: suppress any box that overlaps significantly, regardless of class
            if compute_iou(boxA, boxB) < iou_threshold:
                remaining.append(det)
        sorted_dets = remaining
        
    return keep_detections

def run_fast_rcnn_inference(model, cv_image, min_conf, iou_threshold, device):
    """
    Perform the entire Fast R-CNN pipeline:
    1. Find drawn digits and scale-normalize them into a 256x256 canvas
    2. Generate region proposals centered on normalized digits
    3. Run Backbone CNN and RoI Pooling forward pass
    4. Decode box regression offsets
    5. Perform class-agnostic NMS to get final digit predictions
    """
    # 1. Normalize image digits (centering + scaling to fit MNIST size)
    norm_img, centers = normalize_image_digits(cv_image)
    
    # Standardize image to torch tensor
    img_tensor = torch.tensor(norm_img, dtype=torch.float32) / 255.0
    img_tensor = (img_tensor - 0.1307) / 0.3081
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device) # Shape: 1 x 1 x 256 x 256
    
    # 2. Generate standard region proposals around digit centers
    proposals = generate_region_proposals_centered(centers)
    
    if len(proposals) == 0:
        return norm_img, [], []
        
    # Format proposals as PyTorch tensor [batch_idx, x1, y1, x2, y2]
    rois_list = []
    for prop in proposals:
        rois_list.append([0.0, float(prop[0]), float(prop[1]), float(prop[2]), float(prop[3])])
    rois_tensor = torch.tensor(rois_list, dtype=torch.float32).to(device)
    
    # 3. Model forward pass
    model.eval()
    with torch.no_grad():
        _, cls_logits, bbox_offsets = model(img_tensor, rois_tensor)
        probs = F.softmax(cls_logits, dim=1)
        
    # 4. Decode bounding boxes using regression offsets
    proposals_tensor = rois_tensor[:, 1:] # x1, y1, x2, y2
    refined_boxes_tensor = decode_bbox_regression(proposals_tensor, bbox_offsets)
    
    # Convert to CPU/Numpy for parsing
    probs_np = probs.cpu().numpy()
    refined_boxes_np = refined_boxes_tensor.cpu().numpy()
    
    all_proposal_info = []
    detections = []
    
    for i, prop in enumerate(proposals):
        probabilities = probs_np[i]
        pred_class_idx = int(np.argmax(probabilities))
        score = float(probabilities[pred_class_idx])
        
        # Bounding box coordinates
        rx1, ry1, rx2, ry2 = refined_boxes_np[i]
        refined_box = [int(rx1), int(ry1), int(rx2), int(ry2)]
        
        proposal_detail = {
            "original_box": prop,
            "original_xywh": [prop[0], prop[1], prop[2] - prop[0], prop[3] - prop[1]],
            "refined_box": refined_box,
            "refined_xywh": [refined_box[0], refined_box[1], refined_box[2] - refined_box[0], refined_box[3] - refined_box[1]],
            "class": pred_class_idx,
            "score": round(score, 4),
            "is_background": pred_class_idx == 10
        }
        all_proposal_info.append(proposal_detail)
        
        # Filter: Pred class is not background (< 10) and score >= threshold
        if pred_class_idx < 10 and score >= min_conf:
            detections.append({
                "box": refined_box,
                "xywh": [refined_box[0], refined_box[1], refined_box[2] - refined_box[0], refined_box[3] - refined_box[1]],
                "class": str(pred_class_idx),
                "score": round(score, 4),
                "original_box": prop,
                "original_xywh": [prop[0], prop[1], prop[2] - prop[0], prop[3] - prop[1]],
            })
            
    # 5. Class-agnostic NMS
    final_detections = non_max_suppression_class_agnostic(detections, iou_threshold=iou_threshold)
    
    return norm_img, all_proposal_info, final_detections
