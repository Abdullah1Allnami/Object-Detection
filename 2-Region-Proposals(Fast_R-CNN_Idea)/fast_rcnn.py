import cv2
import numpy as np
import torch
import torch.nn.functional as F
import base64
from PIL import Image

def selective_search(cv_image, min_size=20, max_size=120):
    """
    Generate candidate bounding boxes (x, y, w, h) using MSER and contour extraction.
    Serves as an efficient, dependency-free region proposal generator.
    """
    h, w = cv_image.shape[:2]
    candidates = set()
    
    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # 1. Use MSER (Maximally Stable Extremal Regions)
    mser = cv2.MSER_create(min_area=min_size*min_size, max_area=max_size*max_size)
    regions, _ = mser.detectRegions(gray)
    for p in regions:
        x, y, ww, hh = cv2.boundingRect(p)
        if min_size <= ww <= max_size and min_size <= hh <= max_size:
            candidates.add((x, y, ww, hh))
            
    # 2. Use Multi-scale thresholding and contours
    thresholds = [50, 100, 150, 200]
    for thresh_val in thresholds:
        # Binary Inverse (for dark objects on bright background)
        _, thresh_inv = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        contours_inv, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_inv:
            x, y, ww, hh = cv2.boundingRect(c)
            if min_size <= ww <= max_size and min_size <= hh <= max_size:
                candidates.add((x, y, ww, hh))
                
        # Binary Normal (for bright objects on dark background)
        _, thresh_norm = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        contours_norm, _ = cv2.findContours(thresh_norm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_norm:
            x, y, ww, hh = cv2.boundingRect(c)
            if min_size <= ww <= max_size and min_size <= hh <= max_size:
                candidates.add((x, y, ww, hh))

    # Convert to list
    candidate_list = list(candidates)
    
    # Filter out highly overlapping duplicates to keep proposal count manageable
    sorted_candidates = sorted(candidate_list, key=lambda box: box[2] * box[3])
    unique_candidates = []
    
    for box in sorted_candidates:
        x, y, ww, hh = box
        boxA = [x, y, x + ww, y + hh]
        
        is_duplicate = False
        for u_box in unique_candidates:
            ux, uy, uww, uhh = u_box
            boxB = [ux, uy, ux + uww, uy + uhh]
            
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
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
        
    return interArea / unionArea

def apply_bbox_regression(box, offsets):
    """
    Apply bounding box regression offsets [dx, dy, dw, dh] to a proposal box [x, y, w, h].
    Returns refined [x_ref, y_ref, w_ref, h_ref].
    """
    x, y, w, h = box
    dx, dy, dw, dh = offsets
    
    # Compute centers
    x_ctr = x + w / 2.0
    y_ctr = y + h / 2.0
    
    # Apply offsets (clip exponential to prevent overflow)
    dw = min(dw, 2.0)
    dh = min(dh, 2.0)
    dw = max(dw, -2.0)
    dh = max(dh, -2.0)
    
    x_ctr_ref = x_ctr + w * dx
    y_ctr_ref = y_ctr + h * dy
    w_ref = w * np.exp(dw)
    h_ref = h * np.exp(dh)
    
    # Convert back to top-left coordinate format
    x_ref = x_ctr_ref - w_ref / 2.0
    y_ref = y_ctr_ref - h_ref / 2.0
    
    return [int(round(x_ref)), int(round(y_ref)), int(round(w_ref)), int(round(h_ref))]

def non_max_suppression(detections, iou_threshold=0.3):
    """
    Perform class-specific NMS on final detections.
    detections: list of dicts: {"box": [x,y,w,h], "refined_box": [x,y,w,h], "class": str, "score": float}
    """
    if len(detections) == 0:
        return []
        
    class_groups = {}
    for det in detections:
        cls = det["class"]
        if cls not in class_groups:
            class_groups[cls] = []
        class_groups[cls].append(det)
        
    keep_detections = []
    
    for cls, group in class_groups.items():
        # Sort by score descending
        group = sorted(group, key=lambda x: x["score"], reverse=True)
        
        while len(group) > 0:
            best_det = group.pop(0)
            keep_detections.append(best_det)
            
            boxA = [
                best_det["refined_box"][0],
                best_det["refined_box"][1],
                best_det["refined_box"][0] + best_det["refined_box"][2],
                best_det["refined_box"][1] + best_det["refined_box"][3]
            ]
            
            remaining = []
            for det in group:
                boxB = [
                    det["refined_box"][0],
                    det["refined_box"][1],
                    det["refined_box"][0] + det["refined_box"][2],
                    det["refined_box"][1] + det["refined_box"][3]
                ]
                
                if compute_iou(boxA, boxB) < iou_threshold:
                    remaining.append(det)
            group = remaining
            
    return keep_detections

def run_fast_rcnn_mnist(model, cv_image, min_conf, device):
    """
    Inference pipeline for MNIST Digit Fast R-CNN.
    """
    # 1. Resize image to 256x256
    img_resized = cv2.resize(cv_image, (256, 256), interpolation=cv2.INTER_AREA)
    h_orig, w_orig = cv_image.shape[:2]
    
    # 2. Extract region proposals using Selective Search on the 256x256 canvas
    proposals = selective_search(img_resized, min_size=20, max_size=100)
    
    if len(proposals) == 0:
        # Fallback to a grid of proposals if selective search yields nothing
        proposals = [(114, 114, 28, 28), (50, 50, 28, 28), (150, 150, 28, 28)]
        
    # 3. Preprocess Image
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    avg_color = np.mean(gray)
    if avg_color > 127:
        gray_processed = 255 - gray  # Invert (MNIST is white digits on black)
    else:
        gray_processed = gray.copy()
        
    img_tensor = torch.tensor(gray_processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = (img_tensor - 0.1307) / 0.3081
    img_tensor = img_tensor.to(device)
    
    # 4. Prepare ROIs tensor (format: [batch_index, x1, y1, x2, y2])
    rois_list = []
    for (x, y, w, h) in proposals:
        rois_list.append([0.0, float(x), float(y), float(x + w), float(y + h)])
    rois_tensor = torch.tensor(rois_list, dtype=torch.float32, device=device)
    
    # 5. Model Inference (One-Pass Backbone + RoI Pooling)
    model.eval()
    with torch.no_grad():
        feat_map, cls_logits, bbox_offsets = model(img_tensor, rois_tensor)
        
        # Softmax for probabilities
        probs = F.softmax(cls_logits, dim=1) # Shape: K x 11
        
    # 6. Parse predictions
    all_steps = []
    detections = []
    
    probs = probs.cpu().numpy()
    bbox_offsets = bbox_offsets.cpu().numpy()
    
    # Extract pooled activations for visualizer (we'll collect them for all ROIs)
    # To show in the UI, we can average the 64 channels of the pooled feature map
    # We can perform the average of pooled features in PyTorch before CPU conversion
    with torch.no_grad():
        # pooled_feats shape: (K, 64, 7, 7)
        pooled_feats = model.roi_pool(feat_map, rois_tensor)
        avg_pooled_feats = torch.mean(pooled_feats, dim=1) # (K, 7, 7)
        avg_pooled_feats = avg_pooled_feats.cpu().numpy()
        
    for i, prop in enumerate(proposals):
        x, y, w, h = prop
        
        # Original scale coords
        x_orig = int(round(x * w_orig / 256.0))
        y_orig = int(round(y * h_orig / 256.0))
        w_orig_sz = int(round(w * w_orig / 256.0))
        h_orig_sz = int(round(h * h_orig / 256.0))
        box_orig = [x_orig, y_orig, w_orig_sz, h_orig_sz]
        
        logits = probs[i]
        class_idx = np.argmax(logits)
        score = float(logits[class_idx])
        
        # Apply bbox regression offsets if class is not background (10)
        offsets = bbox_offsets[i]
        refined_prop = apply_bbox_regression(prop, offsets)
        
        # Scale refined coordinates back to original image scale
        rx, ry, rw, rh = refined_prop
        rx_orig = max(0, int(round(rx * w_orig / 256.0)))
        ry_orig = max(0, int(round(ry * h_orig / 256.0)))
        rw_orig = max(2, int(round(rw * w_orig / 256.0)))
        rh_orig = max(2, int(round(rh * h_orig / 256.0)))
        refined_box_orig = [rx_orig, ry_orig, rw_orig, rh_orig]
        
        class_name = str(class_idx) if class_idx < 10 else "Background"
        is_detection = bool((class_idx < 10) and (score >= min_conf))
        
        # Get 7x7 grid values
        grid_7x7 = avg_pooled_feats[i].tolist()
        grid_flat = [float(val) for row in grid_7x7 for val in row]
        # Normalize grid to [0, 1] for heatmap styling
        max_val = max(grid_flat) if max(grid_flat) > 0 else 1.0
        grid_flat = [val / max_val for val in grid_flat]
        
        step_info = {
            "id": i,
            "box": box_orig,
            "refined_box": refined_box_orig,
            "class": class_name,
            "score": round(score, 4),
            "is_detection": is_detection,
            "offsets": [round(float(o), 4) for o in offsets],
            "grid": grid_flat,
            "logits": [round(float(l), 4) for l in logits]
        }
        all_steps.append(step_info)
        
        if is_detection:
            detections.append({
                "box": box_orig,
                "refined_box": refined_box_orig,
                "class": class_name,
                "score": round(score, 4),
                "offsets": [round(float(o), 4) for o in offsets]
            })
            
    # Apply NMS
    final_detections = non_max_suppression(detections, iou_threshold=0.3)
    
    # 7. Generate shared feature map heatmap for display
    # We average the backbone feature map across all 64 channels
    with torch.no_grad():
        feat_avg = torch.mean(feat_map[0], dim=0) # (32, 32)
        # Normalize to [0, 255]
        feat_min = feat_avg.min()
        feat_max = feat_avg.max()
        if feat_max > feat_min:
            feat_norm = (feat_avg - feat_min) / (feat_max - feat_min) * 255.0
        else:
            feat_norm = feat_avg * 0.0
            
        feat_np = feat_norm.cpu().numpy().astype(np.uint8)
        
        # Upsample heatmap to 256x256 to overlay on input image
        heatmap_resized = cv2.resize(feat_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        # Apply JET colormap for glowing thermal look
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        _, buffer = cv2.imencode('.png', heatmap_color)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        
    return all_steps, final_detections, heatmap_b64

def run_fast_rcnn_resnet50(model, categories, cv_image, min_conf, device):
    """
    Adapted ResNet50 for Once-Per-Image feature extraction and RoI Pooling.
    Since we don't have a pretrained regressor, we disable bounding box refinement, 
    but show full feature extraction, ROI pooling, and classification.
    """
    h_orig, w_orig = cv_image.shape[:2]
    
    # 1. Resize image to 448x448 (gives a 14x14 feature map from ResNet50's stride-32 block)
    img_resized = cv2.resize(cv_image, (448, 448), interpolation=cv2.INTER_LINEAR)
    
    # 2. Extract region proposals using Selective Search
    proposals = selective_search(img_resized, min_size=30, max_size=200)
    
    if len(proposals) == 0:
        proposals = [(150, 150, 120, 120), (50, 100, 120, 120)]
        
    # 3. Preprocess Image for ResNet50
    crop_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    crop_tensor = torch.tensor(crop_rgb, dtype=torch.float32) / 255.0
    crop_tensor = crop_tensor.permute(2, 0, 1) # 3 x 448 x 448
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    crop_tensor = (crop_tensor - mean) / std
    img_batch = crop_tensor.unsqueeze(0).to(device)
    
    # 4. Separate ResNet50 into backbone features and classifier heads
    # ResNet50 layers: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4 -> yields (2048, 14, 14)
    # We define the feature extractor as all layers except avgpool and fc.
    # In torchvision resnet, we can construct this dynamically:
    backbone_layers = [
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3, model.layer4
    ]
    backbone = torch.nn.Sequential(*backbone_layers)
    
    # 5. Extract Feature Map (shape: 1, 2048, 14, 14)
    model.eval()
    with torch.no_grad():
        feat_map = backbone(img_batch) # (1, 2048, 14, 14)
        
    # 6. RoI Pooling (project ROIs by spatial scale 1/32 since 448 / 32 = 14)
    # Output pool size is 7x7
    from model import ROIPool
    roi_pool_layer = ROIPool(output_size=(7, 7), spatial_scale=1.0/32.0)
    
    rois_list = []
    for (x, y, w, h) in proposals:
        rois_list.append([0.0, float(x), float(y), float(x + w), float(y + h)])
    rois_tensor = torch.tensor(rois_list, dtype=torch.float32, device=device)
    
    # Run RoI pooling and final classify
    with torch.no_grad():
        pooled_feats = roi_pool_layer(feat_map, rois_tensor) # (K, 2048, 7, 7)
        
        # Flatten and feed into ResNet's top head
        # In torchvision, standard ResNet fc expects (K, 2048).
        # Standard ResNet uses adaptive avg pooling to 1x1 before FC.
        # We can apply adaptive avg pooling to 1x1 on the 7x7 pooled features!
        pooled_avg = F.adaptive_avg_pool2d(pooled_feats, (1, 1)) # (K, 2048, 1, 1)
        pooled_flat = pooled_avg.view(pooled_avg.size(0), -1) # (K, 2048)
        
        logits = model.fc(pooled_flat) # (K, 1000)
        probs = F.softmax(logits, dim=1) # (K, 1000)
        
        # Get average activations of pooled features (across all 2048 channels) for visualization
        avg_pooled_feats = torch.mean(pooled_feats, dim=1) # (K, 7, 7)
        avg_pooled_feats = avg_pooled_feats.cpu().numpy()
        
    # 7. Parse results
    all_steps = []
    detections = []
    
    probs = probs.cpu().numpy()
    
    for i, prop in enumerate(proposals):
        x, y, w, h = prop
        
        # Scale back to original image dimensions
        x_orig = int(round(x * w_orig / 448.0))
        y_orig = int(round(y * h_orig / 448.0))
        w_orig_sz = int(round(w * w_orig / 448.0))
        h_orig_sz = int(round(h * h_orig / 448.0))
        box_orig = [x_orig, y_orig, w_orig_sz, h_orig_sz]
        
        cls_probs = probs[i]
        class_idx = np.argmax(cls_probs)
        score = float(cls_probs[class_idx])
        
        class_name = categories[class_idx]
        is_detection = bool(score >= min_conf)
        
        # Pre-trained has no regression weights, return [0,0,0,0] offsets
        offsets = [0.0, 0.0, 0.0, 0.0]
        
        grid_7x7 = avg_pooled_feats[i].tolist()
        grid_flat = [float(val) for row in grid_7x7 for val in row]
        max_val = max(grid_flat) if max(grid_flat) > 0 else 1.0
        grid_flat = [val / max_val for val in grid_flat]
        
        # Logits: just extract top 5 classes to save bandwidth
        top_indices = np.argsort(cls_probs)[-5:][::-1]
        top_logits = [{"class": categories[idx], "score": round(float(cls_probs[idx]), 4)} for idx in top_indices]
        
        step_info = {
            "id": i,
            "box": box_orig,
            "refined_box": box_orig, # No regression
            "class": class_name,
            "score": round(score, 4),
            "is_detection": is_detection,
            "offsets": offsets,
            "grid": grid_flat,
            "top_logits": top_logits
        }
        all_steps.append(step_info)
        
        if is_detection:
            detections.append({
                "box": box_orig,
                "refined_box": box_orig,
                "class": class_name,
                "score": round(score, 4),
                "offsets": offsets
            })
            
    # Apply NMS
    final_detections = non_max_suppression(detections, iou_threshold=0.3)
    
    # 8. Generate shared feature map heatmap for display (mean of 2048 channels)
    with torch.no_grad():
        feat_avg = torch.mean(feat_map[0], dim=0) # (14, 14)
        feat_min = feat_avg.min()
        feat_max = feat_avg.max()
        if feat_max > feat_min:
            feat_norm = (feat_avg - feat_min) / (feat_max - feat_min) * 255.0
        else:
            feat_norm = feat_avg * 0.0
            
        feat_np = feat_norm.cpu().numpy().astype(np.uint8)
        heatmap_resized = cv2.resize(feat_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        _, buffer = cv2.imencode('.png', heatmap_color)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        
    return all_steps, final_detections, heatmap_b64
