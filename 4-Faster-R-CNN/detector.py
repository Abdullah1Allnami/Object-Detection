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

def generate_anchors(grid_size=32, stride=8, sizes=[32], ratios=[0.5, 1.0, 2.0], device="cpu"):
    """
    Generates standard anchor boxes at each spatial location of the backbone feature map.
    """
    anchors = []
    for y in range(grid_size):
        for x in range(grid_size):
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride
            for size in sizes:
                for ratio in ratios:
                    h = size / np.sqrt(ratio)
                    w = size * np.sqrt(ratio)
                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0
                    anchors.append([x1, y1, x2, y2])
    return torch.tensor(anchors, dtype=torch.float32, device=device)

def custom_nms(boxes, scores, iou_threshold):
    """
    Pure PyTorch implementation of Non-Maximum Suppression (NMS).
    """
    device = boxes.device
    if boxes.size(0) == 0:
        return torch.tensor([], dtype=torch.long, device=device)
        
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    
    _, order = scores.sort(0, descending=True)
    keep = []
    
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        i = order[0].item()
        keep.append(i)
        
        xx1 = torch.clamp(x1[order[1:]], min=x1[i])
        yy1 = torch.clamp(y1[order[1:]], min=y1[i])
        xx2 = torch.clamp(x2[order[1:]], max=x2[i])
        yy2 = torch.clamp(y2[order[1:]], max=y2[i])
        
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / union.clamp(min=1e-6)
        
        ids = torch.where(iou < iou_threshold)[0]
        order = order[ids + 1]
        
    return torch.tensor(keep, dtype=torch.long, device=device)

def get_nms_indices(proposals, scores, nms_thresh):
    try:
        import torchvision
        return torchvision.ops.nms(proposals, scores, nms_thresh)
    except:
        return custom_nms(proposals, scores, nms_thresh)

def generate_proposals(anchors, rpn_cls_score, rpn_bbox_pred, img_size=256, pre_nms_top_n=1000, post_nms_top_n=64, nms_thresh=0.7):
    """
    Apply RPN coordinate regression and NMS filtering to find top candidate proposals.
    """
    device = anchors.device
    scores = torch.sigmoid(rpn_cls_score.squeeze(1))
    
    px1, py1, px2, py2 = anchors.unbind(dim=1)
    pw = px2 - px1
    ph = py2 - py1
    px_ctr = px1 + pw / 2.0
    py_ctr = py1 + ph / 2.0
    
    tx, ty, tw, th = rpn_bbox_pred.unbind(dim=1)
    tw = torch.clamp(tw, max=88.0)
    th = torch.clamp(th, max=88.0)
    
    gx_ctr = px_ctr + tx * pw
    gy_ctr = py_ctr + ty * ph
    gw = pw * torch.exp(tw)
    gh = ph * torch.exp(th)
    
    gx1 = torch.clamp(gx_ctr - gw / 2.0, 0, img_size)
    gy1 = torch.clamp(gy_ctr - gh / 2.0, 0, img_size)
    gx2 = torch.clamp(gx_ctr + gw / 2.0, 0, img_size)
    gy2 = torch.clamp(gy_ctr + gh / 2.0, 0, img_size)
    
    proposals = torch.stack([gx1, gy1, gx2, gy2], dim=1)
    
    w = gx2 - gx1
    h = gy2 - gy1
    keep = (w >= 2) & (h >= 2)
    proposals = proposals[keep]
    scores = scores[keep]
    
    if proposals.size(0) == 0:
        return torch.tensor([[0.0, 0.0, 28.0, 28.0]], device=device), torch.tensor([1.0], device=device)
        
    num_to_keep = min(pre_nms_top_n, proposals.size(0))
    scores, order = torch.topk(scores, num_to_keep)
    proposals = proposals[order]
    
    keep_indices = get_nms_indices(proposals, scores, nms_thresh)
    keep_indices = keep_indices[:post_nms_top_n]
    proposals = proposals[keep_indices]
    scores = scores[keep_indices]
    
    return proposals, scores

def decode_bbox_regression(proposals, bbox_offsets):
    """
    Apply bounding box regression offsets to local proposals.
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
    Perform class-agnostic NMS to eliminate multiple overlapping predictions at a single digit site.
    """
    if len(detections) == 0:
        return []
        
    sorted_dets = sorted(detections, key=lambda x: x["score"], reverse=True)
    keep_detections = []
    
    while len(sorted_dets) > 0:
        best_det = sorted_dets.pop(0)
        keep_detections.append(best_det)
        
        boxA = best_det["box"]
        remaining = []
        for det in sorted_dets:
            boxB = det["box"]
            if compute_iou(boxA, boxB) < iou_threshold:
                remaining.append(det)
        sorted_dets = remaining
        
    return keep_detections

def run_faster_rcnn_inference(model, cv_image, min_conf, iou_threshold, device):
    """
    Full Faster R-CNN Inference Pipeline:
    1. Preprocess and normalize image (centering + scaling to 256x256).
    2. Feed to CNN backbone and RPN.
    3. Generate anchor boxes and extract dynamic proposals from RPN predictions.
    4. Pass these proposals to the RoI pooling layer and classification/bbox regressor heads.
    5. Decode bounding box regression adjustments and filter by score & class-agnostic NMS.
    """
    # 1. Image preprocessing
    norm_img, centers = normalize_image_digits(cv_image)
    
    img_tensor = torch.tensor(norm_img, dtype=torch.float32) / 255.0
    img_tensor = (img_tensor - 0.1307) / 0.3081
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        # 2. Extract shared features & predict regions via RPN
        feat_map = model.backbone(img_tensor)
        rpn_cls_scores, rpn_bbox_preds = model.rpn(feat_map)
        
        rpn_cls_score = rpn_cls_scores[0]
        rpn_bbox_pred = rpn_bbox_preds[0]
        
        # 3. Generate Anchors & Proposals
        anchors = generate_anchors(grid_size=32, stride=8, sizes=[32], ratios=[0.5, 1.0, 2.0], device=device)
        
        proposals, rpn_scores = generate_proposals(
            anchors=anchors,
            rpn_cls_score=rpn_cls_score,
            rpn_bbox_pred=rpn_bbox_pred,
            img_size=256,
            pre_nms_top_n=1000,
            post_nms_top_n=64, # Keep top 64 proposals for classification
            nms_thresh=0.7
        )
        
        if proposals.size(0) == 0:
            return norm_img, [], []
            
        # Format proposals as PyTorch tensor [batch_idx, x1, y1, x2, y2]
        batch_col = torch.zeros((proposals.size(0), 1), dtype=torch.float32, device=device)
        rois_tensor = torch.cat([batch_col, proposals], dim=1)
        
        # 4. RoI Pooling and Heads Forward Pass
        pooled_feats = model.roi_pool(feat_map, rois_tensor)
        pooled_feats_flat = pooled_feats.view(pooled_feats.size(0), -1)
        
        cls_logits = model.classifier(pooled_feats_flat)
        bbox_offsets = model.bbox_regressor(pooled_feats_flat)
        probs = F.softmax(cls_logits, dim=1)
        
    # 5. Decode box regression offsets
    proposals_tensor = rois_tensor[:, 1:] # x1, y1, x2, y2
    refined_boxes_tensor = decode_bbox_regression(proposals_tensor, bbox_offsets)
    
    probs_np = probs.cpu().numpy()
    refined_boxes_np = refined_boxes_tensor.cpu().numpy()
    proposals_np = proposals.cpu().numpy()
    rpn_scores_np = rpn_scores.cpu().numpy()
    
    all_proposal_info = []
    detections = []
    
    for i in range(proposals_np.shape[0]):
        prop = proposals_np[i]
        rpn_score = float(rpn_scores_np[i])
        probabilities = probs_np[i]
        pred_class_idx = int(np.argmax(probabilities))
        score = float(probabilities[pred_class_idx])
        
        rx1, ry1, rx2, ry2 = refined_boxes_np[i]
        refined_box = [int(rx1), int(ry1), int(rx2), int(ry2)]
        
        proposal_detail = {
            "original_box": [int(prop[0]), int(prop[1]), int(prop[2]), int(prop[3])],
            "original_xywh": [int(prop[0]), int(prop[1]), int(prop[2] - prop[0]), int(prop[3] - prop[1])],
            "refined_box": refined_box,
            "refined_xywh": [refined_box[0], refined_box[1], refined_box[2] - refined_box[0], refined_box[3] - refined_box[1]],
            "class": pred_class_idx,
            "score": round(score, 4),
            "rpn_score": round(rpn_score, 4),
            "is_background": pred_class_idx == 10
        }
        all_proposal_info.append(proposal_detail)
        
        if pred_class_idx < 10 and score >= min_conf:
            detections.append({
                "box": refined_box,
                "xywh": [refined_box[0], refined_box[1], refined_box[2] - refined_box[0], refined_box[3] - refined_box[1]],
                "class": str(pred_class_idx),
                "score": round(score, 4),
                "original_box": [int(prop[0]), int(prop[1]), int(prop[2]), int(prop[3])],
                "original_xywh": [int(prop[0]), int(prop[1]), int(prop[2] - prop[0]), int(prop[3] - prop[1])],
            })
            
    final_detections = non_max_suppression_class_agnostic(detections, iou_threshold=iou_threshold)
    
    return norm_img, all_proposal_info, final_detections
