# Object Detection Learning Roadmap (Bottom → Top)

## 1. Sliding Window Detector (Classic Approach)
### Goal: Understand detection concept
- Build CNN classifier
- Slide window over image
- Classify each region

### Learn:
- Why brute force detection is slow
- Bounding box idea

---

## 2. Region Proposals (Fast R-CNN Idea)
### Goal: Improve efficiency
- Use selective search proposals
- Run CNN once per image
- Classify proposals

### Learn:
- Shared feature extraction
- Faster inference than sliding window

---

## 3. IoU + Anchors + NMS (Core Concepts)
### Goal: Understand modern detection logic
- Implement IoU
- Implement Anchor boxes
- Implement Non-Max Suppression (NMS)

### Learn:
- How models decide correct boxes
- Overlapping box filtering

---

## 4. Faster R-CNN (PyTorch)
### Goal: Full 2-stage detector
- Use torchvision Faster R-CNN
- Train on VOC or custom dataset
- Visualize proposals

### Learn:
- Region Proposal Network (RPN)
- Accurate but slower detection

---

## 5. YOLO (Single-stage detection)
### Goal: Real-time detection
- Use YOLOv5 / YOLOv8
- Train on custom dataset
- Understand grid-based prediction

### Learn:
- Objectness score
- Fast end-to-end detection

---

## 6. Mini YOLO (From Scratch)
### Goal: Deep understanding
- Simplified YOLO model
- Custom loss (box + class + objectness)
- Train small dataset

### Learn:
- Full detection pipeline internals

---

## 7. Advanced Detection (Transformers)
- DETR
- Deformable DETR
- RT-DETR

### Learn:
- Attention-based detection
- No anchors (modern approach)

---

## 8. Tracking + Segmentation
- Mask R-CNN
- DeepSORT / ByteTrack
- Pose estimation

### Learn:
- Multi-object tracking
- Pixel-level understanding

---

## 9. Production Systems
- ONNX export
- TensorRT optimization
- OpenCV real-time pipelines

### Build:
- Live webcam detector
- Edge device deployment

---

## 10. Advanced Applications
- Smart surveillance system
- Video understanding AI
- Kaggle competitions (COCO)

---

## Final Path Summary
Sliding Window → Fast R-CNN → IoU/NMS → Faster R-CNN → YOLO → Mini YOLO → DETR → Tracking/Segmentation → Production
