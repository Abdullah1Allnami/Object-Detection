import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Define directory for saving weights and status
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "fast_rcnn_mnist.pth")

def get_model_path():
    if os.path.exists(CHECKPOINT_PATH):
        return CHECKPOINT_PATH
    return DEFAULT_MODEL_PATH

STATUS_PATH = os.path.join(BASE_DIR, "training_status.json")

# Try to find existing MNIST data directory from step 1
parent_data_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "1-R-CNN+Sliding-Window(Detectors)", "data"))
if os.path.exists(parent_data_dir):
    DATA_DIR = parent_data_dir
else:
    DATA_DIR = os.path.join(BASE_DIR, "data")

class ROIPool(nn.Module):
    """
    Region of Interest (RoI) Pooling Layer.
    Extracts fixed-size feature maps (e.g., 7x7) from varying-sized region proposals.
    """
    def __init__(self, output_size, spatial_scale):
        super(ROIPool, self).__init__()
        self.output_size = output_size  # (pooled_h, pooled_w)
        self.spatial_scale = spatial_scale

    def forward(self, features, rois):
        """
        features: (B, C, H_feat, W_feat)
        rois: (N, 5) tensor, each row is [batch_idx, x1, y1, x2, y2] on image scale
        """
        N = rois.size(0)
        C, H, W = features.size(1), features.size(2), features.size(3)
        out_h, out_w = self.output_size
        
        if N == 0:
            return torch.zeros(0, C, out_h, out_w, device=features.device, dtype=features.dtype)
            
        output = torch.zeros(N, C, out_h, out_w, device=features.device, dtype=features.dtype)
        
        for i in range(N):
            roi = rois[i]
            batch_idx = int(roi[0].item())
            
            # Project ROI coordinates to the feature map scale
            x1 = roi[1] * self.spatial_scale
            y1 = roi[2] * self.spatial_scale
            x2 = roi[3] * self.spatial_scale
            y2 = roi[4] * self.spatial_scale
            
            # Quantize/round coordinates to nearest integer
            x1_idx = int(torch.round(x1).item())
            y1_idx = int(torch.round(y1).item())
            x2_idx = int(torch.round(x2).item())
            y2_idx = int(torch.round(y2).item())
            
            # Clamp coordinates to feature map boundaries
            x1_idx = max(0, min(x1_idx, W - 1))
            y1_idx = max(0, min(y1_idx, H - 1))
            x2_idx = max(0, min(x2_idx, W - 1))
            y2_idx = max(0, min(y2_idx, H - 1))
            
            # Calculate width and height of the ROI in feature space
            roi_w = max(1, x2_idx - x1_idx + 1)
            roi_h = max(1, y2_idx - y1_idx + 1)
            
            # Crop the corresponding region from features
            crop = features[batch_idx, :, y1_idx:y1_idx+roi_h, x1_idx:x1_idx+roi_w]
            
            # Perform adaptive max pooling on CPU to avoid MPS-specific bugs with small/adaptive-size inputs
            crop_cpu = crop.cpu().unsqueeze(0)  # Shape: (1, C, H_crop, W_crop)
            pooled_cpu = F.adaptive_max_pool2d(crop_cpu, self.output_size)  # Shape: (1, C, out_h, out_w)
            output[i] = pooled_cpu.squeeze(0).to(features.device)
            
        return output

class FastRCNN(nn.Module):
    """
    Fast R-CNN Model Architecture using a ResNet50 backbone.
    Downsamples the 256x256 image by a factor of 32 to an 8x8 feature map.
    """
    def __init__(self, num_classes=11, pool_size=(7, 7), spatial_scale=0.03125):
        super(FastRCNN, self).__init__()
        
        # Load pre-trained ResNet50 backbone
        import torchvision.models as models
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Modify the first conv layer to accept 1 channel (grayscale) instead of 3
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Extract features (exclude average pooling and fully connected head)
        # ResNet50 layer4 output size is (2048, H/32, W/32). For 256x256 image, output is (2048, 8, 8).
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.Conv2d(2048, 128, kernel_size=1)  # 1x1 conv to reduce channels for lightweight RoI pooling
        )
        
        # RoI Pooling Layer
        self.roi_pool = ROIPool(output_size=pool_size, spatial_scale=spatial_scale)
        
        # Sibling Heads
        self.classifier = nn.Sequential(
            nn.Linear(128 * pool_size[0] * pool_size[1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )
        
        self.bbox_regressor = nn.Sequential(
            nn.Linear(128 * pool_size[0] * pool_size[1], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)
        )
        
    def forward(self, x, rois):
        """
        x: (B, 1, 256, 256) full images
        rois: (N, 5) rois [batch_idx, x1, y1, x2, y2]
        """
        feat_map = self.backbone(x)  # (B, 128, 8, 8)
        pooled_feats = self.roi_pool(feat_map, rois)  # (N, 128, 7, 7)
        pooled_feats_flat = pooled_feats.view(pooled_feats.size(0), -1)
        
        cls_logits = self.classifier(pooled_feats_flat)
        bbox_offsets = self.bbox_regressor(pooled_feats_flat)
        
        return feat_map, cls_logits, bbox_offsets

class SyntheticMNISTDataset(torch.utils.data.Dataset):
    """
    Generates synthetic 256x256 images containing 1-3 random MNIST digits.
    This serves as our multi-object dataset for Fast R-CNN object detection.
    """
    def __init__(self, mnist_dataset, num_samples=500, img_size=256):
        self.mnist = mnist_dataset
        self.num_samples = num_samples
        self.img_size = img_size
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        num_digits = np.random.randint(1, 4)
        
        gt_boxes = []
        gt_classes = []
        
        attempts = 0
        placed_digits = 0
        while placed_digits < num_digits and attempts < 100:
            attempts += 1
            mnist_idx = np.random.randint(0, len(self.mnist))
            digit_img, digit_label = self.mnist[mnist_idx]
            
            if not isinstance(digit_img, np.ndarray):
                digit_img = np.array(digit_img, dtype=np.float32)
            
            x = np.random.randint(15, self.img_size - 43)
            y = np.random.randint(15, self.img_size - 43)
            box = [x, y, x + 28, y + 28]
            
            overlap = False
            for placed_box in gt_boxes:
                if self._compute_iou(box, placed_box) > 0.0:
                    overlap = True
                    break
                    
            if not overlap:
                img[y:y+28, x:x+28] = np.maximum(img[y:y+28, x:x+28], digit_img)
                gt_boxes.append(box)
                gt_classes.append(digit_label)
                placed_digits += 1
                
        if placed_digits == 0:
            mnist_idx = np.random.randint(0, len(self.mnist))
            digit_img, digit_label = self.mnist[mnist_idx]
            if not isinstance(digit_img, np.ndarray):
                digit_img = np.array(digit_img, dtype=np.float32)
            x, y = 114, 114
            box = [x, y, x + 28, y + 28]
            img[y:y+28, x:x+28] = digit_img
            gt_boxes.append(box)
            gt_classes.append(digit_label)
            
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, 256, 256)
        img_tensor = (img_tensor - 0.1307) / 0.3081  # Standard MNIST Normalization
        
        return img_tensor, torch.tensor(gt_boxes, dtype=torch.float32), torch.tensor(gt_classes, dtype=torch.long)
        
    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea
        
        if unionArea == 0:
            return 0.0
        return interArea / unionArea

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    gt_boxes = [item[1] for item in batch]
    gt_classes = [item[2] for item in batch]
    return images, gt_boxes, gt_classes

def compute_single_iou(boxA, boxB):
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
    return (interArea / unionArea).item()

def generate_training_rois(gt_boxes, gt_classes, img_size=256, max_pos=16, max_neg=48):
    device = gt_boxes.device
    num_gt = gt_boxes.size(0)
    
    rois = []
    labels = []
    bbox_targets = []
    
    pos_count = 0
    if num_gt > 0:
        for i in range(num_gt):
            rois.append(gt_boxes[i])
            labels.append(gt_classes[i])
            bbox_targets.append(torch.tensor([0.0, 0.0, 0.0, 0.0], device=device))
            pos_count += 1
            
        attempts = 0
        while pos_count < max_pos and attempts < 150:
            attempts += 1
            idx = torch.randint(0, num_gt, (1,)).item()
            gt_box = gt_boxes[idx]
            gt_cls = gt_classes[idx]
            
            x1, y1, x2, y2 = gt_box[0].item(), gt_box[1].item(), gt_box[2].item(), gt_box[3].item()
            w = x2 - x1
            h = y2 - y1
            
            dx_shift = np.random.uniform(-4, 4)
            dy_shift = np.random.uniform(-4, 4)
            dw_scale = np.random.uniform(-0.15, 0.15)
            dh_scale = np.random.uniform(-0.15, 0.15)
            
            px1 = x1 + dx_shift
            py1 = y1 + dy_shift
            pw = w * (1.0 + dw_scale)
            ph = h * (1.0 + dh_scale)
            px2 = px1 + pw
            py2 = py1 + ph
            
            px1 = max(0.0, min(px1, img_size - 2.0))
            py1 = max(0.0, min(py1, img_size - 2.0))
            px2 = max(px1 + 2.0, min(px2, img_size - 1.0))
            py2 = max(py1 + 2.0, min(py2, img_size - 1.0))
            
            prop_box = torch.tensor([px1, py1, px2, py2], device=device)
            iou = compute_single_iou(prop_box, gt_box)
            
            if iou >= 0.5:
                rois.append(prop_box)
                labels.append(gt_cls)
                
                pw_val = px2 - px1
                ph_val = py2 - py1
                px_ctr = px1 + pw_val / 2.0
                py_ctr = py1 + ph_val / 2.0
                
                gt_w = x2 - x1
                gt_h = y2 - y1
                gt_x_ctr = x1 + gt_w / 2.0
                gt_y_ctr = y1 + gt_h / 2.0
                
                tx = (gt_x_ctr - px_ctr) / pw_val
                ty = (gt_y_ctr - py_ctr) / ph_val
                tw = np.log(gt_w / pw_val)
                th = np.log(gt_h / ph_val)
                
                bbox_targets.append(torch.tensor([tx, ty, tw, th], device=device, dtype=torch.float32))
                pos_count += 1
                
    neg_count = 0
    attempts = 0
    while neg_count < max_neg and attempts < 300:
        attempts += 1
        pw = np.random.uniform(20, 60)
        ph = np.random.uniform(20, 60)
        px1 = np.random.uniform(0, img_size - pw - 1)
        py1 = np.random.uniform(0, img_size - ph - 1)
        px2 = px1 + pw
        py2 = py1 + ph
        
        prop_box = torch.tensor([px1, py1, px2, py2], device=device)
        
        max_iou = 0.0
        for i in range(num_gt):
            iou = compute_single_iou(prop_box, gt_boxes[i])
            if iou > max_iou:
                max_iou = iou
                
        if max_iou < 0.2:
            rois.append(prop_box)
            labels.append(torch.tensor(10, device=device))  # Background label
            bbox_targets.append(torch.tensor([0.0, 0.0, 0.0, 0.0], device=device))
            neg_count += 1
            
    if len(rois) == 0:
        rois.append(torch.tensor([0.0, 0.0, 28.0, 28.0], device=device))
        labels.append(torch.tensor(10, device=device))
        bbox_targets.append(torch.tensor([0.0, 0.0, 0.0, 0.0], device=device))
        
    rois = torch.stack(rois)
    labels = torch.stack(labels)
    bbox_targets = torch.stack(bbox_targets)
    
    return rois, labels, bbox_targets

def fast_rcnn_loss(cls_logits, bbox_offsets, labels, bbox_targets):
    loss_cls = F.cross_entropy(cls_logits, labels)
    
    foreground_mask = (labels < 10).float()
    loss_bbox_full = F.smooth_l1_loss(bbox_offsets, bbox_targets, reduction='none')
    loss_bbox_masked = loss_bbox_full.sum(dim=1) * foreground_mask
    
    num_foreground = foreground_mask.sum()
    if num_foreground > 0:
        loss_bbox = loss_bbox_masked.sum() / num_foreground
    else:
        loss_bbox = loss_bbox_masked.sum() * 0.0
        
    total_loss = loss_cls + 1.0 * loss_bbox
    return total_loss, loss_cls, loss_bbox

def compute_metrics(preds, targets, num_classes=11):
    """
    Compute accuracy and macro F1 score in pure python/numpy.
    """
    preds = np.array(preds)
    targets = np.array(targets)
    
    accuracy = np.mean(preds == targets) * 100.0
    
    f1_classes = []
    for c in range(num_classes):
        tp = np.sum((preds == c) & (targets == c))
        fp = np.sum((preds == c) & (targets != c))
        fn = np.sum((preds != c) & (targets == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_classes.append(f1)
        
    macro_f1 = np.mean(f1_classes)
    return round(accuracy, 2), round(macro_f1, 4)

def update_status(status, progress, message, loss=None, accuracy=None, f1=None, test_acc=None, test_f1=None):
    import tempfile
    data = {
        "status": status,      # "idle", "training", "completed", "failed"
        "progress": progress,  # 0 to 100
        "message": message,
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1,
        "test_accuracy": test_acc,
        "test_f1": test_f1
    }
    dir_name = os.path.dirname(STATUS_PATH)
    os.makedirs(dir_name, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tf:
        json.dump(data, tf)
        temp_name = tf.name
    os.replace(temp_name, STATUS_PATH)

def train_fast_rcnn_model(epochs=3, batch_size=8):
    """
    Train ResNet50 Fast R-CNN model on background thread and update training_status.json
    """
    try:
        update_status("training", 0, "Initializing dataset environment...")
        
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Training Fast R-CNN on device: {device}")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        update_status("training", 10, "Loading MNIST dataset splits...")
        
        os.makedirs(DATA_DIR, exist_ok=True)
        # Separate Train and Test datasets
        mnist_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
        
        update_status("training", 15, "Generating train & test synthetic splits...")
        
        train_dataset = SyntheticMNISTDataset(mnist_train, num_samples=600, img_size=256)
        test_dataset = SyntheticMNISTDataset(mnist_test, num_samples=150, img_size=256)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        model = FastRCNN(num_classes=11).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        total_batches = len(train_loader)
        update_status("training", 20, "Fast R-CNN training loop started...")
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            all_preds = []
            all_targets = []
            
            for batch_idx, (images, gt_boxes_list, gt_classes_list) in enumerate(train_loader):
                images = images.to(device)
                
                batch_rois = []
                batch_labels = []
                batch_bbox_targets = []
                
                for b_idx in range(images.size(0)):
                    gt_boxes = gt_boxes_list[b_idx].to(device)
                    gt_classes = gt_classes_list[b_idx].to(device)
                    
                    rois, labels, bbox_targets = generate_training_rois(gt_boxes, gt_classes, img_size=256)
                    
                    batch_col = torch.full((rois.size(0), 1), b_idx, dtype=torch.float32, device=device)
                    rois_with_batch = torch.cat([batch_col, rois], dim=1)
                    
                    batch_rois.append(rois_with_batch)
                    batch_labels.append(labels)
                    batch_bbox_targets.append(bbox_targets)
                    
                batch_rois = torch.cat(batch_rois, dim=0)
                batch_labels = torch.cat(batch_labels, dim=0)
                batch_bbox_targets = torch.cat(batch_bbox_targets, dim=0)
                
                # Forward Pass
                optimizer.zero_grad()
                _, cls_logits, bbox_offsets = model(images, batch_rois)
                
                # Loss
                loss, loss_cls, loss_bbox = fast_rcnn_loss(cls_logits, bbox_offsets, batch_labels, batch_bbox_targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                _, predicted = cls_logits.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch_labels.cpu().numpy())
                
                # Update status
                if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                    batches_completed = epoch * total_batches + batch_idx
                    total_batches_all = epochs * total_batches
                    progress = int((batches_completed / total_batches_all) * 65) + 20
                    
                    cur_loss = running_loss / (batch_idx + 1)
                    cur_acc, cur_f1 = compute_metrics(all_preds, all_targets)
                    msg = f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{total_batches}"
                    
                    update_status(
                        "training", 
                        progress, 
                        msg, 
                        loss=round(cur_loss, 4), 
                        accuracy=cur_acc, 
                        f1=cur_f1
                    )
            
            epoch_loss = running_loss / total_batches
            epoch_acc, epoch_f1 = compute_metrics(all_preds, all_targets)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, F1: {epoch_f1:.4f}")
            
        # TESTING PHASE
        update_status("training", 85, "Training finished. Starting testing evaluation...")
        model.eval()
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for images, gt_boxes_list, gt_classes_list in test_loader:
                images = images.to(device)
                
                batch_rois = []
                batch_labels = []
                for b_idx in range(images.size(0)):
                    gt_boxes = gt_boxes_list[b_idx].to(device)
                    gt_classes = gt_classes_list[b_idx].to(device)
                    rois, labels, _ = generate_training_rois(gt_boxes, gt_classes, img_size=256)
                    batch_col = torch.full((rois.size(0), 1), b_idx, dtype=torch.float32, device=device)
                    batch_rois.append(torch.cat([batch_col, rois], dim=1))
                    batch_labels.append(labels)
                    
                if len(batch_rois) == 0:
                    continue
                batch_rois = torch.cat(batch_rois, dim=0)
                batch_labels = torch.cat(batch_labels, dim=0)
                
                _, cls_logits, _ = model(images, batch_rois)
                _, predicted = cls_logits.max(1)
                test_preds.extend(predicted.cpu().numpy())
                test_targets.extend(batch_labels.cpu().numpy())
                
        test_acc, test_f1 = compute_metrics(test_preds, test_targets)
        print(f"Test Set Evaluation - Accuracy: {test_acc:.2f}%, Macro F1: {test_f1:.4f}")
        
        update_status("training", 95, "Saving trained model parameters...")
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        update_status(
            "completed", 100, "Model trained successfully!", 
            loss=round(epoch_loss, 4), accuracy=epoch_acc, f1=epoch_f1,
            test_acc=test_acc, test_f1=test_f1
        )
        print("Model saved to", CHECKPOINT_PATH)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        update_status("failed", 100, f"Error: {str(e)}")

if __name__ == "__main__":
    # Initialize status and run
    update_status("idle", 0, "Ready to train.")
    train_fast_rcnn_model(epochs=3, batch_size=8)
