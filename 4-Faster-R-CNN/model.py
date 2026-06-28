import torch
import torch.nn as nn
import torch.nn.functional as F

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
        N = rois.size(0)
        C, H, W = features.size(1), features.size(2), features.size(3)
        out_h, out_w = self.output_size
        
        if N == 0:
            return torch.zeros(0, C, out_h, out_w, device=features.device, dtype=features.dtype)
            
        output = torch.zeros(N, C, out_h, out_w, device=features.device, dtype=features.dtype)
        
        for i in range(N):
            roi = rois[i]
            batch_idx = int(roi[0].item())
            
            x1 = roi[1] * self.spatial_scale
            y1 = roi[2] * self.spatial_scale
            x2 = roi[3] * self.spatial_scale
            y2 = roi[4] * self.spatial_scale
            
            x1_idx = int(torch.round(x1).item())
            y1_idx = int(torch.round(y1).item())
            x2_idx = int(torch.round(x2).item())
            y2_idx = int(torch.round(y2).item())
            
            x1_idx = max(0, min(x1_idx, W - 1))
            y1_idx = max(0, min(y1_idx, H - 1))
            x2_idx = max(0, min(x2_idx, W - 1))
            y2_idx = max(0, min(y2_idx, H - 1))
            
            roi_w = max(1, x2_idx - x1_idx + 1)
            roi_h = max(1, y2_idx - y1_idx + 1)
            
            crop = features[batch_idx, :, y1_idx:y1_idx+roi_h, x1_idx:x1_idx+roi_w]
            
            crop_cpu = crop.cpu().unsqueeze(0)
            pooled_cpu = F.adaptive_max_pool2d(crop_cpu, self.output_size)
            output[i] = pooled_cpu.squeeze(0).to(features.device)
            
        return output

class RPN(nn.Module):
    """
    Region Proposal Network (RPN) for learnable proposal generation.
    """
    def __init__(self, in_channels=64, mid_channels=64, num_anchors=3):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.cls_score = nn.Conv2d(mid_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1)
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        h = F.relu(self.conv(x))
        rpn_cls_scores = self.cls_score(h)
        rpn_bbox_preds = self.bbox_pred(h)
        
        B, _, H, W = x.size()
        rpn_cls_scores = rpn_cls_scores.permute(0, 2, 3, 1).reshape(B, -1, 1)
        rpn_bbox_preds = rpn_bbox_preds.permute(0, 2, 3, 1).reshape(B, -1, 4)
        
        return rpn_cls_scores, rpn_bbox_preds

class FasterRCNN(nn.Module):
    """
    Faster R-CNN Model Architecture (augmented with a Region Proposal Network).
    Downsamples the 256x256 image by a factor of 8 to a 32x32 feature map.
    """
    def __init__(self, num_classes=11, pool_size=(7, 7), spatial_scale=0.125):
        super(FasterRCNN, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x128
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32
        )
        
        self.rpn = RPN(in_channels=64, mid_channels=64, num_anchors=3)
        self.roi_pool = ROIPool(output_size=pool_size, spatial_scale=spatial_scale)
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * pool_size[0] * pool_size[1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )
        
        self.bbox_regressor = nn.Sequential(
            nn.Linear(64 * pool_size[0] * pool_size[1], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)
        )
        
    def forward(self, x, rois):
        feat_map = self.backbone(x)
        pooled_feats = self.roi_pool(feat_map, rois)
        pooled_feats_flat = pooled_feats.view(pooled_feats.size(0), -1)
        
        cls_logits = self.classifier(pooled_feats_flat)
        bbox_offsets = self.bbox_regressor(pooled_feats_flat)
        
        return feat_map, cls_logits, bbox_offsets
