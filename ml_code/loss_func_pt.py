import torch
import torch.nn.functional as F
import math
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.tal import bbox2dist
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.models.yolo.detect import DetectionTrainer
   
def NE_IoU_loss(pred_box, gt_box, n=9, eps=1e-7):

    #──────────────────────────────────────────
    # Unpack coordinates
    #──────────────────────────────────────────
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_box.unbind(-1)
    gt_x1, gt_y1, gt_x2, gt_y2   = gt_box.unbind(-1)

    #──────────────────────────────────────────
    # Widths, heights, centers
    #──────────────────────────────────────────
    pred_w  = (pred_x2 - pred_x1).clamp(min=0.0)
    pred_h  = (pred_y2 - pred_y1).clamp(min=0.0)
    pred_cx = (pred_x1 + pred_x2) * 0.5
    pred_cy = (pred_y1 + pred_y2) * 0.5

    gt_w  = (gt_x2 - gt_x1).clamp(min=0.0)
    gt_h  = (gt_y2 - gt_y1).clamp(min=0.0)
    gt_cx = (gt_x1 + gt_x2) * 0.5
    gt_cy = (gt_y1 + gt_y2) * 0.5

    #──────────────────────────────────────────
    # Intersection
    #──────────────────────────────────────────
    inter_w = (torch.minimum(pred_x2, gt_x2) -
                         torch.maximum(pred_x1, gt_x1)).clamp(min=0.0)
    inter_h = (torch.minimum(pred_y2, gt_y2) -
                         torch.maximum(pred_y1, gt_y1)).clamp(min=0.0)
    inter_area = inter_w * inter_h                          # I
    
    #──────────────────────────────────────────
    #  Union
    #──────────────────────────────────────────
    pred_area  = pred_w * pred_h
    gt_area    = gt_w   * gt_h
    union_area = pred_area + gt_area - inter_area           # U

    #──────────────────────────────────────────
    #  Smallest Enclosing Box
    #──────────────────────────────────────────
    enclose_x1 = torch.minimum(pred_x1, gt_x1)
    enclose_y1 = torch.minimum(pred_y1, gt_y1)
    enclose_x2 = torch.maximum(pred_x2, gt_x2)
    enclose_y2 = torch.maximum(pred_y2, gt_y2)

    enclose_w = enclose_x2 - enclose_x1                    # C_w
    enclose_h = enclose_y2 - enclose_y1                    # C_h
    c_sq      = enclose_w ** 2 + enclose_h ** 2            # c² (diagonal²)

    #──────────────────────────────────────────
    #  L_N-IoU
    #──────────────────────────────────────────
    n_iou   = ((1.0 + n) * inter_area) / (union_area + n * inter_area + eps)
    l_n_iou = 1.0 - n_iou

    #──────────────────────────────────────────
    #  L_dis
    #──────────────────────────────────────────
    rho_sq = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2
    l_dis  = rho_sq / (c_sq + eps)

    #──────────────────────────────────────────
    #  L_asp
    #──────────────────────────────────────────
    l_asp = ((pred_w - gt_w) ** 2 / (enclose_w ** 2 + eps) +
             (pred_h - gt_h) ** 2 / (enclose_h ** 2 + eps))

    #──────────────────────────────────────────
    #  Total Regression Loss
    #──────────────────────────────────────────
    loss = l_n_iou + l_dis + l_asp

    return torch.mean(loss)

class neIoU_bbox_loss(BboxLoss):

    def __init__(self, reg_max=16):
        super().__init__(reg_max)
        self.reg_max = reg_max 
        if not hasattr(self, 'dfl_loss'):
            self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def _df_loss(self, pred_dist, target):
        """DFL loss — reimplemented in case parent version removed it."""
        
        pred_dist = pred_dist.view(-1, self.reg_max + 1)
        
        tl = target.long()
        tr = tl + 1
        wl = (tr - target).clamp(0, 1)
        wr = (target - tl).clamp(0, 1)
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tr.shape) * wr
        ).mean(-1, keepdim=True)
    
    def forward(self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz,      
        stride_tensor):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # 👇 Swap in your custom IoU here
        loss_iou = ((NE_IoU_loss(pred_bboxes, target_bboxes)) * weight).sum() / target_scores_sum

        # Keep DFL loss unchanged
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
        loss_dfl = self._df_loss(pred_dist[fg_mask], target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

class ne_IoU_DetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)

        # Get reg_max from the model's Detect head
        detect_head = unwrap_model(model).model[-1]

        # 👇 Replace the default BboxLoss with your custom one
        self.bbox_loss = neIoU_bbox_loss(
            detect_head.reg_max - 1
        ).to(self.device)

        print(f"[ne_IoU_DetectionLoss] bbox_loss on device: {self.device}")

class ne_IoU_Trainer(DetectionTrainer):
    def criterion(self, preds, batch):
        if not hasattr(self, "compute_loss"):
            self.compute_loss = ne_IoU_DetectionLoss(unwrap_model(self.model))
        return self.compute_loss(preds, batch)
