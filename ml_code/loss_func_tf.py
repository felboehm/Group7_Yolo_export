import tensorflow as tf

def n_eiou_loss(y_true, y_pred, n=9, num_boxes=None):
    """
    N-EIoU loss for bounding box regression.

    L_N-EIoU = L_N-IoU + L_dis + L_asp

    where:
        L_N-IoU = 1 - [(1 + n) * I] / [U + n * I]
        L_dis   = ρ²(b, b_gt) / c²
        L_asp   = ρ²(w, w_gt) / C_w²  +  ρ²(h, h_gt) / C_h²

    Args:
        y_true: Ground-truth boxes, shape (..., 4) as [x1, y1, x2, y2].
        y_pred: Predicted boxes,    shape (..., 4) as [x1, y1, x2, y2].
        n:      N-IoU hyperparameter (default 9).

    Returns:
        Scalar mean loss over the batch.
    """
    eps = 1e-7
        # ── Handle both flat and structured input ────────────────────────
    if len(y_pred.shape) == 2 or (y_pred.shape.rank == 2):
        # Flattened: [batch, num_boxes * 4]  →  [batch, num_boxes, 4]
        if num_boxes is None:
            raise ValueError(
                "num_boxes is required when inputs are rank-2 "
                f"(got shape {y_pred.shape}). "
                "Pass num_boxes so the tensor can be reshaped to "
                "[batch, num_boxes, 4]."
            )
        y_pred = tf.reshape(y_pred, [-1, num_boxes, 4])
        y_true = tf.reshape(y_true, [-1, num_boxes, 4])

    # At this point shapes are [batch, num_boxes, 4]
    # Every [..., i] indexing below produces [batch, num_boxes]
    
    # ── 1. Unpack coordinates ────────────────────────────────────────
    pred_x1 = y_pred[..., 0]
    pred_y1 = y_pred[..., 1]
    pred_x2 = y_pred[..., 2]
    pred_y2 = y_pred[..., 3]

    gt_x1 = y_true[..., 0]
    gt_y1 = y_true[..., 1]
    gt_x2 = y_true[..., 2]
    gt_y2 = y_true[..., 3]

    # ── 2. Widths, heights, centers ──────────────────────────────────
    pred_w  = tf.maximum(pred_x2 - pred_x1, 0.0)
    pred_h  = tf.maximum(pred_y2 - pred_y1, 0.0)
    pred_cx = (pred_x1 + pred_x2) * 0.5
    pred_cy = (pred_y1 + pred_y2) * 0.5

    gt_w  = tf.maximum(gt_x2 - gt_x1, 0.0)
    gt_h  = tf.maximum(gt_y2 - gt_y1, 0.0)
    gt_cx = (gt_x1 + gt_x2) * 0.5
    gt_cy = (gt_y1 + gt_y2) * 0.5

    # ── 3. Intersection area (I) ─────────────────────────────────────
    inter_w = tf.maximum(tf.minimum(pred_x2, gt_x2) -
                         tf.maximum(pred_x1, gt_x1), 0.0)
    inter_h = tf.maximum(tf.minimum(pred_y2, gt_y2) -
                         tf.maximum(pred_y1, gt_y1), 0.0)
    inter_area = inter_w * inter_h                          # I

    # ── 4. Union area (U) ────────────────────────────────────────────
    pred_area  = pred_w * pred_h
    gt_area    = gt_w   * gt_h
    union_area = pred_area + gt_area - inter_area           # U

    # ── 5. Smallest enclosing box ────────────────────────────────────
    enclose_x1 = tf.minimum(pred_x1, gt_x1)
    enclose_y1 = tf.minimum(pred_y1, gt_y1)
    enclose_x2 = tf.maximum(pred_x2, gt_x2)
    enclose_y2 = tf.maximum(pred_y2, gt_y2)

    enclose_w = enclose_x2 - enclose_x1                    # C_w
    enclose_h = enclose_y2 - enclose_y1                    # C_h
    c_sq      = enclose_w ** 2 + enclose_h ** 2            # c² (diagonal²)

    # ── 6. L_N-IoU ───────────────────────────────────────────────────
    #   N-IoU   = (1 + n) * I  /  (U + n * I)
    #   L_N-IoU = 1 − N-IoU  =  (U − I) / (U + n * I)
    n_iou   = ((1.0 + n) * inter_area) / (union_area + n * inter_area + eps)
    l_n_iou = 1.0 - n_iou

    # ── 7. L_dis  (center distance penalty) ──────────────────────────
    rho_sq = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2
    l_dis  = rho_sq / (c_sq + eps)

    # ── 8. L_asp  (separated aspect-ratio penalty) ───────────────────
    l_asp = ((pred_w - gt_w) ** 2 / (enclose_w ** 2 + eps) +
             (pred_h - gt_h) ** 2 / (enclose_h ** 2 + eps))

    # ── 9. Total loss ────────────────────────────────────────────────
    loss = l_n_iou + l_dis + l_asp

    return tf.reduce_mean(loss)
    
def n_eiou_loss_yolo(y_true, y_pred, n=9, num_boxes=None, conf_weight=1.0):
    """
    N-EIoU loss for YOLO-format bounding boxes.

    Args:
        y_true: Ground-truth boxes (..., 4) as [cx, cy, w, h].
        y_pred: Predicted boxes    (..., 4) as [cx, cy, w, h].
        n:      N-IoU hyperparameter (default 9).

    Returns:
        Scalar mean loss.
    """
    eps = 1e-7
    # ── 0. Split prediction into boxes and confidence ─────────────────
    pred_boxes = y_pred[..., :4]   # [batch, max_boxes, 4]
    pred_conf  = y_pred[...,  4]   # [batch, max_boxes]   (sigmoid applied in model)

    # ── 1. Ensure y_true is [batch, max_boxes, 4] ─────────────────────
    if y_true.shape.rank == 2:
        num_boxes = y_pred.shape[-2] or tf.shape(y_pred)[-2]
        y_true = tf.reshape(y_true, [-1, num_boxes, 4])

    # ── 2. Validity mask  (w > 0 AND h > 0) ──────────────────────────
    valid_mask = tf.cast(
        tf.logical_and(tf.abs(y_true[..., 2]) > eps,
                       tf.abs(y_true[..., 3]) > eps),
        tf.float32
    )   # [batch, max_boxes]: 1.0 = real box, 0.0 = padding

    # ── 3. Unpack ─────────────────────────────────────────────────────
    pred_cx = pred_boxes[..., 0]
    pred_cy = pred_boxes[..., 1]
    pred_w  = tf.abs(pred_boxes[..., 2])
    pred_h  = tf.abs(pred_boxes[..., 3])

    gt_cx = y_true[..., 0]
    gt_cy = y_true[..., 1]
    gt_w  = tf.abs(y_true[..., 2])
    gt_h  = tf.abs(y_true[..., 3])

    # ── 4. Corners ────────────────────────────────────────────────────
    pred_x1, pred_x2 = pred_cx - pred_w/2, pred_cx + pred_w/2
    pred_y1, pred_y2 = pred_cy - pred_h/2, pred_cy + pred_h/2
    gt_x1,   gt_x2   = gt_cx   - gt_w/2,   gt_cx   + gt_w/2
    gt_y1,   gt_y2   = gt_cy   - gt_h/2,   gt_cy   + gt_h/2

    # ── 5. Intersection ───────────────────────────────────────────────
    inter_w    = tf.maximum(tf.minimum(pred_x2, gt_x2) - tf.maximum(pred_x1, gt_x1), 0.0)
    inter_h    = tf.maximum(tf.minimum(pred_y2, gt_y2) - tf.maximum(pred_y1, gt_y1), 0.0)
    inter_area = inter_w * inter_h

    # ── 6. Union ──────────────────────────────────────────────────────
    union_area = pred_w*pred_h + gt_w*gt_h - inter_area

    # ── 7. Enclosing box ──────────────────────────────────────────────
    enclose_w = tf.maximum(pred_x2, gt_x2) - tf.minimum(pred_x1, gt_x1)
    enclose_h = tf.maximum(pred_y2, gt_y2) - tf.minimum(pred_y1, gt_y1)
    c_sq      = enclose_w**2 + enclose_h**2

    # ── 8. N-IoU / distance / aspect losses ───────────────────────────
    #n_iou   = ((1.0 + n) * inter_area) / (union_area + n * inter_area + eps)
    n_iou   = (inter_area + n * inter_area) / (union_area + n * inter_area)
    l_n_iou = 1.0 - n_iou
    l_dis   = ((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2) / (c_sq + eps)
    l_asp   = ((pred_w - gt_w)**2 / (enclose_w**2 + eps) +
               (pred_h - gt_h)**2 / (enclose_h**2 + eps))

    reg_loss = (l_n_iou + l_dis + l_asp) * valid_mask   # zero-out padding

    # ── 9. Confidence loss (applied to ALL boxes) ─────────────────────
    # Target: 1.0 for real boxes, 0.0 for padding
    # Explicit BCE — avoids axis-reduction surprises from keras wrappers
    conf_target = valid_mask                             # [batch, max_boxes]
    conf_loss   = -(
        conf_target       * tf.math.log(      pred_conf + eps) +
        (1.0-conf_target) * tf.math.log(1.0 - pred_conf + eps)
    )                                                    # [batch, max_boxes]

    # ── 10. Reduce ────────────────────────────────────────────────────
    num_valid = tf.maximum(tf.reduce_sum(valid_mask, axis=-1), 1.0)   # [batch]
    num_total = tf.cast(tf.shape(y_pred)[1], tf.float32)               # MAX_BOXES

    reg_per_sample  = tf.reduce_sum(reg_loss,  axis=-1) / num_valid   # avg over real boxes
    conf_per_sample = tf.reduce_sum(conf_loss, axis=-1) / num_total   # avg over all boxes

    return reg_per_sample + conf_weight * conf_per_sample

class NEIoULoss(tf.keras.losses.Loss):
    """Keras loss class for N-EIoU."""

    def __init__(self, n=9, num_boxes=None, name="n_eiou_loss", mode='yolo', **kwargs):
        super().__init__(name=name, **kwargs)
         
        self.n = n
        self.num_boxes = num_boxes
        if mode == 'yolo':
            self.loss = n_eiou_loss_yolo
        else:
            self.loss = n_eiou_loss

    def call(self, y_true, y_pred):
        return self.loss(y_true, y_pred, n=self.n, num_boxes=self.num_boxes)

    def get_config(self):
        config = super().get_config()
        config.update({"n": self.n, "num_boxes": self.num_boxes})
        return config