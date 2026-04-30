import torch

def prepare_batch(batch, device):
    """
    Converts your dataloader batch → Ultralytics loss dict format.

    ── Expected dataloader output ──────────────────────────
    batch["images"] : (B, 3, H, W)  uint8 or float tensor
    batch["labels"] : (N, 6)        [batch_idx, class, cx, cy, w, h]
                       N = total objects across ALL images in the batch
                       bboxes are normalized (0–1), YOLO cx/cy/w/h format

    ── Ultralytics loss expects ─────────────────────────────
    {
        "img"       : (B, 3, H, W)  float32, range 0–1
        "batch_idx" : (N,)          which image each object belongs to
        "cls"       : (N, 1)        class index (float)
        "bboxes"    : (N, 4)        cx, cy, w, h  normalized
    }
    """

    if isinstance(batch, dict):
        images = batch["images"]
        labels = batch["labels"]
    elif isinstance(batch, (tuple, list)):
        images = batch[0]   # images always first
        labels = batch[1]   # labels always second
    else:
        raise TypeError(f"Unexpected batch type: {type(batch)}")
        
    images = images.to(device, non_blocking=True).float()
    #labels = labels.to(device, non_blocking=True)

    # Normalize images if they are uint8 (0–255)
    if images.max() > 1.0:
        images = images / 255.0

    # ── Case A: each tensor is (N, 5) → no batch_idx column ──
        stacked = []
        for batch_idx, label_tensor in enumerate(labels):
            if label_tensor.numel() == 0:
                # image has no objects → skip
                continue
            # label_tensor shape: (N, 5) → [cls, cx, cy, w, h]
            idx_col = torch.full(
                (label_tensor.shape[0], 1),
                fill_value=batch_idx,
                dtype=label_tensor.dtype
            )
            stacked.append(torch.cat([idx_col, label_tensor], dim=1))  # (N, 6)

        if len(stacked) > 0:
            labels = torch.cat(stacked, dim=0).to(device)  # (total_N, 6)
        else:
            # Entire batch has no objects (edge case)
            labels = torch.zeros((0, 6), device=device)

    else:
        # Already a tensor → just send to device
        labels = labels.to(device, non_blocking=True)

    return {
        "img"       : images,
        "batch_idx" : labels[:, 0],        # (N,)
        "cls"       : labels[:, 1:2],      # (N, 1)
        "bboxes"    : labels[:, 2:],       # (N, 4)  cx cy w h
    }