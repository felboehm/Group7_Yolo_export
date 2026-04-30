import torch
from types import SimpleNamespace
from .loss_func_pt import ne_IoU_DetectionLoss

def load_model(weights_path: str, device: torch.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Load a pretrained YOLO model.
    Adjust this block to match your YOLO implementation.
    """
    # ── Option A: Ultralytics YOLOv8 ──────────────────────
    from ultralytics import YOLO
    yolo  = YOLO(weights_path)
    model = yolo.model          # underlying nn.Module

    # ── 1. Fix model.args FIRST ────────────────────────────────────────
    if isinstance(model.args, dict):
        model.args = SimpleNamespace(**model.args)

    for attr, default in [("box", 7.5), ("cls", 0.5), ("dfl", 1.5)]:
        if not hasattr(model.args, attr):
            setattr(model.args, attr, default)
    
    # ── 2. Move to device BEFORE creating the criterion ────────────────
    model = model.to(device)
    
    # ── 3. NOW assign custom criterion (model is already on CUDA) ──────
    #    Do NOT call model.init_criterion() after this — it would overwrite it
    model.criterion = ne_IoU_DetectionLoss(model)
    
    # ── 4. Re-enable gradients ─────────────────────────────────────────
    for param in model.parameters():
        param.requires_grad = True
    
    # ── 5. Verify ──────────────────────────────────────────────────────
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params : {trainable:,} / {total:,}")

    if trainable == 0:
        raise RuntimeError("Still no trainable parameters after re-enabling grads!")
    
    model = model.to(device)

    # ── Option B: Custom / other YOLO ─────────────────────
    # from your_module import YOLOModel
    # model = YOLOModel(num_classes=80)
    # model.load_state_dict(torch.load(weights_path, map_location=device))
    # model = model.to(device)

    return model