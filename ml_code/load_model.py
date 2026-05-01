import torch
import torch.nn as nn
from types import SimpleNamespace
from ultralytics.nn.modules.head import Detect
from .loss_func_pt import ne_IoU_DetectionLoss

def load_model(weights_path: str, device: torch.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Load a pretrained YOLO model.
    Adjust this block to match your YOLO implementation.
    """
    # ── Option A: Ultralytics YOLOv8 ──────────────────────
    from ultralytics import YOLO
    yolo  = YOLO(weights_path)

    yolo = modify_yolo(yolo)
    
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

    return model, yolo 


def modify_yolo(model, new_nc=11, class_names={0: "Pedestrian", 1: "People", 2: "Bicycle", 3: "Car", 4: "Van", 5: "Truck", 6: "Tricycle", 7: "Awning-tricycle", 8: "Bus", 9: "Motor", 10: "Other"}):
    """Modify in-place and return the YOLO wrapper, not just the bare model."""
    yolo = model
    detect = yolo.model.model[-1]

    # Swap classification heads
    detect.nc = new_nc
    detect.no = new_nc + detect.reg_max * 4

    for i in range(len(detect.cv3)):
        old_conv = detect.cv3[i][-1]
        detect.cv3[i][-1] = nn.Conv2d(
            old_conv.in_channels, new_nc,
            kernel_size=1, stride=1, padding=0,
        )
        #nn.init.kaiming_normal_(detect.cv3[i][-1].weight, nonlinearity="relu")
        #nn.init.zeros_(detect.cv3[i][-1].bias)
        detect.bias_init()
    # Update bookkeeping on the DetectionModel
    if class_names is None:
        class_names = {i: f"class_{i}" for i in range(new_nc)}
    yolo.model.nc = new_nc
    yolo.model.names = class_names

    # Update the YOLO wrapper's overrides so val() knows the task/nc
    yolo.overrides["nc"] = new_nc

    return yolo  # <-- return the FULL wrapper