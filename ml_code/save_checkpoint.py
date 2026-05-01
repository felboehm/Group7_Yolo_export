import torch
import os
from copy import deepcopy

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_dir, tag="last"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{tag}.pt")
    torch.save({
        "epoch"               : epoch,
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics"             : metrics,
    }, path)
    return path

def save_ultralytics_ckpt(yolo, save_dir, tag="best_custom"):
    """Save so that YOLO('best_custom.pt') works later."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"ultralytics_{tag}.pt")
    ckpt = {
        "model": deepcopy(yolo.model).half(),   # pickled DetectionModel
        "train_args": {
            "task": "detect",
            "nc": yolo.model.nc,
            "names": yolo.model.names,
        },
        "date": None,
        "epoch": -1,
    }
    torch.save(ckpt, path)
    return path
