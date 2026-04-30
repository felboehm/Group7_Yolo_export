import torch
import os

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
