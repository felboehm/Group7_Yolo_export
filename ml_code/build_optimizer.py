import torch
import torch.optim as optim

def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    """
    Split parameters into two groups:
      - weights → apply weight decay
      - biases & BN layers → no weight decay
    """
    decay_params    = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = optim.AdamW(param_groups, lr=lr)
    return optimizer
