from torch.optim.lr_scheduler import CosineAnnealingLR

def build_scheduler(optimizer, num_epochs: int):
    """Cosine annealing: smoothly decays LR to near zero."""
    return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
