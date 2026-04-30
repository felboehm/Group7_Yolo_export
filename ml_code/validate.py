import torch
from tqdm import tqdm
from .prepare_batch import prepare_batch

@torch.no_grad()
def validate(model, loader, device, epoch):
    model.train()

    totals = {"val_loss": 0.0, "val_box": 0.0, "val_cls": 0.0, "val_dfl": 0.0}
    pbar = tqdm(loader, desc=f"Epoch {epoch:>3} [Val]  ", leave=False)

    for batch in pbar:
        ub = prepare_batch(batch, device)

        loss, loss_items = model.loss(ub)
        # ── Fix: sum loss if it is not scalar ──────────────
        # Newer Ultralytics returns loss as (3,) tensor
        # containing [box_loss, cls_loss, dfl_loss]
        if loss.dim() > 0:
            loss = loss.sum()

        totals["val_loss"] += loss.item()
        totals["val_box"]  += loss_items[0].item()
        totals["val_cls"]  += loss_items[1].item()
        totals["val_dfl"]  += loss_items[2].item()

        pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    n = len(loader)
    return {k: v / n for k, v in totals.items()}

