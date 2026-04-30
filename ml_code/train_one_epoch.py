import torch
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
from .prepare_batch import prepare_batch

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, CFG):
    model.train()

    running = {"loss": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0}
    pbar = tqdm(loader, desc=f"Epoch {epoch:>3} [Train]", leave=False)
    device_type = device.type if hasattr(device, 'type') else str(device).split(':')[0]

    for batch in pbar:
        # ── Unpack batch (adjust keys to your dataloader) ──
        ub = prepare_batch(batch, device)

        optimizer.zero_grad()

        # ── Forward ────────────────────────────────────────
        with autocast(device_type=device_type, enabled=scaler is not None):
            # YOLO models in train mode return (total_loss, [box, cls, dfl])
            # Adjust to match your model's return signature
            loss, loss_items = model.loss(ub)

                # ── Fix: sum loss if it is not scalar ──────────────
                # Newer Ultralytics returns loss as (3,) tensor
                # containing [box_loss, cls_loss, dfl_loss]
            if loss.dim() > 0:
                loss = loss.sum()

        # ── Backward ───────────────────────────────────────
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                           # unscale before clip
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CFG["grad_clip"]
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CFG["grad_clip"]
            )
            optimizer.step()

        # ── Accumulate metrics ─────────────────────────────
        running["loss"] += loss.item()
        if loss_items is not None and len(loss_items) >= 3:
            running["box"] += loss_items[0].detach().item()
            running["cls"] += loss_items[1].detach().item()
            running["dfl"] += loss_items[2].detach().item()

        pbar.set_postfix({
            "loss" : f"{loss.item():.4f}",
            "box"  : f"{loss_items[0].item():.4f}" if loss_items is not None else "—",
            "cls"  : f"{loss_items[1].item():.4f}" if loss_items is not None else "—",
        })

    n = len(loader)
    return {k: v / n for k, v in running.items()}
