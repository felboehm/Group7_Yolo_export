import torch
from torch.cuda.amp import GradScaler
from .build_optimizer import build_optimizer
from .build_scheduler import build_scheduler
from .train_one_epoch import train_one_epoch
from .validate import validate
from .save_checkpoint import save_checkpoint, save_ultralytics_ckpt
from .freeze_unfreeze import freeze_backbone, unfreeze_all
from.load_model import load_model

CFG = {
    "weights"       : "yolov8n.pt",   # pretrained weights path
    "num_epochs"    : 50,
    "lr"            : 1e-4,
    "weight_decay"  : 5e-4,
    "grad_clip"     : 10.0,           # max gradient norm
    "use_amp"       : True,           # mixed precision training
    "save_dir"      : "runs/custom",
    "patience"      : 10,             # early stopping patience
}

def train(train_loader, val_loader, cfg=CFG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, yolo  = load_model(cfg["weights"], device)
    # ── Freeze backbone initially ──────────────────────────
    if cfg["freeze_backbone"]:
        freeze_backbone(model)

    optimizer = build_optimizer(model, cfg["lr"], cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, num_epochs=cfg["num_epochs"])
    scaler    = GradScaler() if cfg["use_amp"] and device.type == "cuda" else None

    best_val_loss  = float("inf")
    patience_count = 0
    history        = []

    print(f"\n{'='*65}")
    print(f"  Model   : YOLOv8n           Device  : {device}")
    print(f"  Epochs  : {cfg['num_epochs']}                LR      : {cfg['lr']}")
    print(f"  AMP     : {cfg['use_amp']}              Patience: {cfg['patience']}")
    print(f"  Freeze  : layers 0-9 → unfreeze @ epoch {cfg['unfreeze_epoch']}")
    print(f"{'='*65}\n")


    #from .prepare_batch import prepare_batch
    

    
  
    for epoch in range(1, cfg["num_epochs"] + 1):

        # ── Unfreeze backbone + rebuild optimizer ───────────
        if cfg["freeze_backbone"] and epoch == cfg["unfreeze_epoch"]:
            unfreeze_all(model)
            optimizer = build_optimizer(model, cfg["lr"], cfg["weight_decay"])
            scheduler = build_scheduler(optimizer, num_epochs=cfg["num_epochs"])

        # ── Train ──────────────────────────────────────────
        train_m = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, cfg
        )

        # ── Validate ───────────────────────────────────────
        val_m = validate(model, val_loader, device, epoch)

        # ── Scheduler step ─────────────────────────────────
        scheduler.step()

        # ── Log ────────────────────────────────────────────
        lr      = optimizer.param_groups[0]["lr"]
        metrics = {**train_m, **val_m, "lr": lr, "epoch": epoch}
        history.append(metrics)

        print(
            f"Epoch {epoch:>3}/{cfg['num_epochs']}  │  "
            f"Train loss {train_m['loss']:.4f} "
            f"(box {train_m['box']:.3f}  cls {train_m['cls']:.3f}  dfl {train_m['dfl']:.3f})  │  "
            f"Val {val_m['val_loss']:.4f}  │  "
            f"LR {lr:.2e}"
        )

        # ── Save last ──────────────────────────────────────
        save_checkpoint(
            model, optimizer, scheduler, epoch, metrics, cfg["save_dir"], "last"
        )

        # ── Save best ──────────────────────────────────────
        #if val_m["val_loss"] < best_val_loss:
        #    best_val_loss  = val_m["val_loss"]
        #    patience_count = 0
        #    path = save_checkpoint(
        #        model, optimizer, scheduler, epoch, metrics, cfg["save_dir"], "best"
        #    )
        #    print(f"  ✅ New best → val_loss={best_val_loss:.4f}  saved: {path}")
        #else:
        #    patience_count += 1
        #    print(f"  ⏳ No improvement ({patience_count}/{cfg['patience']})")

        if val_m["val_loss"] < best_val_loss:
            best_val_loss  = val_m["val_loss"]
            patience_count = 0
            path = save_ultralytics_ckpt(
                yolo, cfg["save_dir"])
            print(f"  ✅ New best → val_loss={best_val_loss:.4f}  saved: {path}")
        else:
            patience_count += 1
            print(f"  ⏳ No improvement ({patience_count}/{cfg['patience']})")
        # ── Early stopping ─────────────────────────────────
        if patience_count >= cfg["patience"]:
            print(f"\n  🛑 Early stopping triggered at epoch {epoch}")
            break

    print(f"\n{'='*65}")
    print(f"  Training complete!  Best val_loss : {best_val_loss:.4f}")
    print(f"  Checkpoints saved in : {cfg['save_dir']}/")
    print(f"{'='*65}\n")
    return model, history
