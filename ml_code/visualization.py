import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO
import cv2
#from ultralytics.utils.ops import non_max_suppression
#import ultralytics.utils


def print_history(history):
    if not history:
        print("History is empty.")
        return

    # ── Header ────────────────────────────────────────────
    header = (
        f"{'Epoch':>6} │ "
        f"{'Train Loss':>10} {'Box':>8} {'Cls':>8} {'DFL':>8} │ "
        f"{'Val Loss':>10} {'Val Box':>8} {'Val Cls':>8} {'Val DFL':>8} │ "
        f"{'LR':>10}"
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for row in history:
        print(
            f"{row['epoch']:>6} │ "
            f"{row['loss']:>10.4f} {row['box']:>8.4f} {row['cls']:>8.4f} {row['dfl']:>8.4f} │ "
            f"{row['val_loss']:>10.4f} {row['val_box']:>8.4f} {row['val_cls']:>8.4f} {row['val_dfl']:>8.4f} │ "
            f"{row['lr']:>10.2e}"
        )

    print(sep)

    # ── Summary ───────────────────────────────────────────
    best = min(history, key=lambda x: x["val_loss"])
    last = history[-1]
    print(f"\n  Total epochs trained : {len(history)}")
    print(f"  Best  val_loss       : {best['val_loss']:.4f}  @ epoch {best['epoch']}")
    print(f"  Final val_loss       : {last['val_loss']:.4f}  @ epoch {last['epoch']}")
    print(f"  Final LR             : {last['lr']:.2e}")
    print(f"{sep}\n")

def plot_history(history, save_dir="runs/custom_yolov8n", show=True):
    if not history:
        print("History is empty — nothing to plot.")
        return

    epochs    = [r["epoch"]    for r in history]
    loss      = [r["loss"]     for r in history]
    box       = [r["box"]      for r in history]
    cls       = [r["cls"]      for r in history]
    dfl       = [r["dfl"]      for r in history]
    val_loss  = [r["val_loss"] for r in history]
    val_box   = [r["val_box"]  for r in history]
    val_cls   = [r["val_cls"]  for r in history]
    val_dfl   = [r["val_dfl"]  for r in history]
    lr        = [r["lr"]       for r in history]

    best_epoch = min(history, key=lambda x: x["val_loss"])["epoch"]

    # ── Figure layout ─────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("YOLOv8n Custom Training History", fontsize=16, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax_total = fig.add_subplot(gs[0, :])   # top row: full width — total loss
    ax_box   = fig.add_subplot(gs[1, 0])   # bottom left  — box loss
    ax_cls   = fig.add_subplot(gs[1, 1])   # bottom mid   — cls loss
    ax_dfl   = fig.add_subplot(gs[1, 2])   # bottom right — dfl loss

    TRAIN_COLOR = "#2196F3"   # blue
    VAL_COLOR   = "#F44336"   # red
    BEST_COLOR  = "#4CAF50"   # green
    LR_COLOR    = "#FF9800"   # orange

    # ── Helper ────────────────────────────────────────────
    def _plot_pair(ax, train_vals, val_vals, title, ylabel="Loss"):
        ax.plot(epochs, train_vals, color=TRAIN_COLOR, lw=2,
                label="Train", marker="o", markersize=3)
        ax.plot(epochs, val_vals,   color=VAL_COLOR,   lw=2,
                label="Val",   marker="s", markersize=3, linestyle="--")
        ax.axvline(best_epoch, color=BEST_COLOR, lw=1.5,
                   linestyle=":", label=f"Best (ep {best_epoch})")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(epochs[0], epochs[-1])

    # ── Total loss (top, full width) ──────────────────────
    _plot_pair(ax_total, loss, val_loss, "Total Loss (box + cls + dfl)")

    # Add LR on a twin axis for the total loss plot
    ax_lr = ax_total.twinx()
    ax_lr.plot(epochs, lr, color=LR_COLOR, lw=1.5,
               linestyle="-.", label="LR", alpha=0.7)
    ax_lr.set_ylabel("Learning Rate", color=LR_COLOR, fontsize=9)
    ax_lr.tick_params(axis="y", labelcolor=LR_COLOR)
    ax_lr.set_yscale("log")

    # Combine legends from both axes
    lines1, labels1 = ax_total.get_legend_handles_labels()
    lines2, labels2 = ax_lr.get_legend_handles_labels()
    ax_total.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    # ── Component losses (bottom row) ─────────────────────
    _plot_pair(ax_box, box, val_box, "Box Loss")
    _plot_pair(ax_cls, cls, val_cls, "Cls Loss")
    _plot_pair(ax_dfl, dfl, val_dfl, "DFL Loss")

    # ── Save ──────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "training_history.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  📊 Plot saved → {path}")

    #if show:
     #   plt.show()

    plt.close(fig)
    return path

# ──────────────────────────────────────────────────────────
#  CLASS NAMES  (replace with your own)
# ──────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Pedestrian", 1: "People", 2: "Bicycle", 3: "Car", 4: "Van", 5: "Truck", 6: "Tricycle", 7: "Awning-tricycle", 8: "Bus", 9: "Motor", 10: "Other"
}

# One distinct colour per class (auto-generated if more classes)
COLORS = plt.cm.get_cmap("tab20", len(CLASS_NAMES))


# ──────────────────────────────────────────────────────────
#  LOAD IMAGE  (accepts path, np.ndarray, or PIL.Image)
# ──────────────────────────────────────────────────────────
def load_image(image_input):
    """
    Returns:
        img_rgb  : np.ndarray  (H, W, 3)  uint8  RGB
        img_pil  : PIL.Image               for Ultralytics input
    """
    if isinstance(image_input, str):
        img_pil = Image.open(image_input).convert("RGB")
        img_rgb = np.array(img_pil)

    elif isinstance(image_input, np.ndarray):
        # assume BGR (OpenCV) → convert to RGB
        if image_input.ndim == 3 and image_input.shape[2] == 3:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_input
        img_pil = Image.fromarray(img_rgb)

    elif isinstance(image_input, Image.Image):
        img_pil = image_input.convert("RGB")
        img_rgb = np.array(img_pil)

    else:
        raise TypeError(
            f"image_input must be a file path, np.ndarray, or PIL.Image. "
            f"Got {type(image_input)}"
        )

    return img_rgb, img_pil


# ──────────────────────────────────────────────────────────
#  RUN INFERENCE
# ──────────────────────────────────────────────────────────
def run_inference(model, img_pil, conf_threshold=0.25, iou_threshold=0.45, device=None):
    """
    Runs YOLOv8n inference and returns parsed detections.

    Args:
        model          : Ultralytics YOLO  OR  underlying nn.Module
        img_pil        : PIL.Image
        conf_threshold : float  — discard boxes below this confidence
        iou_threshold  : float  — NMS IoU threshold
        device         : torch.device or None (auto-detect)

    Returns:
        detections : list of dicts, each:
            {
                "bbox"  : [x1, y1, x2, y2],   # absolute pixel coords
                "conf"  : float,
                "cls"   : int,
                "label" : str,
            }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Handle both raw nn.Module and YOLO wrapper ─────────
    if hasattr(model, "predict"):
        # Ultralytics YOLO wrapper → use the built-in predict pipeline
        results = model.predict(
            source    = img_pil,
            conf      = conf_threshold,
            iou       = iou_threshold,
            device    = device,
            verbose   = False,
        )
        result = results[0]   # single image

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf           = float(box.conf[0])
            cls            = int(box.cls[0])
            label          = CLASS_NAMES.get(cls, f"class_{cls}")
            detections.append({
                "bbox"  : [x1, y1, x2, y2],
                "conf"  : conf,
                "cls"   : cls,
                "label" : label,
            })

    else:
        # Raw nn.Module → manual inference pipeline
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(img_tensor)

        # preds is raw — you would need to apply NMS yourself
        # This is only needed if you stripped the YOLO wrapper entirely  

        raise NotImplementedError(
            "Pass the full YOLO wrapper (ultralytics.YOLO) "
            "rather than the raw nn.Module for automatic NMS."
        )

    return detections


# ──────────────────────────────────────────────────────────
#  VISUALIZE
# ──────────────────────────────────────────────────────────
def visualize_detections(
    image_input,
    detections,
    class_names   = CLASS_NAMES,
    figsize       = (12, 8),
    box_alpha     = 0.25,       # fill transparency
    font_size     = 11,
    min_conf      = 0.0,        # extra filter (already filtered by run_inference)
    save_path     = None,       # e.g. "runs/inference/result.png"
    show          = True,
):
    """
    Draws bounding boxes + labels on the image.

    Args:
        image_input : path | np.ndarray | PIL.Image
        detections  : output of run_inference()
        save_path   : if given, saves the figure to this path
        show        : if True, calls plt.show()
    """
    img_rgb, _ = load_image(image_input)
    H, W       = img_rgb.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_rgb)
    ax.axis("off")

    filtered = [d for d in detections if d["conf"] >= min_conf]

    if not filtered:
        ax.set_title("No detections above threshold", fontsize=13)
    else:
        for det in filtered:
            x1, y1, x2, y2 = det["bbox"]
            conf            = det["conf"]
            cls             = det["cls"]
            label           = det["label"]

            # ── Colour per class ───────────────────────────
            color = COLORS(cls % COLORS.N)   # RGBA

            box_w = x2 - x1
            box_h = y2 - y1

            # ── Bounding box rectangle ─────────────────────
            rect = patches.Rectangle(
                (x1, y1), box_w, box_h,
                linewidth = 2,
                edgecolor = color,
                facecolor = (*color[:3], box_alpha),   # semi-transparent fill
            )
            ax.add_patch(rect)

            # ── Label background + text ────────────────────
            label_text = f"{label}  {conf:.2f}"

            # measure text height for background box
            txt = ax.text(
                x1, y1 - 4, label_text,
                fontsize    = font_size,
                color       = "white",
                fontweight  = "bold",
                va          = "bottom",
                ha          = "left",
                bbox        = dict(
                    boxstyle    = "round,pad=0.25",
                    facecolor   = color,
                    edgecolor   = "none",
                    alpha       = 0.85,
                ),
            )

    # ── Summary title ─────────────────────────────────────
    counts = {}
    for d in filtered:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    summary = "  |  ".join([f"{v}× {k}" for k, v in sorted(counts.items())])
    ax.set_title(
        f"Detections: {len(filtered)}    ({summary})",
        fontsize   = 13,
        fontweight = "bold",
        pad        = 10,
    )

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  🖼️  Saved → {save_path}")

    #if show:
     #   plt.show()

    plt.close(fig)


# ──────────────────────────────────────────────────────────
#  ALL-IN-ONE CONVENIENCE FUNCTION
# ──────────────────────────────────────────────────────────
def infer_and_visualize(
    model,
    image_input,
    conf_threshold = 0.25,
    iou_threshold  = 0.45,
    class_names    = CLASS_NAMES,
    figsize        = (12, 8),
    save_path      = None,
    show           = True,
    device         = None,
):
    """
    One call → inference + visualization.

    Args:
        model       : ultralytics.YOLO  (loaded from checkpoint)
        image_input : file path | np.ndarray | PIL.Image
        save_path   : e.g. "runs/inference/result.png"  or None

    Returns:
        detections  : list of detection dicts
    """
    _, img_pil = load_image(image_input)

    print(f"  🔍 Running inference  (conf≥{conf_threshold}  iou≤{iou_threshold})")
    detections = run_inference(model, img_pil, conf_threshold, iou_threshold, device)

    print(f"  ✅ {len(detections)} detections found")
    for d in detections:
        print(
            f"     {d['label']:<15} conf={d['conf']:.3f}  "
            f"bbox=[{d['bbox'][0]:.0f}, {d['bbox'][1]:.0f}, "
            f"{d['bbox'][2]:.0f}, {d['bbox'][3]:.0f}]"
        )

    visualize_detections(
        image_input,
        detections,
        class_names = class_names,
        figsize     = figsize,
        save_path   = save_path,
        show        = show,
    )

    return detections
