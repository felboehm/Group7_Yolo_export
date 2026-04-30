def freeze_backbone(model):
    """Freeze the first 10 layers (backbone) of YOLOv8n."""
    for i, layer in enumerate(model.model):
        if i < 10:
            for param in layer.parameters():
                param.requires_grad = False
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"  🔒 Backbone frozen — {frozen} params locked (layers 0–9)")


def unfreeze_all(model):
    """Unfreeze every layer so full fine-tuning can begin."""
    for param in model.parameters():
        param.requires_grad = True
    print("  🔓 All layers unfrozen")