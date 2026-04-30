from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch
import torchvision.transforms as T
from .visualization import load_image 
from torchinfo import summary

def eval_flops(model, img):
    #input_tensor = torch.randn(1, 3, 640, 640)
    transform = T.Compose([T.Resize((640,640)),
                           T.ToTensor()])
    input_tensor = transform(img).unsqueeze(0)  # Add batch dim → (1, C, H, W)
    flops = FlopCountAnalysis(model, input_tensor)

    print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    print(flops.by_module())         # Per module breakdown
    print(parameter_count_table(model))  # Parameter count

def eval_flops(model, img):
    #transform = T.Compose([T.Resize((640,640)),
    #                       T.ToTensor()])
    #input_tensor = transform(img).unsqueeze(0)  # Add batch dim → (1, C, H, W)

    results = summary(model,
        input_size=(1, 3, 640, 640),    # (batch, C, H, W)
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        verbose=1)
    
    print(f"Total FLOPs: {results.total_mult_adds * 2 / 1e9:.2f} GFLOPs")
    print(f"Total Params: {results.total_params / 1e6:.2f} M")

def evaluate(model, image_input):
    #──────────────────────────────────────────────────────────
    # LOAD IMAGE
    #──────────────────────────────────────────────────────────
    _, img = load_image(image_input)

    #──────────────────────────────────────────────────────────
    # EVALUATION
    #──────────────────────────────────────────────────────────
    eval_flops(model, img)

    #──────────────────────────────────────────────────────────
    # SAVING STATS
    #──────────────────────────────────────────────────────────