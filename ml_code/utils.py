import torch
import argparse
import sys

def collate_fn_list(batch):
    """Return images stacked, bboxes as list"""
    images, bboxes = zip(*batch)
    images = torch.stack(images)
    bboxes = list(bboxes)  # Keep as list of variable-length tensors
    
    return images, bboxes

def arg_reader():
    # 1️⃣ Create the parser
    parser = argparse.ArgumentParser(
        description="Train a model with configurable hyper‑parameters."
    )

    # 2️⃣ Add optional (named) arguments
    parser.add_argument(
        "--epoch",
        type=int,
        default=100,            
        help="Number of training epochs."
    )
    parser.add_argument(
        "--training-scheme",
        type=str,
        choices=["adam", "sgd", "rmsprop"],
        default="adam",
        help="Optimizer to use."
    )
    # You can keep adding as many as you need …
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini‑batch size (default: 32)."
    )

    # 3️⃣ Parse the command line (any order works)
    args = parser.parse_args()

    # 4️⃣ Use the arguments – they are now attributes of `args`
    return args
