import torch
import pandas as pd
from torchinfo import summary

def eval_flops(yolo, input_shape=(1, 3, 640, 640), verbose=False):
    backbone = yolo.model      # <-- the nn.ModuleList that contains the layers
    backbone.eval()
    input_shape=(1, 3, 640, 640)
    with torch.inference_mode():
        s = summary(
            backbone,
            input_size=input_shape,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            verbose=verbose,
        )
    print(f"Backbone GFLOPs: {s.total_mult_adds*2/1e9:.2f}")
    print(f"Backbone Params: {s.total_params/1e6:.2f} M")
    return s

def eval_metrics(yolo, data_yaml):
    results = yolo.val(
        data=data_yaml,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        plots=True,
        save_json=True  # Also saves predictions as JSON
    )
    
    print("\n" + "="*60)
    print("OVERALL METRICS")
    print("="*60)
    print(f"mAP@0.5:0.95 (COCO):  {results.box.map:.4f}")
    print(f"mAP@0.5 (loose):      {results.box.map50:.4f}")
    print(f"mAP@0.75:             {results.box.map75:.4f}")
    print(f"Mean Precision:       {results.box.mp:.4f}")
    print(f"Mean Recall:          {results.box.mr:.4f}")
    
    # Calculate F1
    f1 = 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr)
    print(f"Mean F1-Score:        {f1:.4f}")
    
    print("\n" + "="*60)
    print("PER-CLASS METRICS")
    print("="*60)
    
    class_names = ['person', 'car', 'bicycle']  # Your class names
    for i, class_name in enumerate(class_names):
        ap = results.box.ap[i]  # AP for this class
        p = results.box.p[i]    # Precision for this class
        r = results.box.r[i]    # Recall for this class
        f1_class = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        print(f"\n{class_name}:")
        print(f"  AP@0.5:0.95: {ap:.4f}")
        print(f"  Precision:   {p:.4f}")
        print(f"  Recall:      {r:.4f}")
        print(f"  F1-Score:    {f1_class:.4f}")
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(results.confusion_matrix.matrix)
    
    print("\n" + "="*60)
    print("FILES GENERATED")
    print("="*60)
    print(f"Saved to: {results.save_dir}")
    print("  - confusion_matrix.png")
    print("  - PR_curve.png")
    print("  - F1_curve.png")
    print("  - results.csv")
    
    # Load and display results CSV
    #df = pd.read_csv(f"{results.save_dir}/results.csv")
    #print("\nResults CSV:")
    #print(df)

def evaluate(model, data_yaml):

    #──────────────────────────────────────────────────────────
    # EVALUATION
    #──────────────────────────────────────────────────────────
    eval_flops(model)

    #──────────────────────────────────────────────────────────
    # SAVING STATS
    #──────────────────────────────────────────────────────────
    eval_metrics(model, data_yaml)