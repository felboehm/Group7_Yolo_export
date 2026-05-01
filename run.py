from torch.utils.data import DataLoader
from ml_code import dataset
from ml_code import train
from ml_code import visualization
from ml_code import transformations
from ml_code.utils import collate_fn_list, arg_reader

if __name__ == "__main__":
    vals = arg_reader() 
    #----------------------------------------------------
    # PARAMETERS AND HYPERPARAMETERS
    #----------------------------------------------------
    IMAGE_SIZE = (640,640)
    EPOCHS = vals.epoch
    batch_size = vals.batch_size
    CFG = {
        "weights"       : "yolov8n.pt",   # pretrained weights path
        "num_epochs"    : EPOCHS,
        "lr"            : 1e-4,
        "weight_decay"  : 5e-4,
        "grad_clip"     : 10.0,           # max gradient norm
        "use_amp"       : True,           # mixed precision training
        "save_dir"      : "runs/custom/",
        "patience"      : 10,             # early stopping patience
        "freeze_backbone" : True,      # freeze backbone at start
        "unfreeze_epoch"  : 5,         # unfreeze everything at this epoch
        "batch_size"    : batch_size   
    }
    # TODO: ENSURE SAVE DIR IS ACCEPTABLE ON EX3
    print("CFG VALUES")
    print("="*50)
    for key, value in CFG.items():
        print(f'{key} -> {value}')
    print("="*50)

    #----------------------------------------------------
    # Paths of labels and images
    #----------------------------------------------------

    train_img ="data/yolo/images/train"
    train_labels ="data/yolo/labels/train"
    val_img = "data/yolo/images/val"
    val_labels = "data/yolo/labels/val"
    
    # TODO: FIGURE OUT HOW TO GET DATA TO EX3 CLUSTER

    #------------------------------------------------------------------
    # Transformations for the training and validation data respectively
    #------------------------------------------------------------------
    
    train_transform = transformations.train_transform()
    val_transform = transformations.val_transform()

    #----------------------------------------------------
    # Get dataloader based on the paths
    #----------------------------------------------------
    
    train_data = DataLoader(dataset.YOLODataset(img_dir=train_img, label_dir=train_labels, transform=train_transform),
    batch_size=CFG['batch_size'],
    shuffle=True,
    collate_fn=collate_fn_list)
    val_data = DataLoader(dataset.YOLODataset(img_dir=val_img, label_dir=val_labels, transform=val_transform),
    batch_size=CFG['batch_size'],
    shuffle=True,
    collate_fn=collate_fn_list)

    # TODO: MAKE SURE IT WORKS ON EX3 CLUSTER

    #----------------------------------------------------
    #  Training the model
    #----------------------------------------------------

    model, history = train.train(train_data, val_data, cfg=CFG)

    #----------------------------------------------------
    #   POSSIBLE INFERENCE
    #----------------------------------------------------

    #TODO : DECIDE IF INFERENCE SHOULD BE HERE OR IN A DIFFERENT .py FILE # DONE
    if vals.inference:
        from ultralytics import YOLO
        import os
        yolo = YOLO("runs/custom/ultralytics_best_custom.pt")
        yolo.model.eval()
        
        directory = 'data/test_data/images/test'
        save_dir = "runs/custom_yolov8n/inference_examples"

        visualization.plot_history(history, "runs/custom_yolov8n")
        
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            save = os.path.join(save_dir, filename)
            if os.path.isfile(f):
                visualization.infer_and_visualize(yolo, f, save_path = save)
    
        evaluate.evaluate(yolo, "data/data.yaml")

    
