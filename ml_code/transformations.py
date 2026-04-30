import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_transform(IMAGE_SIZE = (640,640)):
    
    return_func = A.Compose([
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.2),
        
        # Color
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.RandomRain(p=0.1),
        A.GaussNoise(p=0.1),
        
        # Blur/Noise
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.MotionBlur(blur_limit=3, p=0.1),
        
        # Normalization
        A.Resize(*IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], #IMAGENET VALUES
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3  # Discard boxes < 30% visible after transform
    ))
    return return_func

def val_transform(IMAGE_SIZE = (640,640)):
    return_func= A.Compose([
        A.Resize(*IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'],
        min_visibility=0.0,
        min_area=0.0), strict=False)
    return return_func