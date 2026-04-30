import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from albumentations import Resize
import os

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = sorted(os.listdir(img_dir))

        if not self.transform:
            self.transform = Resize(640,640)
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        #print("getting item")
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, 
                                  img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # Load image as numpy array (Albumentations needs this)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load labels
        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    class_labels.append(int(values[0]))
                    # YOLO format: class_id, x_center, y_center, width, height
                    bboxes.append([np.clip(float(v), 0.0, 1.0) for v in values[1:]])

        bboxes = np.array(bboxes)
            # Debug: Print values
        #print(f"Bboxes shape: {bboxes.shape}")
        #print(f"Bboxes min/max: {bboxes.min()}, {bboxes.max()}")
        #print(f"Bboxes:\n{bboxes}")
        #print(f"Class labels: {class_labels}")
        bboxes = np.clip(bboxes, 0.0, 1.0)
        # Apply augmentations (including bbox transformations)
        if self.transform:
            bboxes, class_labels = sanitize_yolo_bboxes(bboxes, class_labels)  # ← FIX: clip BEFORE transform
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            image = transformed['image']
            bboxes = np.clip(transformed['bboxes'], 0.0, 1.0)
            class_labels = transformed['class_labels']

        # Convert image to tensor (FIX)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # Convert to CHW format
        
        # Convert to tensor format with class labels
        if len(bboxes) > 0:
            bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)

            class_labels = [int(label) if isinstance(label, str) else label for label in class_labels]
            class_tensor = torch.tensor(class_labels, dtype=torch.long)
            # Combine: [class_id, x_center, y_center, width, height]
            bboxes_tensor = torch.cat([
                class_tensor.unsqueeze(1),
                bboxes_tensor
            ], dim=1)
        else:
            bboxes_tensor = torch.zeros((0, 5), dtype=torch.float32)
        
        return image, bboxes_tensor

def sanitize_yolo_bboxes(bboxes, labels, min_dim=1e-4):
    """Clip YOLO bboxes so derived corners stay strictly in [0, 1]."""
    bboxes = np.array(bboxes, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    if len(bboxes) == 0:
        return bboxes

    x_c, y_c, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    # Convert to corners (same math albumentations does internally)
    x_min = np.clip(x_c - w / 2, 0.0, 1.0)
    y_min = np.clip(y_c - h / 2, 0.0, 1.0)
    x_max = np.clip(x_c + w / 2, 0.0, 1.0)
    y_max = np.clip(y_c + h / 2, 0.0, 1.0)


    # Filter out boxes where clipping made width or height zero
    valid = (x_max - x_min > min_dim) & (y_max - y_min > min_dim)
    bboxes  = bboxes[valid]
    labels = labels[valid]
    
    x_min, y_min = x_min[valid], y_min[valid]
    x_max, y_max = x_max[valid], y_max[valid]

    if len(bboxes) == 0:
        return bboxes
    # Convert back to YOLO
    bboxes[:, 0] = (x_min + x_max) / 2
    bboxes[:, 1] = (y_min + y_max) / 2
    bboxes[:, 2] = x_max - x_min
    bboxes[:, 3] = y_max - y_min


    return bboxes, labels