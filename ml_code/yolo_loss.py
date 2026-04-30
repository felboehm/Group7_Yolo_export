from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import de_parallel
import torch
import torch.nn as nn

class CustomDetectionLoss(v8DetectionLoss):
     def __init__(self, model):
        super().__init__(model)