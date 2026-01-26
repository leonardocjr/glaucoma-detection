import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    """
    Returns the model architecture.
    
    Args:
        num_classes (int): Number of output classes (default: 2)
    
    Returns:
        model: PyTorch model
    """
    model = models.resnet50(pretrained=False)
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model