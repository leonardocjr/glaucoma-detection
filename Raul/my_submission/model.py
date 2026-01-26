"""
Model architecture for Glaucoma Detection
ResNet50 with custom classifier head
"""

import torch.nn as nn
from torchvision import models


def get_model(num_classes=2):
    """
    Create a VGG19_BN model with custom classifier head for glaucoma detection.
    
    Args:
        num_classes (int): Number of output classes (default: 2 for glaucoma/normal)
    
    Returns:
        torch.nn.Module: VGG19_BN model with modified classifier
    """
    # Load base model with ImageNet pretrained weights
    model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
    
    # Freeze feature extraction layers (transfer learning)
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Custom classifier head with dropout
    in_features = model.classifier[6].in_features  # 4096 for vgg19_bn
    model.classifier[6] = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.7),
        nn.Linear(256, num_classes)
    )
    
    return model

