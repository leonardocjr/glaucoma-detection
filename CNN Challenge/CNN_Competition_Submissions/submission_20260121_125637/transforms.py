"""
Test transforms for Glaucoma Detection model
These transforms must match exactly the ones used during training (without augmentation)
"""

from torchvision import transforms


def get_test_transform():
    """
    This function must return the transforms to apply to test images.

    IMPORTANT: These should match the transforms you used during training
    (without data augmentation).

    Returns:
        transform: torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
