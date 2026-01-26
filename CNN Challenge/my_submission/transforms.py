from torchvision import transforms

def get_test_transform(img_size: int = 224):
    """
    Validation/Test preprocessing (no augmentation).
    Mirrors val_test_transforms from training code.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
