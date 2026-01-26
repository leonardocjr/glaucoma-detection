"""
GradCAM runner (Step 8) for glaucoma model.
Generates side-by-side Original | GradCAM Heatmap | Overlay for all test images.
Requirements: pip install grad-cam pillow torch torchvision matplotlib
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg19_bn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
BASE_PATH = r"c:\Users\raull\Downloads\BIP Project Team L-20260120T105417Z-3-001\BIP Project Team L"
train_dir = f"{BASE_PATH}\\data\\images\\training_set"
test_dir = f"{BASE_PATH}\\data\\images\\test_set"
model_path = Path("model.pth")
output_dir = Path(BASE_PATH) / "gradcam_results"

# -----------------------------------------------------------------------------
# DEVICE & DATALOADER SETTINGS
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 if os.name == "nt" else 2
PIN_MEMORY = torch.cuda.is_available()

# -----------------------------------------------------------------------------
# MODEL BUILDER
# -----------------------------------------------------------------------------
def build_model():
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró {model_path}. Entrena primero para generar model.pth")

    model = vgg19_bn(weights=None)
    # Freeze feature extractor
    for p in model.features.parameters():
        p.requires_grad = False
    # Replace head
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2),
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    # Enable grads for GradCAM
    for p in model.parameters():
        p.requires_grad = True
    return model

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def run_gradcam():
    print(f"Device: {device}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # Transforms: normalized for model, de-normalized for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),
    ])
    vis_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # will de-normalize manually if needed
    ])

    # Dataset for model input (normalized)
    test_dataset = ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    class_names = test_dataset.classes

    model = build_model()
    target_layers = [model.features[48]]
    cam = GradCAM(model=model, target_layers=target_layers)

    for idx, (input_tensor, label) in enumerate(tqdm(test_loader, desc="GradCAM")):
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, 1)[0, pred_class].item()

        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Build original image (de-normalize)
        img_tensor = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
        img_original = std * img_tensor + mean
        img_original = np.clip(img_original, 0, 1)

        cam_image = show_cam_on_image(img_original, grayscale_cam, use_rgb=True)

        # Plot triple
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(img_original)
        axes[0].set_title("Original", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(grayscale_cam, cmap="jet")
        axes[1].set_title("GradCAM", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(cam_image)
        pred_label = class_names[pred_class]
        true_label = class_names[label]
        axes[2].set_title(f"Overlay (Pred: {pred_label} {confidence:.2%})", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()

        filename = os.path.basename(test_dataset.imgs[idx][0])
        filename_base = os.path.splitext(filename)[0]
        save_filename = f"{filename_base}_{true_label}_pred{pred_label}.png"
        save_path = output_dir / save_filename
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()

    print(f"Done. Saved {len(test_dataset)} images to {output_dir}")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    run_gradcam()
