"""
Using Transfer Learning for Glaucoma Detection
Modified for local execution (adapted from Google Colab)
"""

# =============================================================================
# STEP 0: SETUP LOCAL PATHS
# =============================================================================
# Google Colab imports (commented out for local execution)
# from google.colab import drive
# drive.mount('/content/drive')

# Local paths configuration
BASE_PATH = r'c:\Users\raull\Downloads\BIP Project Team L-20260120T105417Z-3-001\BIP Project Team L'
train_dir = f'{BASE_PATH}\\data\\images\\training_set'
test_dir = f'{BASE_PATH}\\data\\images\\test_set'

import os
import sys

print(f"📁 Train: {train_dir} ✅ {os.path.exists(train_dir)}")
print(f"📁 Test:  {test_dir} ✅ {os.path.exists(test_dir)}")

# =============================================================================
# STEP 1: CORE IMPORTS
# =============================================================================
# IMPORTANT: Para usar GPU NVIDIA, instala PyTorch con soporte CUDA:
# 
# Opción 1 - CUDA 11.8 (recomendado para GPUs recientes):
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#
# Opción 2 - CUDA 12.1 (para GPUs más nuevas):
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#
# Opción 3 - Solo CPU (si no tienes GPU):
#   pip install torch torchvision torchaudio
#
# Otros paquetes:
#   pip install scikit-learn matplotlib seaborn pandas
#
# Verifica que tienes los drivers NVIDIA actualizados: https://www.nvidia.com/Download/index.aspx

# PyTorch core imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Vision transforms and models
import torchvision.transforms as transforms
from torchvision.models import vgg19_bn, VGG19_BN_Weights
from torchvision.datasets import ImageFolder

# Metrics and visualization
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import json

# GPU detection and verification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"✅ GPU NVIDIA Detectada: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  GPU no detectada - usando CPU")
    print("   Para usar GPU, instala PyTorch con CUDA:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
print(f"{'='*60}\n")

# =============================================================================
# STEP 2: UTILITY FUNCTIONS
# =============================================================================
def evaluate_auc(model, loader, device):
    """Fast AUC-ROC calculation for validation during training"""
    model.eval()  # Set model to evaluation mode
    all_probs, all_labels = [], []  # Lists to collect predictions

    with torch.no_grad():  # No gradients needed for inference
        for inputs, labels in loader:
            inputs = inputs.to(device)  # Move batch to GPU
            outputs = model(inputs)     # Forward pass
            probs = torch.softmax(outputs, 1)[:, 1].cpu().numpy()  # Probability of glaucoma (class 1)
            all_probs.extend(probs)
            all_labels.extend(labels)

    return roc_auc_score(all_labels, all_probs)  # AUC-ROC score

def evaluate_full(model, loader, device):
    """Complete evaluation: predictions, probabilities, true labels"""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, 1)[:, 1].cpu().numpy()        # Glaucoma probability
            preds = torch.max(outputs, 1)[1].cpu().numpy()              # Argmax predictions (implicit 0.5 threshold)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    return {
        'preds': np.array(all_preds),    # Binary predictions (0=normal, 1=glaucoma)
        'probs': np.array(all_probs),    # Softmax probabilities for glaucoma
        'labels': np.array(all_labels)   # Ground truth labels
    }


def print_competition_metrics(results, class_names):
    """Prints official BIP Competition metrics + confusion matrix + ROC"""
    preds, probs, labels = results['preds'], results['probs'], results['labels']

    # Calculate all BIP competition metrics
    metrics = {
        'Balanced Accuracy': balanced_accuracy_score(labels, preds),     # Main metric!
        'Accuracy': accuracy_score(labels, preds),
        'Precision': precision_score(labels, preds, pos_label=1),
        'Recall (Sensitivity)': recall_score(labels, preds, pos_label=1),
        'Specificity': recall_score(labels, preds, pos_label=0),
        'F1 Score': f1_score(labels, preds, pos_label=1),
        'AUC-ROC': roc_auc_score(labels, probs)
    }

    # Print formatted metrics
    print("="*60)
    print("BIP COMPETITION METRICS")
    print("="*60)
    for metric, value in metrics.items():
        print(f"{metric:20}: {value:.4f}")

    print(f"RANKING Balanced:{metrics['Balanced Accuracy']:.4f} F1:{metrics['F1 Score']:.4f}")

    # Confusion Matrix
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title('Confusion Matrix')
    ax[0].set_xticklabels(class_names)  # ['glaucoma', 'normal']
    ax[0].set_yticklabels(class_names)




    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    ax[1].plot(fpr, tpr, label=f'AUC={metrics["AUC-ROC"]:.4f}')
    ax[1].plot([0,1], [0,1], 'k--')  # Random classifier
    ax[1].legend()
    ax[1].set_title('ROC Curve')
    plt.tight_layout()
    plt.savefig('confusion_matrix_roc_curve.png', dpi=100, bbox_inches="tight")
    plt.show()

    # Save metrics to CSV
    pd.DataFrame(metrics, index=[0]).to_csv('bipmetrics.csv', index=False)
    print("bipmetrics.csv saved")

    return metrics

print("Functions ready!")


# =============================================================================
# CONFIG
# =============================================================================
# Safer DataLoader settings for Windows to avoid multiprocessing spawn issues
NUM_WORKERS = 0 if os.name == 'nt' else 2
PIN_MEMORY = torch.cuda.is_available()

def main():
    # =============================================================================
    # STEP 3: DATA TRANSFORMS & LOADERS
    # =============================================================================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),  # Reducido de 0.5
        transforms.RandomRotation(10),  # Reducido de 15
        # ColorJitter removido - preserva features morfológicas originales
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    test_dataset = ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(f"Train: {len(train_dataset)} images, Classes: {train_dataset.classes}")
    print(f"Test: {len(test_dataset)} images")

    # =============================================================================
    # STEP 4: MODEL DEFINITION
    # =============================================================================
    model = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False

    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.7),  # Aumentado de 0.5 → Mayor regularización
        nn.Linear(256, 2)
    )

    model = model.to(device)
    print("Model + ImageNet pre-trained ready")

    # =============================================================================
    # STEP 5: TRAINING
    # =============================================================================
    class_counts = np.bincount(train_dataset.targets)
    print("Class counts:", class_counts)
    class_weights = 1.0 / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class weights:", class_weights.cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001, weight_decay=1e-3)  # 10x más fuerte
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print("Training...")
    num_epochs = 20  # Reducido de 30 - evita overfit a patrones complejos
    best_auc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        test_auc = evaluate_auc(model, test_loader, device)
        scheduler.step(test_auc)

        print(f"Epoch {epoch+1:2d} Loss={train_loss:.4f} AUC={test_auc:.4f}")

        if test_auc > best_auc:
            best_auc = test_auc
            torch.save(model.state_dict(), 'model.pth')
            print(f"model.pth saved! Best AUC={best_auc:.4f}")

    print(f"Final Best AUC: {best_auc:.4f}")

    # =============================================================================
    # STEP 6: COMPETITION RESULTS
    # =============================================================================
    print("BIP COMPETITION RESULTS")
    results = evaluate_full(model, test_loader, device)
    class_names = train_dataset.classes
    metrics = print_competition_metrics(results, class_names)

    # =============================================================================
    # STEP 6.5: FIND BEST THRESHOLD
    # =============================================================================
    probs = results['probs']
    labels = results['labels']

    best_thr = 0.5
    best_bal = 0.0
    for thr in np.linspace(0, 1, 101):
        preds_thr = (probs >= thr).astype(int)
        bal = balanced_accuracy_score(labels, preds_thr)
        if bal > best_bal:
            best_bal = bal
            best_thr = thr

    print(f"Best threshold: {best_thr:.2f}, Balanced Acc: {best_bal:.4f}")

    # =============================================================================
    # STEP 7: SAVE FINAL MODEL
    # =============================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    torch.save(model.state_dict(), f'bip_model_competition_{timestamp}.pth')
    print(f"Competition model: bip_model_competition_{timestamp}.pth")

if __name__ == '__main__':
    main()  