# STEP 10: SURROGATE MODELS COMPLETO ✅ surrogate_models_assignment.pdf
"""
Surrogate Models para tu VGG19_BN Glaucoma
Predice probabilidades CNN con features morfológicas
Adaptado para ejecución local en Windows
"""

# Instalación de paquetes necesarios (ejecutar una vez):
# pip install scikit-learn pandas matplotlib seaborn torch torchvision pillow

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# PyTorch imports
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm

# ================================
# CONFIGURACIÓN LOCAL
# ================================
BASE_PATH = r'c:\Users\raull\Downloads\BIP Project Team L-20260120T105417Z-3-001\BIP Project Team L'
train_dir = os.path.join(BASE_PATH, 'data', 'images', 'training_set')
test_dir = os.path.join(BASE_PATH, 'data', 'images', 'test_set')

print("🚀 Surrogate Models - Explicando tu ResNet50")
print("=" * 60)
print(f"📁 Base Path: {BASE_PATH}")
print(f"📁 Train Dir: {train_dir} ✅ {os.path.exists(train_dir)}")
print(f"📁 Test Dir: {test_dir} ✅ {os.path.exists(test_dir)}\n")

# ================================
# PASO 1: CARGAR TUS PREDICCIONES CNN
# ================================
print("\n1️⃣ Generando predicciones CNN (model.pth)...")

# Verificar que model.pth existe
model_path = os.path.join(BASE_PATH, 'model.pth')
if not os.path.exists(model_path):
    print(f"❌ ERROR: No se encuentra {model_path}")
    print("   Por favor, entrena primero el modelo ejecutando glaucoma_detection_local.py")
    exit(1)

# Tu modelo (del glaucoma_detection_local.py)
class GlaucomaPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Usando device: {self.device}")
        
        # Cargar modelo ResNet50 con custom head (como en model.py)
        self.model = models.resnet50(weights=None)
        in_features = self.model.fc.in_features  # 2048 para ResNet50
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict_proba(self, image_path):
        """Retorna probabilidad de glaucoma (clase 1)"""
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = torch.softmax(self.model(tensor), 1)[0, 1].item()  # Glaucoma prob
        return prob

predictor = GlaucomaPredictor()

# Generar CNN predictions para TRAIN set
print("   Generando predicciones para TRAIN set...")
trainset = ImageFolder(train_dir, transform=predictor.transform)
print(f"   Total imágenes train: {len(trainset)}")

cnn_train_results = []
for img_path, label in tqdm(trainset.imgs, desc="   Train CNN predictions"):
    filename = Path(img_path).name
    prob = predictor.predict_proba(img_path)
    cnn_train_results.append({
        'filename': filename, 
        'glaucoma_probability': prob,
        'true_label': label
    })

cnn_train_df = pd.DataFrame(cnn_train_results)
train_output_path = os.path.join(BASE_PATH, 'cnn_predictions_train.csv')
cnn_train_df.to_csv(train_output_path, index=False)
print(f"✅ {train_output_path} guardado!")
print(f"   Stats: Mean prob={cnn_train_df['glaucoma_probability'].mean():.3f}, "
      f"Std={cnn_train_df['glaucoma_probability'].std():.3f}")

# Generar CNN predictions para TEST set
print("\n   Generando predicciones para TEST set...")
testset = ImageFolder(test_dir, transform=predictor.transform)
print(f"   Total imágenes test: {len(testset)}")

cnn_test_results = []
for img_path, label in tqdm(testset.imgs, desc="   Test CNN predictions"):
    filename = Path(img_path).name
    prob = predictor.predict_proba(img_path)
    cnn_test_results.append({
        'filename': filename, 
        'glaucoma_probability': prob,
        'true_label': label
    })

cnn_test_df = pd.DataFrame(cnn_test_results)
test_output_path = os.path.join(BASE_PATH, 'cnn_predictions_test.csv')
cnn_test_df.to_csv(test_output_path, index=False)
print(f"✅ {test_output_path} guardado!")
print(f"   Stats: Mean prob={cnn_test_df['glaucoma_probability'].mean():.3f}, "
      f"Std={cnn_test_df['glaucoma_probability'].std():.3f}")

# ================================
# PASO 2: CARGAR FEATURES MORFOLÓGICAS
# ================================
print("\n2️⃣ Cargando features morfológicas...")

# Cargar features de TRAIN
train_features_path = os.path.join(BASE_PATH, 'data', 'train_RIMONE_original.csv')
test_features_path = os.path.join(BASE_PATH, 'data', 'test_RIMONE_original.csv')

if not os.path.exists(train_features_path) or not os.path.exists(test_features_path):
    print(f"❌ ERROR: No se encuentran los archivos de features")
    print(f"   Necesario: {train_features_path}")
    print(f"   Necesario: {test_features_path}")
    exit(1)

print(f"   Cargando train features: {train_features_path}")
train_features = pd.read_csv(train_features_path)
print(f"   ✅ Train features cargadas: {len(train_features)} muestras")

print(f"   Cargando test features: {test_features_path}")
test_features = pd.read_csv(test_features_path)
print(f"   ✅ Test features cargadas: {len(test_features)} muestras")

# Merge con predicciones CNN
print("\n   Combinando features con predicciones CNN...")
train_data = train_features.merge(cnn_train_df, on='filename')
test_data = test_features.merge(cnn_test_df, on='filename')

# Preparar datos (eliminar columnas no numéricas)
X_train = train_data.drop(['filename', 'glaucoma_probability', 'true_label', 'label'], axis=1, errors='ignore')
y_train = train_data['glaucoma_probability']

X_test = test_data.drop(['filename', 'glaucoma_probability', 'true_label', 'label'], axis=1, errors='ignore')
y_test = test_data['glaucoma_probability']

print(f"📊 Datos preparados:")
print(f"   TRAIN: {len(X_train)} muestras | {X_train.shape[1]} features")
print(f"   TEST:  {len(X_test)} muestras | {X_test.shape[1]} features")
print(f"   Features: {list(X_train.columns)}")
print(f"   Target range (train): {y_train.min():.3f} - {y_train.max():.3f}")
print(f"   Target range (test):  {y_test.min():.3f} - {y_test.max():.3f}")

# ================================
# PASO 3: ENTRENAR SURROGATES
# ================================
print("\n3️⃣ Entrenando Surrogate Models...")
print("   Metodología: Entrenar con TRAIN, evaluar con TEST")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")

models = {
    'LinearRegression': LinearRegression(),
    'Tree_depth5': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Tree_depth10': DecisionTreeRegressor(max_depth=10, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
}

results = {}
print("\n   Resultados de entrenamiento:")
print("   " + "-" * 70)
for name, model in models.items():
    if name == 'LinearRegression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'model': model, 'mse': mse, 'mae': mae, 'r2': r2, 'y_pred': y_pred}
    print(f"   {name:20} | MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.3f}")

# ================================
# PASO 4: ANÁLISIS MEJOR MODELO
# ================================
print("\n4️⃣ 🏆 MEJOR MODELO:")
best_name = max(results, key=lambda k: results[k]['r2'])
best = results[best_name]
print(f"   {best_name} - R²={best['r2']:.3f} (explica {best['r2']*100:.0f}% de varianza CNN)")

# Feature importance (para modelos basados en árboles)
if hasattr(best['model'], 'feature_importances_'):
    imp = best['model'].feature_importances_
    plt.figure(figsize=(10, 6))
    feat_imp = pd.DataFrame({
        'feature': X_train.columns, 
        'importance': imp
    }).sort_values('importance', ascending=True)
    
    sns.barplot(data=feat_imp.tail(8), y='feature', x='importance', palette='viridis')
    plt.title(f'{best_name} - ¿Qué features explican tu VGG19_BN?', fontsize=14, fontweight='bold')
    plt.xlabel('Importancia', fontsize=12)
    plt.tight_layout()
    
    importance_path = os.path.join(BASE_PATH, 'surrogate_importance.png')
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"   ✅ Gráfico guardado: {importance_path}")
    
    print("\n   🔥 TOP FEATURES (más importantes):")
    for idx, row in feat_imp.tail().iterrows():
        print(f"      {row['feature']}: {row['importance']:.3f}")
else:
    print("   ℹ️  Modelo lineal - ver coeficientes en lugar de importancia")

# Fidelity plot (qué tan bien el surrogate imita a la CNN)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best['y_pred'], alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
plt.plot([0,1], [0,1], 'r--', lw=2, label='Perfecto (CNN = Surrogate)')
plt.xlabel('Probabilidad CNN Real (VGG19_BN)', fontsize=12)
plt.ylabel('Predicción Surrogate', fontsize=12)
plt.title(f'Fidelity Plot - {best_name}\nMSE={best["mse"]:.4f} | MAE={best["mae"]:.4f} | R²={best["r2"]:.3f}', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

fidelity_path = os.path.join(BASE_PATH, 'fidelity_plot.png')
plt.savefig(fidelity_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"   ✅ Gráfico guardado: {fidelity_path}")

# ================================
# PASO 5: TABLA RESULTADOS ASSIGNMENT
# ================================
comparison = pd.DataFrame({
    'Modelo': list(results.keys()),
    'MSE ↓': [results[k]['mse'] for k in results],
    'MAE ↓': [results[k]['mae'] for k in results],
    'R² ↑': [results[k]['r2'] for k in results]
}).round(4)

print("\n5️⃣ 📊 TABLA COMPARATIVA FINAL:")
print("=" * 70)
print(comparison.to_string(index=False))
print("=" * 70)

# Guardar resultados
results_path = os.path.join(BASE_PATH, 'surrogate_results.csv')
comparison.to_csv(results_path, index=False)
print(f"✅ Tabla guardada: {results_path}")

# ================================
# RESUMEN FINAL
# ================================
print("\n✅ ARCHIVOS GENERADOS:")
print("=" * 70)
print(f"   1. {os.path.join(BASE_PATH, 'cnn_predictions_train.csv')}")
print(f"   2. {os.path.join(BASE_PATH, 'cnn_predictions_test.csv')}")
print(f"   3. {os.path.join(BASE_PATH, 'surrogate_results.csv')}")
print(f"   4. {os.path.join(BASE_PATH, 'surrogate_importance.png')}")
print(f"   5. {os.path.join(BASE_PATH, 'fidelity_plot.png')}")

print("\n🎉 SURROGATE MODELS ASSIGNMENT COMPLETO!")
print("=" * 70)
print("💡 Interpretación:")
print(f"   - Dataset: {len(X_train)} train + {len(X_test)} test muestras")
print(f"   - El modelo {best_name} explica {best['r2']*100:.1f}% de las predicciones CNN en test")
print(f"   - Error promedio (MAE): {best['mae']:.4f} en probabilidades")
print(f"   - Error cuadrático (MSE): {best['mse']:.4f}")
print("\n🔬 Conclusión:")
if best['r2'] > 0.7:
    print("   ✅ R² > 0.7 → CNN aprende patrones morfológicos MUY interpretables")
    print("   → Las features manuales explican bien las predicciones de la CNN")
elif best['r2'] > 0.5:
    print("   ⚠️  R² > 0.5 → CNN aprende patrones morfológicos parcialmente interpretables")
    print("   → La CNN usa tanto features manuales como patrones más complejos")
else:
    print("   ❌ R² < 0.5 → CNN usa características complejas/no lineales")
    print("   → Las features morfológicas simples NO explican bien la CNN")
print("\n📊 Metodología usada:")
print("   - Entrenamiento: Features morfológicas de TRAIN + predicciones CNN de TRAIN")
print("   - Evaluación: Features morfológicas de TEST + predicciones CNN de TEST")
print("   - Esto garantiza que no hay data leakage y resultados son generalizables")
print("=" * 70)
