"""
train_model.py — Entrenamiento de un MLP en PyTorch para clasificar gestos.

Uso:
    python src/train_model.py                             # usa data/raw_samples.csv
    python src/train_model.py --data data/augmented_samples.csv  # usa dataset aumentado

Guarda:
    models/gesture_model.pt
    models/scaler.pkl
    models/label_encoder.pkl
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Añadir raíz del proyecto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import TOTAL_FEATURES

# ─── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_DATA = os.path.join(BASE_DIR, "data", "raw_samples.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.pt")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# ─── Hiperparámetros ───────────────────────────────────────────────────────────
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ─── Definición del modelo MLP ─────────────────────────────────────────────────
class GestureMLP(nn.Module):
    """Red neuronal MLP de 3 capas para clasificación de gestos.

    Arquitectura:
        63 → 128 (ReLU + BN + Dropout)
        128 → 64 (ReLU + BN + Dropout)
        64 → num_classes (Softmax implícito en CrossEntropyLoss)
    """

    def __init__(self, input_size: int = TOTAL_FEATURES, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─── Entrenamiento ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo de gestos.")
    parser.add_argument(
        "--data", type=str, default=DEFAULT_DATA,
        help="Ruta al CSV de entrenamiento (default: data/raw_samples.csv)"
    )
    args = parser.parse_args()
    data_path = args.data

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Cargar datos ──
    if not os.path.isfile(data_path):
        print(f"[ERROR] No se encontró el dataset en {data_path}")
        print("       Ejecuta primero: python src/collect_data.py")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Dataset cargado: {len(df)} muestras")
    print(f"Distribución:\n{df['label'].value_counts().to_string()}\n")

    if len(df) < 20:
        print("[WARN] Muy pocas muestras. Se recomiendan ≥200 por gesto.")

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df["label"].values

    # ── Codificar etiquetas ──
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"Clases: {list(le.classes_)} ({num_classes})")

    # ── Dividir train/test ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Escalar features ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ── Preparar tensores ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # ── Modelo, loss y optimizador ──
    model = GestureMLP(input_size=TOTAL_FEATURES, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Entrenamiento ──
    print("Entrenando...")
    print("-" * 45)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_X.size(0)

        avg_loss = epoch_loss / total
        train_acc = correct / total

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS} | "
                f"Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_acc:.4f}"
            )

    print("-" * 45)

    # ── Evaluación ──
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_preds = test_outputs.argmax(dim=1).cpu().numpy()

    y_test_np = y_test_t.cpu().numpy()
    acc = accuracy_score(y_test_np, test_preds)
    print(f"\nTest Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(
        classification_report(
            y_test_np,
            test_preds,
            target_names=le.classes_,
            zero_division=0,
        )
    )

    # ── Guardar artefactos ──
    # Modelo (solo state_dict + metadata)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": TOTAL_FEATURES,
            "num_classes": num_classes,
            "classes": list(le.classes_),
        },
        MODEL_PATH,
    )
    print(f"Modelo guardado en {MODEL_PATH}")

    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler guardado en {SCALER_PATH}")

    joblib.dump(le, ENCODER_PATH)
    print(f"LabelEncoder guardado en {ENCODER_PATH}")

    print("\n¡Entrenamiento completado!")


if __name__ == "__main__":
    main()
