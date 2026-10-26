
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from models import WearableMLP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

BATCH_SIZE = 32
EPOCHS = 6
LEARNING_RATE = 0.001
SEED = 42


def load_data():
    """Load the official UCI-HAR training and test datasets."""

    train_x_path = DATA_DIR / "train" / "X_train.txt"
    train_y_path = DATA_DIR / "train" / "y_train.txt"
    test_x_path = DATA_DIR / "test" / "X_test.txt"
    test_y_path = DATA_DIR / "test" / "y_test.txt"

    paths = [
        train_x_path,
        train_y_path,
        test_x_path,
        test_y_path,
    ]

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required UCI-HAR file not found: {path}"
            )

    X_train = np.loadtxt(train_x_path, dtype=np.float32)
    y_train = np.loadtxt(train_y_path, dtype=np.int64) - 1

    X_test = np.loadtxt(test_x_path, dtype=np.float32)
    y_test = np.loadtxt(test_y_path, dtype=np.int64) - 1

    if X_train.shape[1] != 561 or X_test.shape[1] != 561:
        raise ValueError("Expected 561 features per sample.")

    return X_train, y_train, X_test, y_test


def train_centralized_model():
    """Train the same WearableMLP on all training samples."""

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X_train, y_train, X_test, y_test = load_data()

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = WearableMLP()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = nn.CrossEntropyLoss()

    print("FEDHEALTH AI - CENTRALIZED PYTORCH BASELINE")
    print("-" * 48)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Training epochs: {EPOCHS}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {total_loss / total:.4f} | "
            f"Training accuracy: {correct / total:.4f}"
        )

    # Evaluate only after training is complete.
    model.eval()
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=64,
        shuffle=False,
    )

    predictions = []
    actual_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            predictions.extend(outputs.argmax(dim=1).numpy())
            actual_labels.extend(labels.numpy())

    accuracy = accuracy_score(actual_labels, predictions)
    macro_f1 = f1_score(
        actual_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    print("\nCENTRALIZED TEST RESULTS")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro-F1: {macro_f1:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(
        [{
            "model": "Centralized PyTorch MLP",
            "epochs": EPOCHS,
            "test_samples": len(actual_labels),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }]
    )

    output_path = RESULTS_DIR / "centralized_pytorch_results.csv"
    results.to_csv(output_path, index=False)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    train_centralized_model()
