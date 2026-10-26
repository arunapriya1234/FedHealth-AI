
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

from models import WearableMLP
from server import set_parameters


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

NUM_CLASSES = 6
BATCH_SIZE = 64

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def load_test_data():
    """Load the official UCI HAR test set."""
    test_x_path = DATA_DIR / "test" / "X_test.txt"
    test_y_path = DATA_DIR / "test" / "y_test.txt"

    if not test_x_path.exists() or not test_y_path.exists():
        raise FileNotFoundError(
            "Official test files not found in data/raw/test/."
        )

    X_test = np.loadtxt(test_x_path, dtype=np.float32)
    y_test = np.loadtxt(test_y_path, dtype=np.int64) - 1

    if X_test.shape != (2947, 561) or len(y_test) != len(X_test):
        raise ValueError(
            f"Unexpected test-data dimensions: X={X_test.shape}, "
            f"y={y_test.shape}"
        )

    if not np.isfinite(X_test).all():
        raise ValueError("Test features contain non-finite values.")

    if not np.isin(y_test, np.arange(NUM_CLASSES)).all():
        raise ValueError("Unexpected activity labels in test data.")

    return X_test, y_test


def load_saved_parameters(model_path):
    """Load model parameters from a saved NumPy archive."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Best validation-selected model not found: {model_path}\n"
            "Run python src\\server.py first."
        )

    with np.load(model_path) as saved_parameters:
        keys = sorted(
            saved_parameters.files,
            key=lambda name: int(name.split("_")[1]),
        )
        return [saved_parameters[key].copy() for key in keys]


def evaluate_federated_model():
    model_path = (
        RESULTS_DIR / "fedavg_best_validation_parameters.npz"
    )

    X_test, y_test = load_test_data()
    parameters = load_saved_parameters(model_path)

    model = WearableMLP()
    set_parameters(model, parameters)
    model.eval()

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=ACTIVITY_NAMES,
        zero_division=0,
        output_dict=True,
    )

    print("\nFEDHEALTH AI - FINAL FEDAVG TEST RESULTS")
    print("----------------------------------------")
    print("Model: Best validation-selected FedAvg model")
    print(f"Model file: {model_path.name}")
    print(f"Official test samples: {len(all_labels)}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Test macro-F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=ACTIVITY_NAMES,
            zero_division=0,
        )
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([{
        "model": "FedAvg (best validation macro-F1)",
        "selected_model_file": model_path.name,
        "test_samples": len(all_labels),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }])

    results_path = RESULTS_DIR / "fedavg_test_results.csv"
    results.to_csv(results_path, index=False)

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
    )

    matrix_path = RESULTS_DIR / "fedavg_confusion_matrix.csv"
    pd.DataFrame(
        matrix,
        index=ACTIVITY_NAMES,
        columns=ACTIVITY_NAMES,
    ).to_csv(matrix_path)

    report_path = RESULTS_DIR / "fedavg_classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)

    print(f"\nSaved test metrics: {results_path}")
    print(f"Saved confusion matrix: {matrix_path}")
    print(f"Saved classification report: {report_path}")


if __name__ == "__main__":
    evaluate_federated_model()
