import sys
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_processing import load_dataset
from models import WearableMLP

RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = RESULTS_DIR / "dp_fedavg_balanced_best_parameters.npz"

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Run src/server_dp_balanced.py first."
        )

    # Load the official UCI HAR dataset.
    _, _, X_test, y_test, features, activities = load_dataset()

    X_test = np.asarray(X_test, dtype=np.float32)

    # IMPORTANT:
    # UCI HAR test labels are 1–6, but the trained model predicts 0–5.
    y_test = np.asarray(y_test).reshape(-1).astype(np.int64) - 1

    # Load the trained model.
    model = WearableMLP(input_dim=561, num_classes=6)
    saved_parameters = np.load(MODEL_PATH)

    model_state = model.state_dict()

    for key in model_state:
        if key not in saved_parameters:
            raise KeyError(f"Missing model parameter: {key}")

        model_state[key] = torch.tensor(
            saved_parameters[key],
            dtype=model_state[key].dtype,
        )

    model.load_state_dict(model_state)
    model.eval()

    # Generate predictions.
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(inputs)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

    labels = list(range(6))

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    print("\n===== Balanced DP-FedAvg: Official Test Results =====")
    print(f"Test samples: {len(y_test)}")
    print(f"Number of features: {X_test.shape[1]}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro-F1: {macro_f1 * 100:.2f}%")

    print("\n===== Classification Report =====")
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=ACTIVITY_NAMES,
        zero_division=0,
        output_dict=True,
    )
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=ACTIVITY_NAMES,
            zero_division=0,
        )
    )

    print("\n===== Predicted Class Distribution =====")
    for class_id, name in enumerate(ACTIVITY_NAMES):
        count = int(np.sum(y_pred == class_id))
        print(f"{name}: {count}")

    # Save overall metrics.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{
            "model": "Balanced DP-FedAvg",
            "test_samples": len(y_test),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }]
    ).to_csv(
        RESULTS_DIR / "dp_fedavg_balanced_test_results.csv",
        index=False,
    )

    # Save confusion matrix.
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    pd.DataFrame(
        cm,
        index=ACTIVITY_NAMES,
        columns=ACTIVITY_NAMES,
    ).to_csv(
        RESULTS_DIR / "dp_fedavg_balanced_confusion_matrix.csv"
    )

    # Save classification report.
    pd.DataFrame(report).transpose().to_csv(
        RESULTS_DIR / "dp_fedavg_balanced_classification_report.csv"
    )

    print("\nResults saved in the results folder.")


if __name__ == "__main__":
    main()