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
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "balanced_dp_noise07"

sys.path.insert(0, str(SRC_DIR))

from data_processing import load_dataset
from models import WearableMLP


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = (
        RESULTS_DIR / "dp_fedavg_balanced_noise07_best_parameters.npz"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please confirm that Experiment 2 training completed successfully."
        )

    print("=" * 65)
    print("DP-FEDAVG BALANCED — EXPERIMENT 2 EVALUATION")
    print("Noise multiplier: 0.7")
    print("=" * 65)

    # Load the official UCI HAR dataset.
    _, _, X_test, y_test, features, activities = load_dataset()

    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64).reshape(-1) - 1

    print(f"\nTest samples: {len(X_test)}")
    print(f"Input features: {X_test.shape[1]}")

    # Create the model.
    model = WearableMLP(
        input_dim=X_test.shape[1],
        num_classes=6,
    )

    # Load the best model parameters from Experiment 2.
    saved_parameters = np.load(model_path, allow_pickle=False)

    state_dict = {
        key: torch.tensor(saved_parameters[key], dtype=torch.float32)
        for key in saved_parameters.files
    }

    model.load_state_dict(state_dict)
    model.eval()

    # Generate predictions.
    with torch.no_grad():
        inputs = torch.from_numpy(X_test)
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    # Activity names in the model's zero-based label order.
    activity_names = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    print("\nOFFICIAL TEST RESULTS")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro-F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")

    # Save the overall metrics.
    results_df = pd.DataFrame(
        [
            {
                "experiment": "Balanced DP-FedAvg",
                "noise_multiplier": 0.7,
                "test_samples": len(y_test),
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100,
                "macro_f1": macro_f1,
                "macro_f1_percent": macro_f1 * 100,
            }
        ]
    )

    results_path = (
        RESULTS_DIR / "dp_fedavg_balanced_noise07_test_results.csv"
    )
    results_df.to_csv(results_path, index=False)

    # Save the classification report.
    report = classification_report(
        y_test,
        predictions,
        labels=list(range(6)),
        target_names=activity_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_path = (
        RESULTS_DIR
        / "dp_fedavg_balanced_noise07_classification_report.csv"
    )
    report_df.to_csv(report_path)

    # Save the confusion matrix.
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(6)),
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=activity_names,
        columns=activity_names,
    )

    matrix_path = (
        RESULTS_DIR / "dp_fedavg_balanced_noise07_confusion_matrix.csv"
    )
    matrix_df.to_csv(matrix_path)

    # Show the class-wise results.
    print("\nCLASSIFICATION REPORT")
    print("-" * 40)
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(6)),
            target_names=activity_names,
            zero_division=0,
        )
    )

    print("\nPREDICTED ACTIVITY DISTRIBUTION")
    print("-" * 40)

    counts = np.bincount(predictions, minlength=6)

    for name, count in zip(activity_names, counts):
        print(f"{name}: {count}")

    print("\nFILES SAVED")
    print("-" * 40)
    print(results_path)
    print(report_path)
    print(matrix_path)

    print("\nExperiment 2 evaluation completed successfully.")


if __name__ == "__main__":
    main()