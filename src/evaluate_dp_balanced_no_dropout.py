
from pathlib import Path
import sys

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
RESULTS_DIR = PROJECT_ROOT / "experiments" / "balanced_dp_no_dropout"

sys.path.insert(0, str(SRC_DIR))

from data_processing import load_dataset
from models_no_dropout import WearableMLP


ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def main():
    model_path = RESULTS_DIR / "best_parameters.npz"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}"
        )

    # Load the official test set.
    _, _, X_test, y_test, _, _ = load_dataset()

    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64).reshape(-1) - 1

    model = WearableMLP(
        input_dim=X_test.shape[1],
        num_classes=6,
    )

    with np.load(model_path, allow_pickle=False) as saved:
        parameters = {
            key: torch.tensor(saved[key], dtype=torch.float32)
            for key in saved.files
        }

    model.load_state_dict(parameters)
    model.eval()

    with torch.no_grad():
        outputs = model(torch.from_numpy(X_test))
        predictions = outputs.argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=list(range(6)),
        average="macro",
        zero_division=0,
    )

    print("=" * 60)
    print("DP-FEDAVG WITHOUT DROPOUT — OFFICIAL TEST RESULTS")
    print("=" * 60)
    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro-F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")

    print("\nCLASSIFICATION REPORT")
    print("-" * 60)

    report_text = classification_report(
        y_test,
        predictions,
        labels=list(range(6)),
        target_names=ACTIVITY_NAMES,
        zero_division=0,
    )
    print(report_text)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "experiment": "Balanced DP-FedAvg without dropout",
        "noise_multiplier": 0.7,
        "test_samples": len(y_test),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "macro_f1": macro_f1,
        "macro_f1_percent": macro_f1 * 100,
    }]).to_csv(
        RESULTS_DIR / "test_results.csv",
        index=False,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(6)),
        target_names=ACTIVITY_NAMES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        RESULTS_DIR / "classification_report.csv"
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(6)),
    )

    pd.DataFrame(
        matrix,
        index=ACTIVITY_NAMES,
        columns=ACTIVITY_NAMES,
    ).to_csv(RESULTS_DIR / "confusion_matrix.csv")

    print("\nPREDICTED ACTIVITY DISTRIBUTION")

    counts = np.bincount(predictions, minlength=6)

    for name, count in zip(ACTIVITY_NAMES, counts):
        print(f"{name}: {count}")

    print("\nResults saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
