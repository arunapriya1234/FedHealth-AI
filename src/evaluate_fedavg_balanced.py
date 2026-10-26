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

from models import WearableMLP


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_fedavg"
    / "fedavg_balanced_best_parameters.npz"
)

DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "balanced_fedavg"

ACTIVITIES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

    X_path = DATA_DIR / "test" / "X_test.txt"
    y_path = DATA_DIR / "test" / "y_test.txt"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError("Official UCI HAR test files not found.")

    X_test = np.loadtxt(X_path, dtype=np.float32)
    y_test = np.loadtxt(y_path, dtype=np.int64) - 1

    if X_test.shape != (2947, 561) or len(y_test) != len(X_test):
        raise ValueError(
            f"Unexpected test-data dimensions: {X_test.shape}, {y_test.shape}"
        )

    model = WearableMLP()

    saved_parameters = np.load(MODEL_PATH)

    expected_keys = list(model.state_dict().keys())
    parameter_arrays = []

    for index, key in enumerate(expected_keys):
        array_key = f"arr_{index}"

        if array_key not in saved_parameters:
            raise ValueError(f"Missing saved parameter array: {array_key}")

        array = saved_parameters[array_key]

        if array.shape != model.state_dict()[key].shape:
            raise ValueError(
                f"Shape mismatch for {key}: "
                f"saved {array.shape}, expected {model.state_dict()[key].shape}"
            )

        parameter_arrays.append(array)

    state = model.state_dict()

    updated_state = {
        key: torch.as_tensor(
            parameter_arrays[index],
            dtype=value.dtype,
        )
        for index, (key, value) in enumerate(state.items())
    }

    model.load_state_dict(updated_state)
    model.eval()

    with torch.no_grad():
        outputs = model(torch.tensor(X_test, dtype=torch.float32))
        predictions = outputs.argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    print("\nBALANCED NON-PRIVATE FEDAVG: OFFICIAL TEST RESULTS")
    print("-" * 54)
    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro-F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(6),
        target_names=ACTIVITIES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(6),
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=ACTIVITIES,
        columns=ACTIVITIES,
    )

    predicted_counts = pd.Series(
        predictions
    ).value_counts().reindex(range(6), fill_value=0)

    print("\nPER-ACTIVITY CLASSIFICATION REPORT")
    print(report_df.round(4).to_string())

    print("\nCONFUSION MATRIX")
    print("Rows = actual activity; columns = predicted activity")
    print(matrix_df.to_string())

    print("\nPREDICTED ACTIVITY COUNTS")
    for index, activity in enumerate(ACTIVITIES):
        print(f"{activity}: {predicted_counts[index]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "model": "Balanced non-private FedAvg",
        "test_samples": len(y_test),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }]).to_csv(
        RESULTS_DIR / "fedavg_balanced_test_results.csv",
        index=False,
    )

    report_df.to_csv(
        RESULTS_DIR / "fedavg_balanced_classification_report.csv"
    )

    matrix_df.to_csv(
        RESULTS_DIR / "fedavg_balanced_confusion_matrix.csv"
    )

    print(f"\nEvaluation files saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()