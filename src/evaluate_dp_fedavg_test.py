
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from models_no_dropout import WearableMLP


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_ROOT / "data" / "raw" / "test"

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_dp_no_dropout_audit_run"
    / "best_parameters.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_dp_no_dropout_test_evaluation"
)

DEVICE = torch.device("cpu")

NUM_FEATURES = 561
NUM_CLASSES = 6

ACTIVITY_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 65)
    print("FEDHEALTH AI")
    print("DP-FEDAVG OFFICIAL UCI HAR TEST EVALUATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    x_path = TEST_DIR / "X_test.txt"
    y_path = TEST_DIR / "y_test.txt"

    if not x_path.exists():
        raise FileNotFoundError(
            f"Test features not found: {x_path}"
        )

    if not y_path.exists():
        raise FileNotFoundError(
            f"Test labels not found: {y_path}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load official UCI HAR test data
    # --------------------------------------------------------

    print("\nLoading official test dataset...")

    X_test = np.loadtxt(x_path, dtype=np.float32)
    y_test = np.loadtxt(y_path, dtype=np.int64)

    if X_test.ndim != 2 or X_test.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Expected test features with {NUM_FEATURES} columns. "
            f"Found shape {X_test.shape}."
        )

    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            "Test feature and label counts do not match."
        )

    if len(y_test) == 0:
        raise ValueError("The test dataset is empty.")

    # UCI HAR labels are 1 through 6.
    # PyTorch model classes are indexed from 0 through 5.
    y_test = y_test - 1

    if not np.isin(y_test, np.arange(NUM_CLASSES)).all():
        raise ValueError("Unexpected activity labels found.")

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
        device=DEVICE,
    )

    print(f"Test samples: {len(y_test)}")
    print(f"Features per sample: {X_test.shape[1]}")

    # --------------------------------------------------------
    # Load saved best model
    # --------------------------------------------------------

    print("\nLoading best saved DP-FedAvg model...")

    model = WearableMLP(
        input_dim=NUM_FEATURES,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    saved_parameters = np.load(MODEL_PATH)

    current_state = model.state_dict()

    if set(saved_parameters.files) != set(current_state.keys()):
        raise ValueError(
            "Saved model parameter names do not match the "
            "current model architecture. Check models_no_dropout.py."
        )

    loaded_state = {}

    for key, current_tensor in current_state.items():

        saved_array = saved_parameters[key]

        if tuple(saved_array.shape) != tuple(current_tensor.shape):
            raise ValueError(
                f"Shape mismatch for {key}: "
                f"saved {saved_array.shape}, "
                f"expected {tuple(current_tensor.shape)}"
            )

        loaded_state[key] = torch.as_tensor(
            saved_array,
            dtype=current_tensor.dtype,
            device=DEVICE,
        )

    model.load_state_dict(loaded_state)
    model.eval()

    print("Best saved model loaded successfully.")

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print("\nEvaluating model...")

    with torch.no_grad():
        logits = model(X_tensor)
        predictions = logits.argmax(dim=1).cpu().numpy()

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        labels=list(range(NUM_CLASSES)),
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=ACTIVITY_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(NUM_CLASSES)),
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("OFFICIAL TEST RESULTS")
    print("=" * 65)

    print(f"Test samples: {len(y_test)}")
    print(f"Correct predictions: {(predictions == y_test).sum()}")
    print(f"Incorrect predictions: {(predictions != y_test).sum()}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"Macro-F1: {macro_f1 * 100:.2f}%")
    print(f"Weighted-F1: {weighted_f1 * 100:.2f}%")

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=ACTIVITY_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(matrix)

    # --------------------------------------------------------
    # Save results to a NEW output folder
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([{
        "model": "DP-FedAvg Best Model (Noise 0.7)",
        "test_samples": len(y_test),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "macro_f1": macro_f1,
        "macro_f1_percent": macro_f1 * 100,
        "weighted_f1": weighted_f1,
        "weighted_f1_percent": weighted_f1 * 100,
        "model_file": str(MODEL_PATH),
    }])

    summary.to_csv(
        OUTPUT_DIR / "test_summary.csv",
        index=False,
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    pd.DataFrame(
        matrix,
        index=ACTIVITY_NAMES,
        columns=ACTIVITY_NAMES,
    ).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    pd.DataFrame({
        "actual_label": y_test,
        "predicted_label": predictions,
        "correct": predictions == y_test,
    }).to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    print("\nEvaluation completed successfully!")

    print(f"Results saved in:\n{OUTPUT_DIR}")

    print("\nGenerated files:")
    print("1. test_summary.csv")
    print("2. classification_report.csv")
    print("3. confusion_matrix.csv")
    print("4. test_predictions.csv")

    print(
        "\nNote: These are test metrics for the saved best "
        "validation model. They are separate from validation "
        "metrics and do not independently establish a formal "
        "differential privacy guarantee."
    )


if __name__ == "__main__":
    main()
