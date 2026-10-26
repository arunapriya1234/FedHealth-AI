
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
    / "balanced_dp_classweighted_epochs5_run"
    / "best_parameters.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_dp_classweighted_test_evaluation"
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
    print("CLASS-WEIGHTED DP-FEDAVG OFFICIAL TEST EVALUATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    x_path = TEST_DIR / "X_test.txt"
    y_path = TEST_DIR / "y_test.txt"

    for path in [x_path, y_path, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------------
    # Load official UCI HAR test data
    # --------------------------------------------------------

    print("\nLoading official UCI HAR test dataset...")

    X_test = np.loadtxt(x_path, dtype=np.float32)
    y_test = np.loadtxt(y_path, dtype=np.int64)

    if X_test.ndim != 2 or X_test.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} features per sample. "
            f"Found shape: {X_test.shape}"
        )

    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            "Test feature and label counts do not match."
        )

    if len(y_test) == 0:
        raise ValueError("The test dataset is empty.")

    # Original UCI HAR labels: 1-6.
    # Model labels: 0-5.
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
    # Load class-weighted model
    # --------------------------------------------------------

    print("\nLoading class-weighted DP model...")

    model = WearableMLP(
        input_dim=NUM_FEATURES,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    saved_parameters = np.load(MODEL_PATH)
    current_state = model.state_dict()

    if set(saved_parameters.files) != set(current_state.keys()):
        raise ValueError(
            "Saved parameter names do not match the model "
            "architecture. Check models_no_dropout.py."
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

    print("Class-weighted model loaded successfully.")

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print("\nEvaluating model on official test data...")

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
    print("CLASS-WEIGHTED MODEL: OFFICIAL TEST RESULTS")
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
    # Specifically inspect Sitting performance
    # --------------------------------------------------------

    sitting_index = ACTIVITY_NAMES.index("Sitting")

    sitting_precision = report["Sitting"]["precision"]
    sitting_recall = report["Sitting"]["recall"]
    sitting_f1 = report["Sitting"]["f1-score"]

    sitting_correct = int(
        matrix[sitting_index, sitting_index]
    )

    sitting_total = int(matrix[sitting_index].sum())

    print("\n" + "=" * 65)
    print("SITTING ACTIVITY DIAGNOSTICS")
    print("=" * 65)

    print(f"Actual Sitting samples: {sitting_total}")
    print(f"Correct Sitting predictions: {sitting_correct}")
    print(f"Sitting precision: {sitting_precision * 100:.2f}%")
    print(f"Sitting recall: {sitting_recall * 100:.2f}%")
    print(f"Sitting F1-score: {sitting_f1 * 100:.2f}%")

    # --------------------------------------------------------
    # Save results in a separate output folder
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([{
        "model": "Class-Weighted DP-FedAvg (5 Local Epochs)",
        "test_samples": len(y_test),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "macro_f1": macro_f1,
        "macro_f1_percent": macro_f1 * 100,
        "weighted_f1": weighted_f1,
        "weighted_f1_percent": weighted_f1 * 100,
        "sitting_precision": sitting_precision,
        "sitting_recall": sitting_recall,
        "sitting_f1": sitting_f1,
        "sitting_correct": sitting_correct,
        "sitting_total": sitting_total,
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
        "actual_activity": [
            ACTIVITY_NAMES[i] for i in y_test
        ],
        "predicted_activity": [
            ACTIVITY_NAMES[i] for i in predictions
        ],
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
        "\nNote: These results evaluate the saved model on the "
        "official test dataset. They do not independently "
        "establish an end-to-end differential privacy guarantee."
    )


if __name__ == "__main__":
    main()
