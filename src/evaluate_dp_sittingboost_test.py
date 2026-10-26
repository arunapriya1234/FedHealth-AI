
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
    / "balanced_dp_sittingboost_epochs5_run"
    / "best_parameters.npz"
)

# Use a NEW output folder to preserve the existing results.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_dp_sittingboost_threshold4_test_evaluation"
)

DEVICE = torch.device("cpu")

NUM_FEATURES = 561
NUM_CLASSES = 6

SITTING_INDEX = 3
STANDING_INDEX = 4

SITTING_MULTIPLIER = 4.0

ACTIVITY_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, predictions):

    report = classification_report(
        y_true,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=ACTIVITY_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=list(range(NUM_CLASSES)),
    )

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true,
            predictions,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            predictions,
            labels=list(range(NUM_CLASSES)),
            average="weighted",
            zero_division=0,
        ),
        "sitting_precision": report["Sitting"]["precision"],
        "sitting_recall": report["Sitting"]["recall"],
        "sitting_f1": report["Sitting"]["f1-score"],
        "standing_precision": report["Standing"]["precision"],
        "standing_recall": report["Standing"]["recall"],
        "standing_f1": report["Standing"]["f1-score"],
        "sitting_correct": int(matrix[SITTING_INDEX, SITTING_INDEX]),
        "sitting_total": int(matrix[SITTING_INDEX].sum()),
        "sitting_predicted_as_standing": int(
            matrix[SITTING_INDEX, STANDING_INDEX]
        ),
        "standing_predicted_as_sitting": int(
            matrix[STANDING_INDEX, SITTING_INDEX]
        ),
    }

    return metrics, report, matrix


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("FEDHEALTH AI")
    print("SITTING-BOOST MODEL: ORIGINAL VS FIXED x4 ADJUSTMENT")
    print("=" * 70)

    x_path = TEST_DIR / "X_test.txt"
    y_path = TEST_DIR / "y_test.txt"

    for path in [x_path, y_path, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------------
    # Load official test dataset
    # --------------------------------------------------------

    print("\nLoading official UCI HAR test dataset...")

    X_test = np.loadtxt(x_path, dtype=np.float32)
    y_test = np.loadtxt(y_path, dtype=np.int64) - 1

    if X_test.ndim != 2 or X_test.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} features. "
            f"Found shape: {X_test.shape}"
        )

    if X_test.shape[0] != len(y_test):
        raise ValueError(
            "Test feature and label counts do not match."
        )

    if len(y_test) == 0:
        raise ValueError("The test dataset is empty.")

    if not np.isin(y_test, np.arange(NUM_CLASSES)).all():
        raise ValueError("Unexpected activity labels found.")

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
        device=DEVICE,
    )

    print(f"Official test samples: {len(y_test)}")
    print(f"Features per sample: {X_test.shape[1]}")

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print("\nLoading trained Sitting-Boost model...")

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

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Get model probabilities
    # --------------------------------------------------------

    print("\nGenerating test predictions...")

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    # Original model predictions.
    original_predictions = probabilities.argmax(axis=1)

    # --------------------------------------------------------
    # Apply the multiplier selected using validation data.
    # --------------------------------------------------------

    adjusted_predictions = original_predictions.copy()

    sitting_probability = probabilities[:, SITTING_INDEX]
    standing_probability = probabilities[:, STANDING_INDEX]

    choose_sitting = (
        sitting_probability * SITTING_MULTIPLIER
        > standing_probability
    )

    # Only change original Standing predictions into Sitting.
    # Other original predictions remain unchanged.
    change_mask = (
        (original_predictions == STANDING_INDEX)
        & choose_sitting
    )

    adjusted_predictions[change_mask] = SITTING_INDEX

    print(
        f"Predictions adjusted from Standing to Sitting: "
        f"{int(change_mask.sum())}"
    )

    # --------------------------------------------------------
    # Evaluate both prediction sets
    # --------------------------------------------------------

    original_metrics, original_report, original_matrix = (
        calculate_metrics(y_test, original_predictions)
    )

    adjusted_metrics, adjusted_report, adjusted_matrix = (
        calculate_metrics(y_test, adjusted_predictions)
    )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OFFICIAL TEST RESULTS: COMPARISON")
    print("=" * 70)

    print(
        f"{'Metric':<32}"
        f"{'Original':>15}"
        f"{'Adjusted x4':>17}"
    )

    print("-" * 64)

    comparison_items = [
        ("Accuracy", "accuracy"),
        ("Macro-F1", "macro_f1"),
        ("Weighted-F1", "weighted_f1"),
        ("Sitting precision", "sitting_precision"),
        ("Sitting recall", "sitting_recall"),
        ("Sitting F1", "sitting_f1"),
        ("Standing precision", "standing_precision"),
        ("Standing recall", "standing_recall"),
        ("Standing F1", "standing_f1"),
    ]

    for display_name, key in comparison_items:

        original_value = original_metrics[key] * 100
        adjusted_value = adjusted_metrics[key] * 100

        print(
            f"{display_name:<32}"
            f"{original_value:>14.2f}%"
            f"{adjusted_value:>16.2f}%"
        )

    print("\nSitting diagnostic counts:")
    print(
        "Actual Sitting samples:",
        adjusted_metrics["sitting_total"],
    )
    print(
        "Original correct Sitting predictions:",
        original_metrics["sitting_correct"],
    )
    print(
        "Adjusted correct Sitting predictions:",
        adjusted_metrics["sitting_correct"],
    )
    print(
        "Adjusted Sitting predicted as Standing:",
        adjusted_metrics["sitting_predicted_as_standing"],
    )
    print(
        "Actual Standing predicted as Sitting after adjustment:",
        adjusted_metrics["standing_predicted_as_sitting"],
    )

    print("\nAdjusted model classification report:")

    print(
        classification_report(
            y_test,
            adjusted_predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=ACTIVITY_NAMES,
            zero_division=0,
        )
    )

    print("Adjusted model confusion matrix:")
    print(adjusted_matrix)

    # --------------------------------------------------------
    # Save all results in a separate folder
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_name, metrics in [
        ("Original Sitting-Boost DP-FedAvg", original_metrics),
        ("Sitting-Boost with validation-selected x4 adjustment",
         adjusted_metrics),
    ]:

        row = {
            "model": model_name,
            "test_samples": len(y_test),
            "sitting_multiplier": (
                1.0 if "Original" in model_name
                else SITTING_MULTIPLIER
            ),
        }

        row.update(metrics)
        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    comparison_df.to_csv(
        OUTPUT_DIR / "test_model_comparison.csv",
        index=False,
    )

    pd.DataFrame(
        original_report
    ).transpose().to_csv(
        OUTPUT_DIR / "original_classification_report.csv"
    )

    pd.DataFrame(
        adjusted_report
    ).transpose().to_csv(
        OUTPUT_DIR / "adjusted_classification_report.csv"
    )

    pd.DataFrame(
        original_matrix,
        index=ACTIVITY_NAMES,
        columns=ACTIVITY_NAMES,
    ).to_csv(
        OUTPUT_DIR / "original_confusion_matrix.csv"
    )

    pd.DataFrame(
        adjusted_matrix,
        index=ACTIVITY_NAMES,
        columns=ACTIVITY_NAMES,
    ).to_csv(
        OUTPUT_DIR / "adjusted_confusion_matrix.csv"
    )

    probability_data = pd.DataFrame({
        "actual_label": y_test,
        "actual_activity": [
            ACTIVITY_NAMES[i] for i in y_test
        ],
        "original_prediction": original_predictions,
        "original_activity": [
            ACTIVITY_NAMES[i] for i in original_predictions
        ],
        "adjusted_prediction": adjusted_predictions,
        "adjusted_activity": [
            ACTIVITY_NAMES[i] for i in adjusted_predictions
        ],
        "sitting_probability": sitting_probability,
        "standing_probability": standing_probability,
        "adjusted_prediction_changed": (
            original_predictions != adjusted_predictions
        ),
        "original_correct": original_predictions == y_test,
        "adjusted_correct": adjusted_predictions == y_test,
    })

    probability_data.to_csv(
        OUTPUT_DIR / "test_prediction_comparison.csv",
        index=False,
    )

    print("\nEvaluation completed successfully!")
    print(f"Results saved to:\n{OUTPUT_DIR}")

    print("\nGenerated files:")
    print("1. test_model_comparison.csv")
    print("2. original_classification_report.csv")
    print("3. adjusted_classification_report.csv")
    print("4. original_confusion_matrix.csv")
    print("5. adjusted_confusion_matrix.csv")
    print("6. test_prediction_comparison.csv")

    print(
        "\nIMPORTANT: The x4 multiplier was selected using "
        "validation data before this test evaluation. "
        "The trained model was not modified."
    )

    print(
        "This experiment evaluates a decision adjustment; "
        "it does not establish an end-to-end differential "
        "privacy guarantee."
    )


if __name__ == "__main__":
    main()
