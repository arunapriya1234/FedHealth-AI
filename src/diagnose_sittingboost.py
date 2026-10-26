
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import confusion_matrix
from models_no_dropout import WearableMLP


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_ROOT / "data" / "raw" / "test"

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_dp_sittingboost_epochs5_run"
    / "best_parameters.npz"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_dp_sittingboost_test_evaluation"
    / "test_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_dp_sittingboost_diagnostics"
)

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

SITTING_INDEX = 3
STANDING_INDEX = 4


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("FEDHEALTH AI - SITTING-BOOST DIAGNOSTICS")
    print("=" * 65)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    x_path = TEST_DIR / "X_test.txt"
    y_path = TEST_DIR / "y_test.txt"

    for path in [
        x_path,
        y_path,
        MODEL_PATH,
        PREDICTIONS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------------
    # Load official test data
    # --------------------------------------------------------

    print("\nLoading official test dataset...")

    X_test = np.loadtxt(x_path, dtype=np.float32)
    y_test = np.loadtxt(y_path, dtype=np.int64) - 1

    if X_test.ndim != 2 or X_test.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} features. "
            f"Found shape {X_test.shape}"
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Test feature and label counts do not match."
        )

    if not np.isin(y_test, np.arange(NUM_CLASSES)).all():
        raise ValueError("Unexpected activity labels found.")

    existing_predictions = pd.read_csv(PREDICTIONS_PATH)

    if len(existing_predictions) != len(y_test):
        raise ValueError(
            "Saved predictions and official test data have "
            "different sample counts."
        )

    # Verify sample ordering and label alignment.
    expected_activity = [
        ACTIVITY_NAMES[i] for i in y_test
    ]

    if not np.array_equal(
        existing_predictions["actual_activity"].to_numpy(),
        np.asarray(expected_activity),
    ):
        raise ValueError(
            "Saved predictions do not match the official "
            "test label ordering."
        )

    print(f"Test samples: {len(y_test)}")
    print(f"Features per sample: {X_test.shape[1]}")

    # --------------------------------------------------------
    # Load trained Sitting-Boost model
    # --------------------------------------------------------

    print("\nLoading Sitting-Boost model...")

    model = WearableMLP(
        input_dim=NUM_FEATURES,
        num_classes=NUM_CLASSES,
    )

    saved_parameters = np.load(MODEL_PATH)
    current_state = model.state_dict()

    if set(saved_parameters.files) != set(current_state.keys()):
        raise ValueError(
            "Saved parameter names do not match the model."
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
        )

    model.load_state_dict(loaded_state)
    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    print("\nCalculating prediction probabilities...")

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
    )

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1).numpy()

    predicted_labels = probabilities.argmax(axis=1)

    # Verify model predictions match the saved evaluation.
    saved_labels = (
        existing_predictions["predicted_label"]
        .to_numpy(dtype=np.int64)
    )

    if not np.array_equal(predicted_labels, saved_labels):
        raise ValueError(
            "New model predictions do not match the saved "
            "evaluation predictions. Check the model file."
        )

    # --------------------------------------------------------
    # Build detailed probability report
    # --------------------------------------------------------

    report_data = {
        "actual_label": y_test,
        "actual_activity": [
            ACTIVITY_NAMES[i] for i in y_test
        ],
        "predicted_label": predicted_labels,
        "predicted_activity": [
            ACTIVITY_NAMES[i] for i in predicted_labels
        ],
        "correct": predicted_labels == y_test,
        "predicted_confidence": probabilities.max(axis=1),
        "sitting_probability": probabilities[:, SITTING_INDEX],
        "standing_probability": probabilities[:, STANDING_INDEX],
    }

    for i, activity in enumerate(ACTIVITY_NAMES):
        report_data[
            f"probability_{activity.lower().replace(' ', '_')}"
        ] = probabilities[:, i]

    detailed = pd.DataFrame(report_data)

    # --------------------------------------------------------
    # Analyze Sitting samples
    # --------------------------------------------------------

    sitting_mask = y_test == SITTING_INDEX
    sitting_as_standing = (
        sitting_mask
        & (predicted_labels == STANDING_INDEX)
    )
    sitting_correct_mask = (
        sitting_mask
        & (predicted_labels == SITTING_INDEX)
    )

    sitting_count = int(sitting_mask.sum())
    sitting_correct = int(sitting_correct_mask.sum())
    sitting_predicted_standing = int(sitting_as_standing.sum())

    print("\n" + "=" * 65)
    print("SITTING CLASS ANALYSIS")
    print("=" * 65)

    print(f"Actual Sitting samples: {sitting_count}")
    print(
        f"Correctly predicted Sitting: "
        f"{sitting_correct} "
        f"({100 * sitting_correct / max(sitting_count, 1):.2f}%)"
    )
    print(
        f"Sitting predicted as Standing: "
        f"{sitting_predicted_standing} "
        f"({100 * sitting_predicted_standing / max(sitting_count, 1):.2f}%)"
    )

    if sitting_count:
        print(
            "Mean Sitting probability for actual Sitting samples: "
            f"{probabilities[sitting_mask, SITTING_INDEX].mean() * 100:.2f}%"
        )

        print(
            "Mean Standing probability for actual Sitting samples: "
            f"{probabilities[sitting_mask, STANDING_INDEX].mean() * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Analyze Sitting vs Standing probabilities
    # --------------------------------------------------------

    print("\nSitting probability compared with Standing probability:")

    if sitting_count:
        sitting_prob = probabilities[sitting_mask, SITTING_INDEX]
        standing_prob = probabilities[sitting_mask, STANDING_INDEX]

        print(
            "Sitting probability > Standing probability: "
            f"{int((sitting_prob > standing_prob).sum())} samples"
        )

        print(
            "Sitting probability = Standing probability: "
            f"{int((sitting_prob == standing_prob).sum())} samples"
        )

        print(
            "Sitting probability < Standing probability: "
            f"{int((sitting_prob < standing_prob).sum())} samples"
        )

    # --------------------------------------------------------
    # Analyze all activity confusions
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predicted_labels,
        labels=list(range(NUM_CLASSES)),
    )

    print("\nConfusion matrix:")
    print(matrix)

    # --------------------------------------------------------
    # Save diagnostic reports
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detailed.to_csv(
        OUTPUT_DIR / "test_probabilities.csv",
        index=False,
    )

    sitting_errors = detailed.loc[
        sitting_mask & (predicted_labels != SITTING_INDEX)
    ].copy()

    sitting_errors.to_csv(
        OUTPUT_DIR / "sitting_misclassifications.csv",
        index=False,
    )

    sitting_standing = detailed.loc[
        (y_test == SITTING_INDEX)
        | (y_test == STANDING_INDEX)
    ].copy()

    sitting_standing.to_csv(
        OUTPUT_DIR / "sitting_standing_analysis.csv",
        index=False,
    )

    confusion_table = pd.DataFrame(
        matrix,
        index=ACTIVITY_NAMES,
        columns=ACTIVITY_NAMES,
    )

    confusion_table.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    print("\nDiagnostic analysis completed successfully.")

    print(f"Results saved to:\n{OUTPUT_DIR}")

    print("\nFiles generated:")
    print("1. test_probabilities.csv")
    print("2. sitting_misclassifications.csv")
    print("3. sitting_standing_analysis.csv")
    print("4. confusion_matrix.csv")

    print(
        "\nImportant: This script analyzes the existing model. "
        "It does not retrain the model or change its predictions."
    )


if __name__ == "__main__":
    main()
