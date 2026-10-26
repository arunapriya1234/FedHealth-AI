
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import confusion_matrix, accuracy_score
from models_no_dropout import WearableMLP


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_ROOT / "data" / "raw" / "test"

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_dp_clip2_run"
    / "best_parameters.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "fedhealth_sentinel"
    / "failure_investigation"
)

ACTIVITIES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

NUM_FEATURES = 561
NUM_CLASSES = 6


def main():
    print("=" * 65)
    print("FEDHEALTH SENTINEL")
    print("ACTIVITY FAILURE INVESTIGATOR")
    print("=" * 65)

    x_path = TEST_DIR / "X_test.txt"
    y_path = TEST_DIR / "y_test.txt"

    for path in [x_path, y_path, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    X = np.loadtxt(x_path, dtype=np.float32)
    y = np.loadtxt(y_path, dtype=np.int64) - 1

    if X.ndim != 2 or X.shape[1] != NUM_FEATURES:
        raise ValueError(f"Unexpected feature shape: {X.shape}")

    if len(X) != len(y):
        raise ValueError("Feature and label counts do not match.")

    if not np.isfinite(X).all():
        raise ValueError("Test features contain non-finite values.")

    if not np.isin(y, np.arange(NUM_CLASSES)).all():
        raise ValueError("Unexpected activity labels.")

    model = WearableMLP(
        input_dim=NUM_FEATURES,
        num_classes=NUM_CLASSES,
    )

    saved = np.load(MODEL_PATH)
    current_state = model.state_dict()

    if set(saved.files) != set(current_state.keys()):
        raise ValueError("Saved parameters do not match model architecture.")

    state = {}

    for key, tensor in current_state.items():
        array = saved[key]

        if array.shape != tuple(tensor.shape):
            raise ValueError(f"Shape mismatch: {key}")

        state[key] = torch.as_tensor(
            array,
            dtype=tensor.dtype,
        )

    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).numpy()

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == y

    # --------------------------------------------------------
    # 1. Overall diagnostics
    # --------------------------------------------------------

    print(f"\nTest samples: {len(y)}")
    print(f"Overall accuracy: {accuracy_score(y, predictions) * 100:.2f}%")
    print(f"Mean confidence: {confidence.mean() * 100:.2f}%")

    print(
        "Mean confidence on correct predictions: "
        f"{confidence[correct].mean() * 100:.2f}%"
    )

    if (~correct).any():
        print(
            "Mean confidence on incorrect predictions: "
            f"{confidence[~correct].mean() * 100:.2f}%"
        )

    # --------------------------------------------------------
    # 2. Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y,
        predictions,
        labels=list(range(NUM_CLASSES)),
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=[f"Actual {name}" for name in ACTIVITIES],
        columns=[f"Predicted {name}" for name in ACTIVITIES],
    )

    print("\nConfusion matrix:")
    print(confusion_df.to_string())

    # --------------------------------------------------------
    # 3. Activity-level confidence diagnostics
    # --------------------------------------------------------

    activity_rows = []

    for i, activity in enumerate(ACTIVITIES):
        mask = y == i
        activity_predictions = predictions[mask]

        row = {
            "activity": activity,
            "sample_count": int(mask.sum()),
            "correct_predictions": int(correct[mask].sum()),
            "accuracy_percent": (
                accuracy_score(y[mask], predictions[mask]) * 100
            ),
            "mean_top_confidence_percent": (
                confidence[mask].mean() * 100
            ),
        }

        for j, predicted_activity in enumerate(ACTIVITIES):
            row[
                f"mean_probability_{predicted_activity.lower().replace(' ', '_')}_percent"
            ] = probabilities[mask, j].mean() * 100

        activity_rows.append(row)

    activity_df = pd.DataFrame(activity_rows)

    print("\nActivity-level diagnostics:")
    print(
        activity_df[
            [
                "activity",
                "sample_count",
                "correct_predictions",
                "accuracy_percent",
                "mean_top_confidence_percent",
            ]
        ].round(2).to_string(index=False)
    )

    # --------------------------------------------------------
    # 4. Specifically investigate Sitting
    # --------------------------------------------------------

    sitting_index = ACTIVITIES.index("Sitting")
    actual_sitting = y == sitting_index

    sitting_rows = []

    for i, activity in enumerate(ACTIVITIES):
        sitting_rows.append({
            "actual_activity": "Sitting",
            "probability_for_activity": activity,
            "mean_probability_percent": (
                probabilities[actual_sitting, i].mean() * 100
            ),
        })

    sitting_df = pd.DataFrame(sitting_rows)

    print("\nSITTING INVESTIGATION")
    print(
        "Actual Sitting samples:",
        int(actual_sitting.sum()),
    )
    print(
        "Correctly predicted Sitting samples:",
        int((predictions[actual_sitting] == sitting_index).sum()),
    )
    print(
        "Most common prediction for actual Sitting:",
        ACTIVITIES[
            np.bincount(
                predictions[actual_sitting],
                minlength=NUM_CLASSES,
            ).argmax()
        ],
    )
    print("\nMean probabilities for actual Sitting samples:")
    print(sitting_df.to_string(index=False))

    # --------------------------------------------------------
    # 5. Confidence threshold diagnostics
    # --------------------------------------------------------

    threshold_rows = []

    for threshold in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        accepted = confidence >= threshold

        threshold_rows.append({
            "threshold": threshold,
            "accepted_samples": int(accepted.sum()),
            "coverage_percent": accepted.mean() * 100,
            "accepted_accuracy_percent": (
                accuracy_score(y[accepted], predictions[accepted]) * 100
                if accepted.any()
                else np.nan
            ),
        })

    threshold_df = pd.DataFrame(threshold_rows)

    print("\nConfidence threshold diagnostics:")
    print(threshold_df.round(2).to_string(index=False))

    # --------------------------------------------------------
    # 6. Save reports
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    confusion_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    activity_df.to_csv(OUTPUT_DIR / "activity_diagnostics.csv", index=False)
    sitting_df.to_csv(OUTPUT_DIR / "sitting_diagnostics.csv", index=False)
    threshold_df.to_csv(OUTPUT_DIR / "threshold_diagnostics.csv", index=False)

    predictions_df = pd.DataFrame({
        "actual_activity": [ACTIVITIES[i] for i in y],
        "predicted_activity": [ACTIVITIES[i] for i in predictions],
        "confidence_percent": confidence * 100,
        "correct": correct,
    })

    predictions_df.to_csv(
        OUTPUT_DIR / "sample_level_diagnostics.csv",
        index=False,
    )

    print(f"\nReports saved to: {OUTPUT_DIR}")
    print("Activity Failure Investigator completed successfully.")


if __name__ == "__main__":
    main()
