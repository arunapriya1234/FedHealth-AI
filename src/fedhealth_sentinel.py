
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, classification_report

from models_no_dropout import WearableMLP


# ============================================================
# FEDHEALTH SENTINEL - RELIABILITY MONITOR
# ============================================================

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

# Predictions below this confidence are flagged for review.
CONFIDENCE_THRESHOLD = 0.60


def main():
    print("=" * 65)
    print("FEDHEALTH SENTINEL")
    print("Privacy-Aware Activity Reliability Monitor")
    print("=" * 65)

    # 1. Check required files.
    x_path = TEST_DIR / "X_test.txt"
    y_path = TEST_DIR / "y_test.txt"

    for path in [x_path, y_path, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # 2. Load the official UCI HAR test dataset.
    X_test = np.loadtxt(x_path, dtype=np.float32)
    y_test = np.loadtxt(y_path, dtype=np.int64) - 1

    if X_test.ndim != 2 or X_test.shape[1] != NUM_FEATURES:
        raise ValueError(f"Unexpected feature shape: {X_test.shape}")

    if len(X_test) != len(y_test):
        raise ValueError("Feature and label counts do not match.")

    if not np.isfinite(X_test).all():
        raise ValueError("Test features contain NaN or infinite values.")

    if not np.isin(y_test, np.arange(NUM_CLASSES)).all():
        raise ValueError("Unexpected activity labels.")

    # 3. Load the trained model.
    model = WearableMLP(
        input_dim=NUM_FEATURES,
        num_classes=NUM_CLASSES,
    )

    saved_parameters = np.load(MODEL_PATH)
    current_state = model.state_dict()

    if set(saved_parameters.files) != set(current_state.keys()):
        raise ValueError("Saved parameters do not match the model.")

    loaded_state = {}

    for key, current_tensor in current_state.items():
        saved_array = saved_parameters[key]

        if saved_array.shape != tuple(current_tensor.shape):
            raise ValueError(f"Parameter shape mismatch: {key}")

        loaded_state[key] = torch.as_tensor(
            saved_array,
            dtype=current_tensor.dtype,
        )

    model.load_state_dict(loaded_state)
    model.eval()

    # 4. Generate probabilities and predictions.
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).numpy()

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == y_test

    # 5. Flag uncertain predictions.
    needs_review = confidence < CONFIDENCE_THRESHOLD

    # 6. Create a detailed prediction report.
    results = pd.DataFrame({
        "actual_activity": [ACTIVITY_NAMES[i] for i in y_test],
        "predicted_activity": [
            ACTIVITY_NAMES[i] for i in predictions
        ],
        "confidence": confidence,
        "confidence_percent": confidence * 100,
        "correct": correct,
        "needs_review": needs_review,
    })

    for class_index, activity in enumerate(ACTIVITY_NAMES):
        results[f"probability_{activity.lower().replace(' ', '_')}"] = (
            probabilities[:, class_index]
        )

    # 7. Overall reliability summary.
    overall_accuracy = accuracy_score(y_test, predictions)
    review_rate = needs_review.mean()
    accuracy_accepted = (
        accuracy_score(y_test[~needs_review], predictions[~needs_review])
        if (~needs_review).any()
        else np.nan
    )

    summary = pd.DataFrame([{
        "model": "DP-FedAvg clipping_norm_2.0",
        "test_samples": len(y_test),
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_percent": overall_accuracy * 100,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "predictions_flagged": int(needs_review.sum()),
        "review_rate_percent": review_rate * 100,
        "accepted_predictions": int((~needs_review).sum()),
        "accepted_prediction_accuracy": accuracy_accepted,
        "accepted_prediction_accuracy_percent": (
            accuracy_accepted * 100
            if not np.isnan(accuracy_accepted)
            else np.nan
        ),
        "mean_confidence_percent": confidence.mean() * 100,
        "model_file": str(MODEL_PATH),
    }])

    # 8. Analyze confidence thresholds.
    threshold_rows = []

    for threshold in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        accepted = confidence >= threshold

        threshold_rows.append({
            "confidence_threshold": threshold,
            "accepted_predictions": int(accepted.sum()),
            "coverage_percent": accepted.mean() * 100,
            "accepted_accuracy_percent": (
                accuracy_score(y_test[accepted], predictions[accepted]) * 100
                if accepted.any()
                else np.nan
            ),
        })

    threshold_report = pd.DataFrame(threshold_rows)

    # 9. Activity-level performance.
    activity_report = pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=ACTIVITY_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    # 10. Save all outputs separately.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(OUTPUT_DIR / "prediction_reliability.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "reliability_summary.csv", index=False)
    threshold_report.to_csv(
        OUTPUT_DIR / "confidence_threshold_analysis.csv",
        index=False,
    )
    activity_report.to_csv(
        OUTPUT_DIR / "activity_classification_report.csv"
    )

    # 11. Display a concise summary.
    print(f"\nTest samples: {len(y_test)}")
    print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Mean confidence: {confidence.mean() * 100:.2f}%")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"Predictions flagged for review: {needs_review.sum()}")
    print(f"Review rate: {review_rate * 100:.2f}%")

    if not np.isnan(accuracy_accepted):
        print(
            "Accuracy among accepted predictions: "
            f"{accuracy_accepted * 100:.2f}%"
        )

    print("\nConfidence threshold analysis:")
    print(threshold_report.to_string(index=False))

    print("\nActivity-level classification report:")
    print(activity_report.to_string())

    print(f"\nReports saved to: {OUTPUT_DIR}")
    print("Reliability analysis completed successfully.")


if __name__ == "__main__":
    main()
