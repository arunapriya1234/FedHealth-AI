import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_dp_epochs5_run"
    / "best_parameters.npz"
)

TEST_FEATURES_PATH = PROJECT_ROOT / "data" / "raw" / "test" / "X_test.txt"
TEST_LABELS_PATH = PROJECT_ROOT / "data" / "raw" / "test" / "y_test.txt"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "fedhealth_sentinel"
    / "failure_investigation_epochs5"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

NUM_FEATURES = 561
NUM_CLASSES = 6

CONFIDENCE_THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ============================================================
# 2. DEFINE THE MODEL
# ============================================================

class ActivityMLP(nn.Module):
    """
    Feed-forward neural network used by FedHealth AI.

    Architecture:
        561 input features
        128 hidden neurons
        64 hidden neurons
        6 activity classes
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(561, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

def load_test_data():
    print("\nLoading UCI HAR test data...")

    if not TEST_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Test feature file not found:\n{TEST_FEATURES_PATH}"
        )

    if not TEST_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Test label file not found:\n{TEST_LABELS_PATH}"
        )

    X_test = np.loadtxt(TEST_FEATURES_PATH, dtype=np.float32)
    y_test = np.loadtxt(TEST_LABELS_PATH, dtype=np.int64)

    if X_test.ndim != 2:
        raise ValueError(
            f"Expected a 2D feature array, got shape {X_test.shape}"
        )

    if X_test.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} features, "
            f"but found {X_test.shape[1]}."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "The number of feature rows does not match the number of labels."
        )

    if not np.isfinite(X_test).all():
        raise ValueError("Test features contain NaN or infinite values.")

    # UCI HAR labels are 1 through 6.
    # The model uses class indices 0 through 5.
    if not np.isin(y_test, [1, 2, 3, 4, 5, 6]).all():
        raise ValueError(
            f"Unexpected activity labels: {np.unique(y_test)}"
        )

    y_test = y_test - 1

    print(f"Test samples: {len(X_test)}")
    print(f"Number of features: {X_test.shape[1]}")
    print(f"Activity classes: {np.unique(y_test) + 1}")

    return X_test, y_test


# ============================================================
# 4. LOAD TRAINED MODEL PARAMETERS
# ============================================================

def load_trained_model():
    print("\nLoading 5-epoch differentially private model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model parameter file not found:\n{MODEL_PATH}"
        )

    model = ActivityMLP().to(DEVICE)

    with np.load(MODEL_PATH, allow_pickle=False) as saved_parameters:
        saved_names = list(saved_parameters.files)

        print("Saved parameter names:", saved_names)

        expected_names = list(model.state_dict().keys())

        if set(saved_names) != set(expected_names):
            raise ValueError(
                "The saved model parameters do not match the model "
                "architecture.\n"
                f"Expected: {expected_names}\n"
                f"Found: {saved_names}"
            )

        state_dict = {
            name: torch.from_numpy(
                saved_parameters[name].copy()
            ).to(dtype=model.state_dict()[name].dtype)
            for name in expected_names
        }

    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# 5. MAKE PREDICTIONS
# ============================================================

def make_predictions(model, X_test):
    print("\nGenerating predictions...")

    X_tensor = torch.from_numpy(X_test).float().to(DEVICE)

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1)

    probabilities = probabilities.cpu().numpy()

    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)

    print("Predictions generated successfully.")

    return predictions, probabilities, confidence


# ============================================================
# 6. SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual {name}" for name in CLASS_NAMES],
        columns=[f"Predicted {name}" for name in CLASS_NAMES],
    )

    output_path = OUTPUT_DIR / "confusion_matrix.csv"
    matrix_df.to_csv(output_path)

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(matrix)

    print(f"Saved: {output_path}")

    return matrix


# ============================================================
# 7. ACTIVITY-LEVEL DIAGNOSTICS
# ============================================================

def investigate_activities(y_true, y_pred, probabilities, confidence):
    records = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = y_true == class_index
        sample_count = int(mask.sum())

        if sample_count == 0:
            continue

        correct_count = int(
            np.sum(y_pred[mask] == class_index)
        )

        class_accuracy = correct_count / sample_count

        mean_confidence = float(np.mean(confidence[mask]))

        predicted_counts = np.bincount(
            y_pred[mask],
            minlength=NUM_CLASSES,
        )

        most_common_prediction = int(np.argmax(predicted_counts))

        record = {
            "activity": class_name,
            "actual_samples": sample_count,
            "correct_predictions": correct_count,
            "incorrect_predictions": sample_count - correct_count,
            "accuracy": class_accuracy,
            "mean_top_confidence": mean_confidence,
            "most_common_prediction": CLASS_NAMES[
                most_common_prediction
            ],
        }

        for predicted_index, predicted_name in enumerate(CLASS_NAMES):
            record[
                f"predicted_as_{predicted_name}"
            ] = int(predicted_counts[predicted_index])

            record[
                f"mean_probability_{predicted_name}"
            ] = float(
                probabilities[mask, predicted_index].mean()
            )

        records.append(record)

    activity_df = pd.DataFrame(records)

    output_path = OUTPUT_DIR / "activity_diagnostics.csv"
    activity_df.to_csv(output_path, index=False)

    print("\nACTIVITY-LEVEL DIAGNOSTICS")
    print("=" * 65)

    print(
        activity_df[
            [
                "activity",
                "actual_samples",
                "correct_predictions",
                "accuracy",
                "mean_top_confidence",
                "most_common_prediction",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {output_path}")

    return activity_df


# ============================================================
# 8. INVESTIGATE SITTING SPECIFICALLY
# ============================================================

def investigate_sitting(y_true, y_pred, probabilities, confidence):
    sitting_index = CLASS_NAMES.index("Sitting")
    standing_index = CLASS_NAMES.index("Standing")

    sitting_mask = y_true == sitting_index
    standing_mask = y_true == standing_index

    sitting_count = int(sitting_mask.sum())
    standing_count = int(standing_mask.sum())

    if sitting_count == 0:
        print("\nNo actual Sitting samples were found.")
        return

    sitting_correct = int(
        np.sum(y_pred[sitting_mask] == sitting_index)
    )

    sitting_as_standing = int(
        np.sum(y_pred[sitting_mask] == standing_index)
    )

    sitting_as_laying = int(
        np.sum(
            y_pred[sitting_mask]
            == CLASS_NAMES.index("Laying")
        )
    )

    sitting_recall = sitting_correct / sitting_count

    print("\nSITTING-CLASS INVESTIGATION")
    print("=" * 65)

    print(f"Actual Sitting samples: {sitting_count}")
    print(f"Actual Standing samples: {standing_count}")

    print(
        f"Sitting samples correctly predicted: "
        f"{sitting_correct}"
    )

    print(
        f"Sitting recall: {sitting_recall:.2%}"
    )

    print(
        f"Sitting predicted as Standing: "
        f"{sitting_as_standing}"
    )

    print(
        f"Sitting predicted as Laying: "
        f"{sitting_as_laying}"
    )

    print(
        "\nAverage probabilities for actual Sitting samples:"
    )

    sitting_probabilities = probabilities[sitting_mask].mean(axis=0)

    for index, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_name}: "
            f"{sitting_probabilities[index]:.2%}"
        )

    if standing_count > 0:
        standing_probabilities = probabilities[standing_mask].mean(
            axis=0
        )

        print(
            "\nAverage Sitting probability for actual Sitting: "
            f"{sitting_probabilities[sitting_index]:.2%}"
        )

        print(
            "Average Sitting probability for actual Standing: "
            f"{standing_probabilities[sitting_index]:.2%}"
        )

    print(
        "\nHighest Sitting probability across all samples: "
        f"{probabilities[:, sitting_index].max():.2%}"
    )

    print(
        "Samples predicted as Sitting: "
        f"{int(np.sum(y_pred == sitting_index))}"
    )

    sitting_df = pd.DataFrame(
        {
            "actual_activity": np.array(CLASS_NAMES)[y_true[sitting_mask]],
            "predicted_activity": np.array(CLASS_NAMES)[
                y_pred[sitting_mask]
            ],
            "top_confidence": confidence[sitting_mask],
            "sitting_probability": probabilities[
                sitting_mask, sitting_index
            ],
            "standing_probability": probabilities[
                sitting_mask, standing_index
            ],
            "laying_probability": probabilities[
                sitting_mask, CLASS_NAMES.index("Laying")
            ],
        }
    )

    sitting_path = OUTPUT_DIR / "sitting_diagnostics.csv"
    sitting_df.to_csv(sitting_path, index=False)

    print(f"\nSaved: {sitting_path}")


# ============================================================
# 9. CONFIDENCE THRESHOLD ANALYSIS
# ============================================================

def investigate_confidence_thresholds(
    y_true,
    y_pred,
    confidence,
):
    records = []

    for threshold in CONFIDENCE_THRESHOLDS:
        accepted_mask = confidence >= threshold

        accepted_count = int(accepted_mask.sum())
        rejected_count = int((~accepted_mask).sum())

        coverage = accepted_count / len(y_true)

        if accepted_count > 0:
            accepted_accuracy = accuracy_score(
                y_true[accepted_mask],
                y_pred[accepted_mask],
            )
        else:
            accepted_accuracy = np.nan

        records.append(
            {
                "confidence_threshold": threshold,
                "total_samples": len(y_true),
                "accepted_samples": accepted_count,
                "flagged_for_review": rejected_count,
                "coverage": coverage,
                "accepted_accuracy": accepted_accuracy,
            }
        )

    threshold_df = pd.DataFrame(records)

    output_path = OUTPUT_DIR / "threshold_diagnostics.csv"
    threshold_df.to_csv(output_path, index=False)

    print("\nCONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 65)

    print(threshold_df.to_string(index=False))

    print(f"\nSaved: {output_path}")

    return threshold_df


# ============================================================
# 10. SAVE SAMPLE-LEVEL DIAGNOSTICS
# ============================================================

def save_sample_diagnostics(
    y_true,
    y_pred,
    probabilities,
    confidence,
):
    correct = y_true == y_pred

    predicted_labels = np.array(CLASS_NAMES)[y_pred]
    actual_labels = np.array(CLASS_NAMES)[y_true]

    sample_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true)),
            "actual_activity": actual_labels,
            "predicted_activity": predicted_labels,
            "correct_prediction": correct,
            "top_confidence": confidence,
            "needs_review_below_60_percent": confidence < 0.60,
        }
    )

    for index, class_name in enumerate(CLASS_NAMES):
        sample_df[
            f"probability_{class_name}"
        ] = probabilities[:, index]

    output_path = OUTPUT_DIR / "sample_level_diagnostics.csv"
    sample_df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

def main():
    print("=" * 65)
    print("FEDHEALTH AI — 5-EPOCH FAILURE INVESTIGATION")
    print("=" * 65)

    print(f"Model path: {MODEL_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    X_test, y_test = load_test_data()

    model = load_trained_model()

    y_pred, probabilities, confidence = make_predictions(
        model,
        X_test,
    )

    overall_accuracy = accuracy_score(y_test, y_pred)

    print("\nOVERALL TEST RESULTS")
    print("=" * 65)

    print(f"Test samples: {len(y_test)}")
    print(f"Correct predictions: {int(np.sum(y_test == y_pred))}")
    print(f"Incorrect predictions: {int(np.sum(y_test != y_pred))}")
    print(f"Test accuracy: {overall_accuracy:.2%}")
    print(f"Mean confidence: {confidence.mean():.2%}")

    print("\nCLASSIFICATION REPORT")
    print("=" * 65)

    report_text = classification_report(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print(report_text)

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()

    report_path = OUTPUT_DIR / "classification_report.csv"
    report_df.to_csv(report_path)

    print(f"\nSaved: {report_path}")

    save_confusion_matrix(y_test, y_pred)

    investigate_activities(
        y_test,
        y_pred,
        probabilities,
        confidence,
    )

    investigate_sitting(
        y_test,
        y_pred,
        probabilities,
        confidence,
    )

    investigate_confidence_thresholds(
        y_test,
        y_pred,
        confidence,
    )

    save_sample_diagnostics(
        y_test,
        y_pred,
        probabilities,
        confidence,
    )

    summary_df = pd.DataFrame(
        [
            {
                "model": "DP-FedAvg",
                "local_epochs": 5,
                "test_samples": len(y_test),
                "test_accuracy": overall_accuracy,
                "mean_confidence": float(confidence.mean()),
                "correct_predictions": int(np.sum(y_test == y_pred)),
                "incorrect_predictions": int(np.sum(y_test != y_pred)),
                "sitting_recall": float(
                    np.sum(
                        (y_test == CLASS_NAMES.index("Sitting"))
                        & (y_pred == CLASS_NAMES.index("Sitting"))
                    )
                    / max(
                        1,
                        np.sum(
                            y_test == CLASS_NAMES.index("Sitting")
                        ),
                    )
                ),
            }
        ]
    )

    summary_path = OUTPUT_DIR / "investigation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nINVESTIGATION COMPLETE")
    print("=" * 65)
    print(f"Results saved in:\n{OUTPUT_DIR}")
    print("\nGenerated files:")
    print("1. investigation_summary.csv")
    print("2. classification_report.csv")
    print("3. confusion_matrix.csv")
    print("4. activity_diagnostics.csv")
    print("5. sitting_diagnostics.csv")
    print("6. threshold_diagnostics.csv")
    print("7. sample_level_diagnostics.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR DURING INVESTIGATION")
        print("=" * 65)
        print(error)
        sys.exit(1)