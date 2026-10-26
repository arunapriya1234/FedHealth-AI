
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
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALIDATION_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_data"
    / "validation.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_dp_sittingboost_epochs5_run"
    / "best_parameters.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_dp_sittingboost_validation_diagnostics"
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
    print("FEDHEALTH AI")
    print("SITTING-BOOST VALIDATION DIAGNOSTICS")
    print("=" * 65)

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    for path in [VALIDATION_PATH, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------------

    print("\nLoading validation dataset...")

    validation_df = pd.read_csv(VALIDATION_PATH)

    if "activity" not in validation_df.columns:
        raise ValueError(
            "The validation CSV does not contain an "
            "'activity' column."
        )

    feature_columns = [
        column
        for column in validation_df.columns
        if column != "activity"
    ]

    if len(feature_columns) != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} feature columns, "
            f"but found {len(feature_columns)}."
        )

    X_validation = validation_df[feature_columns].to_numpy(
        dtype=np.float32
    )

    y_original = validation_df["activity"].to_numpy(
        dtype=np.int64
    )

    # The project uses original activity labels 1-6.
    y_validation = y_original - 1

    if len(y_validation) == 0:
        raise ValueError("The validation dataset is empty.")

    if not np.isin(
        y_validation,
        np.arange(NUM_CLASSES),
    ).all():
        raise ValueError(
            "Unexpected labels found in the validation dataset."
        )

    print(f"Validation samples: {len(y_validation)}")
    print(f"Features per sample: {X_validation.shape[1]}")

    # --------------------------------------------------------
    # Load Sitting-Boost model
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
            "Saved parameter names do not match the model "
            "architecture."
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
    # Generate predictions and probabilities
    # --------------------------------------------------------

    print("\nEvaluating validation samples...")

    X_tensor = torch.tensor(
        X_validation,
        dtype=torch.float32,
    )

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.softmax(
            logits,
            dim=1,
        ).numpy()

    predictions = probabilities.argmax(axis=1)

    # --------------------------------------------------------
    # Calculate overall metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_validation,
        predictions,
        labels=list(range(NUM_CLASSES)),
        average="weighted",
        zero_division=0,
    )

    report_text = classification_report(
        y_validation,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=ACTIVITY_NAMES,
        zero_division=0,
    )

    report_dict = classification_report(
        y_validation,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=ACTIVITY_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=list(range(NUM_CLASSES)),
    )

    # --------------------------------------------------------
    # Sitting diagnostics
    # --------------------------------------------------------

    sitting_mask = y_validation == SITTING_INDEX

    sitting_correct_mask = (
        sitting_mask
        & (predictions == SITTING_INDEX)
    )

    sitting_as_standing_mask = (
        sitting_mask
        & (predictions == STANDING_INDEX)
    )

    sitting_total = int(sitting_mask.sum())
    sitting_correct = int(sitting_correct_mask.sum())
    sitting_as_standing = int(sitting_as_standing_mask.sum())

    sitting_recall = report_dict["Sitting"]["recall"]
    sitting_precision = report_dict["Sitting"]["precision"]
    sitting_f1 = report_dict["Sitting"]["f1-score"]

    print("\n" + "=" * 65)
    print("VALIDATION RESULTS")
    print("=" * 65)

    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    print(f"Macro-F1: {macro_f1 * 100:.2f}%")
    print(f"Weighted-F1: {weighted_f1 * 100:.2f}%")

    print("\nClassification report:")
    print(report_text)

    print("Confusion matrix:")
    print(matrix)

    print("\n" + "=" * 65)
    print("SITTING VS STANDING")
    print("=" * 65)

    print(f"Actual Sitting samples: {sitting_total}")
    print(f"Correct Sitting predictions: {sitting_correct}")
    print(f"Sitting predicted as Standing: {sitting_as_standing}")
    print(f"Sitting precision: {sitting_precision * 100:.2f}%")
    print(f"Sitting recall: {sitting_recall * 100:.2f}%")
    print(f"Sitting F1-score: {sitting_f1 * 100:.2f}%")

    if sitting_total > 0:

        sitting_prob = probabilities[
            sitting_mask,
            SITTING_INDEX,
        ]

        standing_prob = probabilities[
            sitting_mask,
            STANDING_INDEX,
        ]

        print(
            "Mean Sitting probability for actual Sitting: "
            f"{sitting_prob.mean() * 100:.2f}%"
        )

        print(
            "Mean Standing probability for actual Sitting: "
            f"{standing_prob.mean() * 100:.2f}%"
        )

        print(
            "Sitting probability greater than Standing: "
            f"{int((sitting_prob > standing_prob).sum())}"
        )

        print(
            "Sitting probability lower than Standing: "
            f"{int((sitting_prob < standing_prob).sum())}"
        )

    # --------------------------------------------------------
    # Save probability-level diagnostics
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    details = {
        "actual_label": y_validation,
        "actual_activity": [
            ACTIVITY_NAMES[i] for i in y_validation
        ],
        "predicted_label": predictions,
        "predicted_activity": [
            ACTIVITY_NAMES[i] for i in predictions
        ],
        "correct": predictions == y_validation,
        "predicted_confidence": probabilities.max(axis=1),
        "sitting_probability": probabilities[:, SITTING_INDEX],
        "standing_probability": probabilities[:, STANDING_INDEX],
    }

    for i, activity in enumerate(ACTIVITY_NAMES):
        details[
            f"probability_{activity.lower().replace(' ', '_')}"
        ] = probabilities[:, i]

    details_df = pd.DataFrame(details)

    details_df.to_csv(
        OUTPUT_DIR / "validation_probabilities.csv",
        index=False,
    )

    pd.DataFrame(
        report_dict
    ).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    pd.DataFrame(
        matrix,
        index=ACTIVITY_NAMES,
        columns=ACTIVITY_NAMES,
    ).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    pd.DataFrame([{
        "model": "Sitting-Boost DP-FedAvg (5 Local Epochs)",
        "validation_samples": len(y_validation),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "macro_f1": macro_f1,
        "macro_f1_percent": macro_f1 * 100,
        "weighted_f1": weighted_f1,
        "weighted_f1_percent": weighted_f1 * 100,
        "sitting_total": sitting_total,
        "sitting_correct": sitting_correct,
        "sitting_as_standing": sitting_as_standing,
        "sitting_precision": sitting_precision,
        "sitting_recall": sitting_recall,
        "sitting_f1": sitting_f1,
        "model_file": str(MODEL_PATH),
    }]).to_csv(
        OUTPUT_DIR / "validation_summary.csv",
        index=False,
    )

    print("\nValidation diagnostic completed successfully!")
    print(f"Results saved to:\n{OUTPUT_DIR}")

    print("\nFiles generated:")
    print("1. validation_summary.csv")
    print("2. classification_report.csv")
    print("3. confusion_matrix.csv")
    print("4. validation_probabilities.csv")

    print(
        "\nNote: These results use validation data to help "
        "analyze and select model settings. The official test "
        "results remain separate."
    )


if __name__ == "__main__":
    main()
