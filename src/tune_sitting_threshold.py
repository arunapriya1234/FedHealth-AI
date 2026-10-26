
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_dp_sittingboost_validation_diagnostics"
    / "validation_probabilities.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_dp_sitting_threshold_tuning"
)

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

# Multipliers tested against the Sitting probability.
# Larger values make it easier for Sitting to beat Standing.
SITTING_MULTIPLIERS = [
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.5,
    3.0,
    4.0,
]


# ============================================================
# DECISION ADJUSTMENT
# ============================================================

def adjust_predictions(probabilities, sitting_multiplier):
    """
    Adjust the Sitting-versus-Standing decision only.

    If Sitting's probability multiplied by the chosen factor
    exceeds Standing's probability, select Sitting.

    Otherwise, retain the original six-class argmax prediction.

    This changes decisions only; it does not retrain the model.
    """

    predictions = probabilities.argmax(axis=1)

    sitting_probability = probabilities[:, SITTING_INDEX]
    standing_probability = probabilities[:, STANDING_INDEX]

    choose_sitting = (
        sitting_probability * sitting_multiplier
        > standing_probability
    )

    # Only change a prediction when the original prediction
    # was Standing and the adjusted rule selects Sitting.
    change_mask = (
        (predictions == STANDING_INDEX)
        & choose_sitting
    )

    predictions[change_mask] = SITTING_INDEX

    return predictions


# ============================================================
# METRIC CALCULATION
# ============================================================

def evaluate_predictions(y_true, predictions, multiplier):

    report = classification_report(
        y_true,
        predictions,
        labels=list(range(len(ACTIVITY_NAMES))),
        target_names=ACTIVITY_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=list(range(len(ACTIVITY_NAMES))),
    )

    sitting_total = int(
        (y_true == SITTING_INDEX).sum()
    )

    sitting_correct = int(
        matrix[SITTING_INDEX, SITTING_INDEX]
    )

    standing_total = int(
        (y_true == STANDING_INDEX).sum()
    )

    standing_correct = int(
        matrix[STANDING_INDEX, STANDING_INDEX]
    )

    return {
        "sitting_multiplier": multiplier,
        "accuracy": accuracy_score(y_true, predictions),
        "accuracy_percent": (
            accuracy_score(y_true, predictions) * 100
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            labels=list(range(len(ACTIVITY_NAMES))),
            average="macro",
            zero_division=0,
        ),
        "macro_f1_percent": f1_score(
            y_true,
            predictions,
            labels=list(range(len(ACTIVITY_NAMES))),
            average="macro",
            zero_division=0,
        ) * 100,
        "weighted_f1": f1_score(
            y_true,
            predictions,
            labels=list(range(len(ACTIVITY_NAMES))),
            average="weighted",
            zero_division=0,
        ),
        "sitting_precision": report["Sitting"]["precision"],
        "sitting_recall": report["Sitting"]["recall"],
        "sitting_f1": report["Sitting"]["f1-score"],
        "sitting_correct": sitting_correct,
        "sitting_total": sitting_total,
        "standing_precision": report["Standing"]["precision"],
        "standing_recall": report["Standing"]["recall"],
        "standing_f1": report["Standing"]["f1-score"],
        "standing_correct": standing_correct,
        "standing_total": standing_total,
        "sitting_as_standing": int(
            matrix[SITTING_INDEX, STANDING_INDEX]
        ),
        "standing_as_sitting": int(
            matrix[STANDING_INDEX, SITTING_INDEX]
        ),
        "report": report,
        "confusion_matrix": matrix,
        "predictions": predictions,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FEDHEALTH AI - VALIDATION THRESHOLD TUNING")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Validation probabilities not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "actual_label",
        "probability_walking",
        "probability_walking_upstairs",
        "probability_walking_downstairs",
        "probability_sitting",
        "probability_standing",
        "probability_laying",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    y_true = df["actual_label"].to_numpy(dtype=np.int64)

    if not np.isin(
        y_true,
        np.arange(len(ACTIVITY_NAMES)),
    ).all():
        raise ValueError(
            "Unexpected labels in validation probabilities."
        )

    probability_columns = [
        "probability_walking",
        "probability_walking_upstairs",
        "probability_walking_downstairs",
        "probability_sitting",
        "probability_standing",
        "probability_laying",
    ]

    probabilities = df[probability_columns].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Non-finite values found in prediction probabilities."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Some probability rows do not sum to approximately 1."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Run all candidate adjustments
    # --------------------------------------------------------

    all_results = []
    predictions_by_multiplier = {}

    for multiplier in SITTING_MULTIPLIERS:

        predictions = adjust_predictions(
            probabilities,
            multiplier,
        )

        result = evaluate_predictions(
            y_true,
            predictions,
            multiplier,
        )

        predictions_by_multiplier[multiplier] = predictions

        all_results.append({
            key: value
            for key, value in result.items()
            if key not in [
                "report",
                "confusion_matrix",
                "predictions",
            ]
        })

    results_df = pd.DataFrame(all_results)

    # The unadjusted model is the baseline.
    baseline = results_df.loc[
        results_df["sitting_multiplier"] == 1.0
    ].iloc[0]

    results_df["accuracy_change_vs_baseline_pp"] = (
        results_df["accuracy_percent"]
        - baseline["accuracy_percent"]
    )

    results_df["macro_f1_change_vs_baseline_pp"] = (
        results_df["macro_f1_percent"]
        - baseline["macro_f1_percent"]
    )

    results_df.to_csv(
        OUTPUT_DIR / "threshold_comparison.csv",
        index=False,
    )

    print("\nValidation results for each multiplier:")
    print(
        results_df[
            [
                "sitting_multiplier",
                "accuracy_percent",
                "macro_f1_percent",
                "sitting_recall",
                "sitting_precision",
                "standing_recall",
                "sitting_as_standing",
                "standing_as_sitting",
            ]
        ].to_string(
            index=False,
            formatters={
                "accuracy_percent": "{:.2f}".format,
                "macro_f1_percent": "{:.2f}".format,
                "sitting_recall": "{:.3f}".format,
                "sitting_precision": "{:.3f}".format,
                "standing_recall": "{:.3f}".format,
            },
        )
    )

    # --------------------------------------------------------
    # Choose candidate using validation macro-F1
    # --------------------------------------------------------

    # Select the highest macro-F1, breaking ties by accuracy.
    best_row = results_df.sort_values(
        by=[
            "macro_f1_percent",
            "accuracy_percent",
        ],
        ascending=[False, False],
    ).iloc[0]

    best_multiplier = float(
        best_row["sitting_multiplier"]
    )

    best_predictions = predictions_by_multiplier[
        best_multiplier
    ]

    best_result = evaluate_predictions(
        y_true,
        best_predictions,
        best_multiplier,
    )

    baseline_predictions = predictions_by_multiplier[1.0]

    # --------------------------------------------------------
    # Save baseline and selected candidate diagnostics
    # --------------------------------------------------------

    pd.DataFrame(
        confusion_matrix(
            y_true,
            baseline_predictions,
            labels=list(range(len(ACTIVITY_NAMES))),
        ),
        index=ACTIVITY_NAMES,
        columns=ACTIVITY_NAMES,
    ).to_csv(
        OUTPUT_DIR / "baseline_confusion_matrix.csv"
    )

    pd.DataFrame(
        best_result["confusion_matrix"],
        index=ACTIVITY_NAMES,
        columns=ACTIVITY_NAMES,
    ).to_csv(
        OUTPUT_DIR / "selected_confusion_matrix.csv"
    )

    pd.DataFrame(
        best_result["report"]
    ).transpose().to_csv(
        OUTPUT_DIR / "selected_classification_report.csv"
    )

    comparison_predictions = pd.DataFrame({
        "actual_label": y_true,
        "actual_activity": [
            ACTIVITY_NAMES[i] for i in y_true
        ],
        "baseline_prediction": baseline_predictions,
        "baseline_activity": [
            ACTIVITY_NAMES[i] for i in baseline_predictions
        ],
        "selected_prediction": best_predictions,
        "selected_activity": [
            ACTIVITY_NAMES[i] for i in best_predictions
        ],
        "baseline_correct": baseline_predictions == y_true,
        "selected_correct": best_predictions == y_true,
    })

    comparison_predictions.to_csv(
        OUTPUT_DIR / "selected_prediction_comparison.csv",
        index=False,
    )

    summary = pd.DataFrame([{
        "selected_multiplier": best_multiplier,
        "validation_accuracy_percent": (
            best_result["accuracy"] * 100
        ),
        "validation_macro_f1_percent": (
            best_result["macro_f1"] * 100
        ),
        "validation_weighted_f1_percent": (
            best_result["weighted_f1"] * 100
        ),
        "sitting_recall": best_result["sitting_recall"],
        "sitting_precision": best_result["sitting_precision"],
        "standing_recall": best_result["standing_recall"],
        "sitting_as_standing": best_result["sitting_as_standing"],
        "standing_as_sitting": best_result["standing_as_sitting"],
        "selection_method": (
            "Highest validation macro-F1; accuracy breaks ties"
        ),
    }])

    summary.to_csv(
        OUTPUT_DIR / "selected_threshold_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Print final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SELECTED VALIDATION CANDIDATE")
    print("=" * 70)

    print(f"Selected Sitting multiplier: {best_multiplier:.2f}")
    print(
        f"Validation accuracy: "
        f"{best_result['accuracy'] * 100:.2f}%"
    )
    print(
        f"Validation macro-F1: "
        f"{best_result['macro_f1'] * 100:.2f}%"
    )
    print(
        f"Validation weighted-F1: "
        f"{best_result['weighted_f1'] * 100:.2f}%"
    )
    print(
        f"Sitting recall: "
        f"{best_result['sitting_recall'] * 100:.2f}%"
    )
    print(
        f"Sitting precision: "
        f"{best_result['sitting_precision'] * 100:.2f}%"
    )
    print(
        f"Standing recall: "
        f"{best_result['standing_recall'] * 100:.2f}%"
    )

    print(f"\nResults saved to:\n{OUTPUT_DIR}")

    print("\nGenerated files:")
    print("1. threshold_comparison.csv")
    print("2. selected_threshold_summary.csv")
    print("3. baseline_confusion_matrix.csv")
    print("4. selected_confusion_matrix.csv")
    print("5. selected_classification_report.csv")
    print("6. selected_prediction_comparison.csv")

    print(
        "\nIMPORTANT: The selected multiplier is a validation-only "
        "candidate. It has not been evaluated on the official test "
        "set, and it does not change the trained model."
    )


if __name__ == "__main__":
    main()
