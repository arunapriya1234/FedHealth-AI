
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITIES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

# The Clip 2 evaluation file uses zero-based labels: 0 through 5.
ZERO_BASED_LABEL_MAP = {
    "0": "Walking",
    "1": "Walking Upstairs",
    "2": "Walking Downstairs",
    "3": "Sitting",
    "4": "Standing",
    "5": "Laying",
}

# The original UCI HAR dataset uses labels 1 through 6.
UCI_LABEL_MAP = {
    "1": "Walking",
    "2": "Walking Upstairs",
    "3": "Walking Downstairs",
    "4": "Sitting",
    "5": "Standing",
    "6": "Laying",
}


# ============================================================
# PREDICTION FILES
# ============================================================

PREDICTION_EXPERIMENTS = [
    {
        "name": "Balanced FedAvg (Non-Private, No Dropout)",
        "path": (
            PROJECT_ROOT
            / "experiments"
            / "balanced_fedavg_no_dropout"
            / "test_results.csv"
        ),
        "actual_column": "actual_activity",
        "predicted_column": "predicted_activity",
        "label_format": "activity",
    },
    {
        "name": "DP-FedAvg (Noise 0.7, Clip 2.0, 5 Epochs)",
        "path": (
            PROJECT_ROOT
            / "experiments"
            / "fedhealth_sentinel"
            / "failure_investigation_epochs5"
            / "sample_level_diagnostics.csv"
        ),
        "actual_column": "actual_activity",
        "predicted_column": "predicted_activity",
        "label_format": "activity",
    },
    {
        "name": "DP-FedAvg (Noise 0.7, Clip 2.0, 2 Epochs)",
        "path": (
            PROJECT_ROOT
            / "experiments"
            / "balanced_dp_clip2_test_evaluation"
            / "test_predictions.csv"
        ),
        "actual_column": "actual_label",
        "predicted_column": "predicted_label",
        "label_format": "zero_based",
    },
]


# ============================================================
# METRIC-ONLY RESULTS
# ============================================================
# These are previously recorded metrics. They are not recalculated
# from prediction files by this script.

METRIC_ONLY_MODELS = [
    {
        "name": "Centralized MLP",
        "accuracy": 94.81,
        "macro_f1": 94.78,
        "weighted_f1": None,
        "samples": 2947,
    },
    {
        "name": "FedAvg Non-IID",
        "accuracy": 91.21,
        "macro_f1": 91.10,
        "weighted_f1": None,
        "samples": 2947,
    },
    {
        "name": "Balanced FedAvg",
        "accuracy": 94.06,
        "macro_f1": 94.00,
        "weighted_f1": None,
        "samples": 2947,
    },
    {
        "name": "DP-FedAvg (Noise 1.0)",
        "accuracy": 58.98,
        "macro_f1": 49.00,
        "weighted_f1": None,
        "samples": 2947,
    },
    {
        "name": "DP-FedAvg (Noise 0.7)",
        "accuracy": 65.32,
        "macro_f1": 58.03,
        "weighted_f1": None,
        "samples": 2947,
    },
    {
        "name": "DP-FedAvg (No Dropout, 2 Epochs)",
        "accuracy": 69.66,
        "macro_f1": 62.84,
        "weighted_f1": None,
        "samples": 2947,
    },
]


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_activity(value):
    """Normalize activity names and original UCI labels."""

    text = str(value).strip()

    if text in UCI_LABEL_MAP:
        return UCI_LABEL_MAP[text]

    text = text.replace("_", " ").strip().lower()

    name_map = {
        "walking": "Walking",
        "walking upstairs": "Walking Upstairs",
        "walking downstairs": "Walking Downstairs",
        "sitting": "Sitting",
        "standing": "Standing",
        "laying": "Laying",
        "lying": "Laying",
    }

    return name_map.get(text, text.title())


def normalize_zero_based_label(value):
    """Map model output indices 0-5 to activity names."""

    text = str(value).strip()

    if text not in ZERO_BASED_LABEL_MAP:
        return None

    return ZERO_BASED_LABEL_MAP[text]


# ============================================================
# LOAD AND EVALUATE PREDICTIONS
# ============================================================

def load_prediction_file(config):
    path = config["path"]

    if not path.exists():
        print(f"\nSKIPPED: File not found: {path}")
        return None, None, None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"\nSKIPPED: Could not read {path.name}: {exc}")
        return None, None, None

    actual_column = config["actual_column"]
    predicted_column = config["predicted_column"]

    missing_columns = [
        column
        for column in [actual_column, predicted_column]
        if column not in df.columns
    ]

    if missing_columns:
        print(
            f"\nSKIPPED: {path.name} is missing columns: "
            f"{missing_columns}"
        )
        print(f"Available columns: {list(df.columns)}")
        return None, None, None

    df = df[[actual_column, predicted_column]].dropna().copy()

    if df.empty:
        print(f"\nSKIPPED: No usable predictions in {path.name}")
        return None, None, None

    if config["label_format"] == "zero_based":
        actual = df[actual_column].apply(normalize_zero_based_label)
        predicted = df[predicted_column].apply(normalize_zero_based_label)
    else:
        actual = df[actual_column].apply(normalize_activity)
        predicted = df[predicted_column].apply(normalize_activity)

    valid_activities = set(ACTIVITIES)

    invalid_actual = sorted(
        set(actual.dropna()) - valid_activities
    )
    invalid_predicted = sorted(
        set(predicted.dropna()) - valid_activities
    )

    if (
        actual.isna().any()
        or predicted.isna().any()
        or invalid_actual
        or invalid_predicted
    ):
        print(f"\nSKIPPED: Unrecognized activity labels in {path.name}")
        print(f"Unrecognized actual labels: {invalid_actual}")
        print(f"Unrecognized predicted labels: {invalid_predicted}")
        print("Check the label format in this prediction file.")
        return None, None, None

    actual = actual.reset_index(drop=True)
    predicted = predicted.reset_index(drop=True)

    return actual, predicted, df


def calculate_metrics(model_name, actual, predicted):
    accuracy = accuracy_score(actual, predicted)

    macro_f1 = f1_score(
        actual,
        predicted,
        labels=ACTIVITIES,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        actual,
        predicted,
        labels=ACTIVITIES,
        average="weighted",
        zero_division=0,
    )

    summary = {
        "model": model_name,
        "test_samples": len(actual),
        "accuracy_percent": round(accuracy * 100, 2),
        "macro_f1_percent": round(macro_f1 * 100, 2),
        "weighted_f1_percent": round(weighted_f1 * 100, 2),
        "result_source": "Recalculated from prediction file",
    }

    report = classification_report(
        actual,
        predicted,
        labels=ACTIVITIES,
        output_dict=True,
        zero_division=0,
    )

    activity_rows = []

    for activity in ACTIVITIES:
        activity_rows.append(
            {
                "model": model_name,
                "activity": activity,
                "precision_percent": round(
                    report[activity]["precision"] * 100, 2
                ),
                "recall_percent": round(
                    report[activity]["recall"] * 100, 2
                ),
                "f1_percent": round(
                    report[activity]["f1-score"] * 100, 2
                ),
                "support": int(report[activity]["support"]),
            }
        )

    return summary, activity_rows


# ============================================================
# MAIN
# ============================================================

def main():
    summaries = []
    activity_results = []
    successfully_loaded = {}

    print("=" * 75)
    print("FEDHEALTH AI - DETAILED MODEL COMPARISON")
    print("=" * 75)

    # Evaluate models for which prediction files are available.
    for config in PREDICTION_EXPERIMENTS:
        print(f"\nModel: {config['name']}")

        actual, predicted, original_df = load_prediction_file(config)

        if actual is None:
            continue

        summary, activity_rows = calculate_metrics(
            config["name"],
            actual,
            predicted,
        )

        summaries.append(summary)
        activity_results.extend(activity_rows)

        successfully_loaded[config["name"]] = {
            "actual": actual,
            "predicted": predicted,
        }

        print(f"Test samples: {summary['test_samples']}")
        print(f"Accuracy: {summary['accuracy_percent']:.2f}%")
        print(f"Macro-F1: {summary['macro_f1_percent']:.2f}%")
        print(f"Weighted-F1: {summary['weighted_f1_percent']:.2f}%")

    # Include previously recorded metrics for models without prediction
    # files configured above. These values are not recalculated here.
    prediction_model_names = {
        config["name"] for config in PREDICTION_EXPERIMENTS
    }

    for model in METRIC_ONLY_MODELS:
        if model["name"] in prediction_model_names:
            continue

        summaries.append(
            {
                "model": model["name"],
                "test_samples": model["samples"],
                "accuracy_percent": model["accuracy"],
                "macro_f1_percent": model["macro_f1"],
                "weighted_f1_percent": model["weighted_f1"],
                "result_source": "Previously recorded metric",
            }
        )

    # Compare actual-label sequences for prediction files.
    print("\n" + "=" * 75)
    print("TEST-LABEL COMPARABILITY CHECK")
    print("=" * 75)

    model_names = list(successfully_loaded.keys())

    if len(model_names) >= 2:
        reference_name = model_names[0]
        reference_actual = successfully_loaded[reference_name]["actual"]

        for other_name in model_names[1:]:
            other_actual = successfully_loaded[other_name]["actual"]

            same_length = len(reference_actual) == len(other_actual)
            same_labels = (
                same_length
                and reference_actual.equals(other_actual)
            )

            print(f"\nReference: {reference_name}")
            print(f"Compared with: {other_name}")
            print(f"Same number of samples: {same_length}")
            print(f"Same actual-label sequence: {same_labels}")

            if not same_labels:
                print(
                    "WARNING: Verify that these prediction files use "
                    "the same test samples in the same order before "
                    "making a direct comparison."
                )
            else:
                print(
                    "Actual labels match in order. This is useful "
                    "evidence, but does not independently prove that "
                    "the underlying test samples are identical."
                )
    else:
        print(
            "Not enough prediction files were loaded to compare "
            "actual-label sequences."
        )

    # Save detailed model-level results.
    summary_df = pd.DataFrame(summaries)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by="accuracy_percent",
            ascending=False,
        )

    summary_path = RESULTS_DIR / "detailed_model_comparison.csv"
    summary_df.to_csv(summary_path, index=False)

    # Save per-activity metrics for models with available predictions.
    activity_df = pd.DataFrame(activity_results)

    activity_path = RESULTS_DIR / "activity_level_model_comparison.csv"
    activity_df.to_csv(activity_path, index=False)

    print("\n" + "=" * 75)
    print("OVERALL MODEL COMPARISON")
    print("=" * 75)

    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No model results were available.")

    print("\n" + "=" * 75)
    print("FILES SAVED")
    print("=" * 75)
    print(f"Overall comparison: {summary_path}")
    print(f"Activity comparison: {activity_path}")

    print("\nComparison completed.")


if __name__ == "__main__":
    main()
