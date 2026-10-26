
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

EXPERIMENTS = [
    {
        "model": "Centralized MLP",
        "file": RESULTS_DIR / "centralized_pytorch_results.csv",
        "format": "metrics",
    },
    {
        "model": "FedAvg (Non-IID)",
        "file": RESULTS_DIR / "fedavg_test_results.csv",
        "format": "metrics",
    },
    {
        "model": "Balanced FedAvg (Non-Private)",
        "file": EXPERIMENTS_DIR / "balanced_fedavg" / "fedavg_balanced_test_results.csv",
        "format": "metrics",
    },
    {
        "model": "DP-FedAvg (Noise 1.0)",
        "file": RESULTS_DIR / "dp_fedavg_balanced_test_results.csv",
        "format": "metrics",
    },
    {
        "model": "DP-FedAvg (Noise 0.7)",
        "file": EXPERIMENTS_DIR / "balanced_dp_noise07" / "dp_fedavg_balanced_noise07_test_results.csv",
        "format": "metrics",
    },
    {
        "model": "Balanced FedAvg (No Dropout)",
        "file": EXPERIMENTS_DIR / "balanced_fedavg_no_dropout" / "test_results.csv",
        "format": "predictions",
    },
    {
        "model": "DP-FedAvg (No Dropout, Noise 0.7)",
        "file": EXPERIMENTS_DIR / "balanced_dp_no_dropout" / "test_results.csv",
        "format": "metrics",
    },
]


def find_metric(row, possible_names):
    for name in possible_names:
        if name in row.index and pd.notna(row[name]):
            try:
                return float(row[name])
            except (TypeError, ValueError):
                pass
    return None


def main():
    records = []

    for experiment in EXPERIMENTS:
        model_name = experiment["model"]
        file_path = experiment["file"]

        print(f"\nChecking: {model_name}")

        if not file_path.exists():
            print(f"WARNING: File not found: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)

            if df.empty:
                print("WARNING: Results file is empty.")
                continue

            if experiment["format"] == "predictions":
                required = {"actual_activity", "predicted_activity"}

                if not required.issubset(df.columns):
                    print(f"WARNING: Missing prediction columns: {required}")
                    continue

                actual = df["actual_activity"].astype(str).str.strip()
                predicted = df["predicted_activity"].astype(str).str.strip()

                accuracy = accuracy_score(actual, predicted)
                macro_f1 = f1_score(
                    actual,
                    predicted,
                    average="macro",
                    zero_division=0,
                )

                test_samples = len(df)

            else:
                row = df.iloc[0]

                accuracy = find_metric(
                    row,
                    ["accuracy", "test_accuracy", "accuracy_score"],
                )
                macro_f1 = find_metric(
                    row,
                    ["macro_f1", "macro_f1_score", "f1_macro"],
                )

                if accuracy is None or macro_f1 is None:
                    print(f"WARNING: Could not identify metrics. Columns: {list(df.columns)}")
                    continue

                if accuracy > 1:
                    accuracy /= 100

                if macro_f1 > 1:
                    macro_f1 /= 100

                test_samples = find_metric(
                    row,
                    ["test_samples", "samples", "n_samples"],
                )

            records.append(
                {
                    "Model": model_name,
                    "Accuracy (%)": round(accuracy * 100, 2),
                    "Macro-F1 (%)": round(macro_f1 * 100, 2),
                    "Test Samples": (
                        int(test_samples) if test_samples is not None else "Not recorded"
                    ),
                    "Source File": str(file_path.relative_to(PROJECT_ROOT)),
                }
            )

            print(
                f"Accuracy: {accuracy * 100:.2f}% | "
                f"Macro-F1: {macro_f1 * 100:.2f}%"
            )

        except Exception as exc:
            print(f"ERROR reading {file_path}: {exc}")

    if not records:
        print("\nNo model results could be compared.")
        return

    comparison = pd.DataFrame(records)
    output_csv = RESULTS_DIR / "all_models_comparison.csv"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_csv, index=False)

    print("\n" + "=" * 75)
    print("FEDHEALTH AI - MODEL COMPARISON")
    print("=" * 75)
    print(comparison.to_string(index=False))
    print(f"\nComparison saved to:\n{output_csv}")
    print(
        "\nIMPORTANT: Confirm that the metric-based files use official test-set "
        "results before reporting the comparison."
    )


if __name__ == "__main__":
    main()
