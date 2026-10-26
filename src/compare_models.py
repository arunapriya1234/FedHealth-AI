
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_single_result(path, expected_model_name):
    """Load and validate one model's final test results."""
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)

    required = {"test_samples", "accuracy", "macro_f1"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )

    if len(df) != 1:
        raise ValueError(
            f"{path.name} should contain exactly one final result row."
        )

    result = df.iloc[0]

    return {
        "model": expected_model_name,
        "test_samples": int(result["test_samples"]),
        "test_accuracy": float(result["accuracy"]),
        "macro_f1": float(result["macro_f1"]),
    }


def main():
    centralized_path = RESULTS_DIR / "centralized_pytorch_results.csv"
    federated_path = RESULTS_DIR / "fedavg_test_results.csv"

    centralized = load_single_result(
        centralized_path,
        "Centralized PyTorch MLP",
    )
    federated = load_single_result(
        federated_path,
        "FedAvg (best validation-selected model)",
    )

    if centralized["test_samples"] != federated["test_samples"]:
        raise ValueError(
            "The models were evaluated on different numbers of test samples. "
            "Check the evaluation setup before comparing them."
        )

    comparison = pd.DataFrame([centralized, federated])

    comparison["test_accuracy_percent"] = (
        comparison["test_accuracy"] * 100
    )
    comparison["macro_f1_percent"] = comparison["macro_f1"] * 100

    output_csv = RESULTS_DIR / "model_comparison.csv"
    comparison.to_csv(output_csv, index=False)

    accuracy_gap = (
        centralized["test_accuracy"] - federated["test_accuracy"]
    ) * 100
    f1_gap = (
        centralized["macro_f1"] - federated["macro_f1"]
    ) * 100

    print("\nFEDHEALTH AI - FINAL TEST COMPARISON")
    print("-" * 48)
    print(
        comparison[
            [
                "model",
                "test_samples",
                "test_accuracy_percent",
                "macro_f1_percent",
            ]
        ].to_string(index=False, formatters={
            "test_accuracy_percent": "{:.2f}%".format,
            "macro_f1_percent": "{:.2f}%".format,
        })
    )

    print(f"\nCentralized accuracy advantage: {accuracy_gap:.2f} percentage points")
    print(f"Centralized macro-F1 advantage: {f1_gap:.2f} percentage points")

    # Plot accuracy and macro-F1 using the same test set.
    metrics = ["test_accuracy_percent", "macro_f1_percent"]
    labels = ["Test accuracy", "Macro-F1"]
    x_positions = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    for index, (_, row) in enumerate(comparison.iterrows()):
        values = [row[metric] for metric in metrics]
        positions = [
            position + (index - 0.5) * width
            for position in x_positions
        ]

        bars = ax.bar(
            positions,
            values,
            width,
            label=row["model"],
        )
        ax.bar_label(bars, fmt="%.2f%%", padding=3)

    ax.set_title("Centralized vs Federated Learning — UCI HAR Test Set")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(list(x_positions), labels)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_png = RESULTS_DIR / "model_comparison.png"
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

    print(f"\nComparison CSV saved to: {output_csv}")
    print(f"Comparison chart saved to: {output_png}")


if __name__ == "__main__":
    main()
