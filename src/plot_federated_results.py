
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

HISTORY_FILE = (
    RESULTS_DIR / "federated_training_history_detailed.csv"
)


def plot_federated_results():
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"Training history not found: {HISTORY_FILE}\n"
            "Run python src/server.py first."
        )

    df = pd.read_csv(HISTORY_FILE)

    required_columns = {
        "round",
        "test_accuracy",
        "test_macro_f1",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"History file must contain: {required_columns}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Plot test accuracy by federated round.
    plt.figure(figsize=(8, 5))
    plt.plot(
        df["round"],
        df["test_accuracy"] * 100,
        marker="o",
        label="Test Accuracy (%)",
    )
    plt.xticks(df["round"])
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy (%)")
    plt.title("FedHealth AI - Global Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    accuracy_path = RESULTS_DIR / "federated_accuracy_by_round.png"
    plt.savefig(accuracy_path, dpi=300)
    plt.close()

    # Plot macro-F1 by federated round.
    plt.figure(figsize=(8, 5))
    plt.plot(
        df["round"],
        df["test_macro_f1"],
        marker="o",
        label="Macro-F1",
    )
    plt.xticks(df["round"])
    plt.xlabel("Federated Round")
    plt.ylabel("Macro-F1 Score")
    plt.title("FedHealth AI - Global Macro-F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    f1_path = RESULTS_DIR / "federated_f1_by_round.png"
    plt.savefig(f1_path, dpi=300)
    plt.close()

    print("Federated result graphs created successfully!")
    print(f"Accuracy graph: {accuracy_path}")
    print(f"Macro-F1 graph: {f1_path}")


if __name__ == "__main__":
    plot_federated_results()
