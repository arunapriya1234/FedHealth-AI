import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

csv_path = RESULTS_DIR / "all_models_comparison.csv"
df = pd.read_csv(csv_path)

required_columns = ["Model", "Accuracy (%)", "Macro-F1 (%)"]

if not all(column in df.columns for column in required_columns):
    raise ValueError(
        f"CSV must contain these columns: {required_columns}"
    )

# Graph 1: Test Accuracy Comparison
plt.figure(figsize=(11, 6))
bars = plt.bar(df["Model"], df["Accuracy (%)"])

plt.title("FedHealth AI - Test Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 105)
plt.xticks(rotation=20, ha="right")

for bar, value in zip(bars, df["Accuracy (%)"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
accuracy_path = RESULTS_DIR / "model_accuracy_comparison.png"
plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
plt.show()

# Graph 2: Macro-F1 Comparison
plt.figure(figsize=(11, 6))
bars = plt.bar(df["Model"], df["Macro-F1 (%)"])

plt.title("FedHealth AI - Macro-F1 Comparison")
plt.xlabel("Model")
plt.ylabel("Macro-F1 (%)")
plt.ylim(0, 105)
plt.xticks(rotation=20, ha="right")

for bar, value in zip(bars, df["Macro-F1 (%)"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
f1_path = RESULTS_DIR / "model_macro_f1_comparison.png"
plt.savefig(f1_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nBoth graphs saved successfully:")
print(accuracy_path)
print(f1_path)
