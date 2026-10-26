from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "experiments" / "balanced_data"

validation = pd.read_csv(DATA_DIR / "validation.csv")
reference_features = [
    column for column in validation.columns
    if column != "activity"
]

print("Validation feature count:", len(reference_features))
print("First five validation features:", reference_features[:5])

all_match = True

for client_id in range(1, 6):
    client_path = DATA_DIR / f"client_{client_id}.csv"
    client_data = pd.read_csv(client_path)

    client_features = [
        column for column in client_data.columns
        if column != "activity"
    ]

    same_order = client_features == reference_features
    valid_labels = sorted(client_data["activity"].unique())

    print(
        f"Client {client_id}: "
        f"feature count={len(client_features)}, "
        f"same names and order={same_order}, "
        f"labels={valid_labels}"
    )

    if not same_order:
        all_match = False

print("\nALL CLIENT FEATURES MATCH VALIDATION:", all_match)