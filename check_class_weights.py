import pandas as pd
import torch
from pathlib import Path

data_dir = Path("experiments/balanced_data")

activity_names = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

print("\nCLASS WEIGHTS PER CLIENT")
print("=" * 60)

for client_id in range(1, 6):
    file_path = data_dir / f"client_{client_id}.csv"
    df = pd.read_csv(file_path)

    labels = torch.tensor(
        df["activity"].to_numpy() - 1,
        dtype=torch.long,
    )

    counts = torch.bincount(labels, minlength=6).float()

    weights = counts.sum() / (6 * counts.clamp_min(1))
    weights = weights / weights.mean()

    print(f"\nClient {client_id}")

    for index, name in enumerate(activity_names):
        print(
            f"{name:22s} "
            f"Count: {int(counts[index]):4d}  "
            f"Weight: {weights[index].item():.3f}"
        )

print("\nCheck completed.")