
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "experiments" / "balanced_data"

files = [
    data_dir / f"client_{i}.csv"
    for i in range(1, 6)
]
files.append(data_dir / "validation.csv")

for file_path in files:
    df = pd.read_csv(file_path)

    # Activity 4 represents Sitting in the original UCI HAR labels.
    sitting = df[df["activity"] == 4]
    features = sitting.drop(columns=["activity"])

    values = features.to_numpy(dtype=float)

    print(f"\n{'=' * 50}")
    print(f"File: {file_path.name}")
    print(f"Total samples: {len(df)}")
    print(f"Sitting samples: {len(sitting)}")

    if len(sitting) > 0:
        print(f"Feature minimum: {np.nanmin(values):.4f}")
        print(f"Feature maximum: {np.nanmax(values):.4f}")
        print(f"Feature mean: {np.nanmean(values):.4f}")
        print(f"NaN values: {np.isnan(values).sum()}")
        print(f"Infinite values: {np.isinf(values).sum()}")
