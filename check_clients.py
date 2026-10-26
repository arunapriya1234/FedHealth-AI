from pathlib import Path
import pandas as pd

data_dir = Path("experiments/balanced_data")

print("=" * 50)
print("FEDHEALTH AI - CLIENT DATA VALIDATION")
print("=" * 50)

files = sorted(data_dir.glob("client_*.csv"))

for file in files:
    df = pd.read_csv(file)

    print(f"\nFile: {file.name}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Activity column exists: {'activity' in df.columns}")

    if "activity" in df.columns:
        print(f"Activity labels: {sorted(df['activity'].unique().tolist())}")
        print(f"Feature count: {len(df.columns) - 1}")

print("\nTotal client files:", len(files))
print("\nValidation complete!")