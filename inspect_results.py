from pathlib import Path
import pandas as pd

paths = sorted(Path("results").glob("*test_results.csv"))
paths += sorted(Path("experiments").rglob("*test_results.csv"))

for path in paths:
    print("\n" + "=" * 60)
    print("FILE:", path)

    try:
        df = pd.read_csv(path)
        print(df.to_string(index=False))
    except Exception as exc:
        print("ERROR:", exc)