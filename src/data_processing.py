
from pathlib import Path
import pandas as pd

# Project and dataset paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_dataset():
    """Load the official UCI-HAR training and testing data."""

    X_train = pd.read_csv(
        DATA_DIR / "train" / "X_train.txt",
        sep=r"\s+",
        header=None,
    )

    y_train = pd.read_csv(
        DATA_DIR / "train" / "y_train.txt",
        header=None,
        names=["activity"],
    )

    X_test = pd.read_csv(
        DATA_DIR / "test" / "X_test.txt",
        sep=r"\s+",
        header=None,
    )

    y_test = pd.read_csv(
        DATA_DIR / "test" / "y_test.txt",
        header=None,
        names=["activity"],
    )

    features = pd.read_csv(
        DATA_DIR / "features.txt",
        sep=r"\s+",
        header=None,
        names=["id", "feature_name"],
    )

    activities = pd.read_csv(
        DATA_DIR / "activity_labels.txt",
        sep=r"\s+",
        header=None,
        names=["activity", "activity_name"],
    )

    return X_train, y_train, X_test, y_test, features, activities


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, features, activities = load_dataset()

    print("FEDHEALTH AI - DATASET INSPECTION")
    print("-" * 40)
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])
    print("Number of features:", X_train.shape[1])
    print("Number of activity classes:", y_train["activity"].nunique())
    print("Training data missing values:", X_train.isna().sum().sum())
    print("Testing data missing values:", X_test.isna().sum().sum())

    print("\nTraining activity distribution:")
    print(y_train["activity"].value_counts().sort_index())

    print("\nActivity label mapping:")
    print(activities.to_string(index=False))

    assert X_train.shape == (7352, 561), "Unexpected training feature dimensions"
    assert X_test.shape == (2947, 561), "Unexpected testing feature dimensions"
    assert len(features) == 561, "Unexpected number of features"

    print("\nDataset loaded and validated successfully!")
