import sys
from pathlib import Path

import pandas as pd

# Allow imports from the src directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_processing import load_dataset


def test_training_data_shape():
    X_train, _, _, _, _, _ = load_dataset()

    assert X_train.shape == (7352, 561)


def test_testing_data_shape():
    _, _, X_test, _, _, _ = load_dataset()

    assert X_test.shape == (2947, 561)


def test_activity_labels_are_valid():
    _, y_train, _, y_test, _, activities = load_dataset()

    expected_labels = {1, 2, 3, 4, 5, 6}

    assert set(y_train["activity"].unique()) == expected_labels
    assert set(y_test["activity"].unique()) == expected_labels
    assert set(activities["activity"]) == expected_labels


def test_feature_names_count():
    _, _, _, _, features, _ = load_dataset()

    assert len(features) == 561
    assert "feature_name" in features.columns


def test_feature_data_has_no_missing_values():
    X_train, _, X_test, _, _, _ = load_dataset()

    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()