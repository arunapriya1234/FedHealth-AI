
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from data_processing import load_dataset


def train_centralized_baseline():
    X_train, y_train, X_test, y_test, _, activities = load_dataset()

    # Labels must be one-dimensional for scikit-learn
    y_train = y_train["activity"]
    y_test = y_test["activity"]

    # Scale using training data only to avoid test-data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initial neural network baseline
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=50,
        random_state=42,
        early_stopping=True,
    )

    print("Training centralized baseline...")
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print("\nCENTRALIZED BASELINE RESULTS")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=activities["activity"].tolist(),
            target_names=activities["activity_name"].tolist(),
            zero_division=0,
        )
    )

    # Save the measured results
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([{
        "method": "Centralized MLP",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }])

    results.to_csv(
        results_dir / "baseline_results.csv",
        index=False,
    )

    print(f"Results saved to: {results_dir / 'baseline_results.csv'}")


if __name__ == "__main__":
    train_centralized_baseline()
