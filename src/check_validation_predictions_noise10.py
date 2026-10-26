
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models_no_dropout import WearableMLP

# Paths
DATA_PATH = ROOT / "experiments" / "balanced_data" / "validation.csv"
MODEL_PATH = ROOT / "experiments" / "balanced_dp_noise_10" / "best_parameters.npz"

# Activity labels
CLASS_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]


def main():
    print("=" * 70)
    print("FEDHEALTH AI - NOISE MULTIPLIER 1.0 VALIDATION EVALUATION")
    print("=" * 70)

    # Check files
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Validation dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model parameters not found: {MODEL_PATH}")

    # Load validation dataset
    df = pd.read_csv(DATA_PATH)

    if "activity" not in df.columns:
        raise ValueError("The validation CSV must contain an 'activity' column.")

    X = df.drop(columns=["activity"]).to_numpy(dtype=np.float32)
    y = df["activity"].to_numpy(dtype=np.int64) - 1

    if X.shape[1] != 561:
        raise ValueError(f"Expected 561 features, but found {X.shape[1]}.")

    if not np.isfinite(X).all():
        raise ValueError("Validation features contain NaN or infinite values.")

    print(f"\nValidation samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")

    # Create model
    model = WearableMLP(input_dim=X.shape[1], num_classes=6)

    # Load saved model parameters
    saved_parameters = np.load(MODEL_PATH, allow_pickle=False)

    state_dict = model.state_dict()

    if set(saved_parameters.files) != set(state_dict.keys()):
        raise ValueError(
            "Saved model parameter names do not match the model architecture.\n"
            f"Saved: {saved_parameters.files}\n"
            f"Expected: {list(state_dict.keys())}"
        )

    with torch.no_grad():
        for name, parameter in state_dict.items():
            parameter.copy_(
                torch.as_tensor(
                    saved_parameters[name],
                    dtype=parameter.dtype,
                )
            )

    model.eval()

    # Predict
    with torch.no_grad():
        probabilities = torch.softmax(
            model(torch.tensor(X, dtype=torch.float32)),
            dim=1,
        ).numpy()

    y_pred = probabilities.argmax(axis=1)

    # Overall metrics
    accuracy = accuracy_score(y, y_pred)
    print(f"\nValidation accuracy: {accuracy * 100:.2f}%")

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y,
            y_pred,
            labels=list(range(6)),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=list(range(6)))

    print("CONFUSION MATRIX")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))

    # Sitting-specific diagnostics
    sitting_index = CLASS_NAMES.index("Sitting")
    actual_sitting = y == sitting_index

    print("\nSITTING CONFIDENCE ANALYSIS")
    print(f"Actual Sitting samples: {int(actual_sitting.sum())}")
    print(f"Samples predicted as Sitting: {int((y_pred == sitting_index).sum())}")

    if actual_sitting.any():
        sitting_probabilities = probabilities[actual_sitting]

        print(
            "Average Sitting probability for actual Sitting samples: "
            f"{sitting_probabilities[:, sitting_index].mean() * 100:.2f}%"
        )

        print("\nAverage predicted probabilities for actual Sitting samples:")

        for i, class_name in enumerate(CLASS_NAMES):
            print(
                f"{class_name}: "
                f"{sitting_probabilities[:, i].mean() * 100:.2f}%"
            )

        print("\nPredictions for actual Sitting samples:")

        for i, class_name in enumerate(CLASS_NAMES):
            count = int((y_pred[actual_sitting] == i).sum())
            print(f"Predicted as {class_name}: {count}")

    # Save results
    output_dir = ROOT / "experiments" / "balanced_dp_noise_10_validation_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "actual_activity": [CLASS_NAMES[i] for i in y],
            "predicted_activity": [CLASS_NAMES[i] for i in y_pred],
            "confidence": probabilities.max(axis=1),
            "sitting_probability": probabilities[:, sitting_index],
            "correct": y == y_pred,
        }
    ).to_csv(output_dir / "validation_predictions.csv", index=False)

    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(output_dir / "confusion_matrix.csv")

    report = classification_report(
        y,
        y_pred,
        labels=list(range(6)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        output_dir / "classification_report.csv"
    )

    print(f"\nResults saved to: {output_dir}")
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
