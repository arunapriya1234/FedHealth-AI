
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report

# Allow importing the model from src
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models_no_dropout import WearableMLP

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "experiments" / "balanced_data" / "validation.csv"

MODEL_PATH = (
    ROOT
    / "experiments"
    / "balanced_dp_epochs5_run"
    / "best_parameters.npz"
)

ACTIVITY_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

NUM_CLASSES = len(ACTIVITY_NAMES)
SITTING_CLASS = 3
STANDING_CLASS = 4


def main():
    # Check that the required files exist
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}"
        )

    # Load validation data
    df = pd.read_csv(DATA_PATH)

    if "activity" not in df.columns:
        raise ValueError(
            "The validation CSV must contain an 'activity' column."
        )

    X = df.drop(columns=["activity"]).to_numpy(dtype=np.float32)
    y = df["activity"].to_numpy(dtype=np.int64) - 1

    # Validate the dataset
    if X.shape[1] != 561:
        raise ValueError(
            f"Expected 561 features, but found {X.shape[1]}."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "The validation features contain NaN or infinite values."
        )

    if not np.isin(y, np.arange(NUM_CLASSES)).all():
        raise ValueError(
            "Unexpected activity labels found in validation data."
        )

    # Load saved model parameters
    saved = np.load(MODEL_PATH)

    print("Saved parameter names:", saved.files)

    # Initialize model
    model = WearableMLP(
        input_dim=561,
        num_classes=NUM_CLASSES,
    )

    state_dict = model.state_dict()

    # Verify parameter count
    if len(saved.files) != len(state_dict):
        raise ValueError(
            "Saved parameter count does not match the model.\n"
            f"Saved arrays: {len(saved.files)}\n"
            f"Model parameters: {len(state_dict)}"
        )

    # Load saved arrays into model parameters
    for (name, current_tensor), saved_name in zip(
        state_dict.items(),
        saved.files,
    ):
        array = saved[saved_name]

        if tuple(array.shape) != tuple(current_tensor.shape):
            raise ValueError(
                f"Shape mismatch for {name}:\n"
                f"Expected: {tuple(current_tensor.shape)}\n"
                f"Found in {saved_name}: {tuple(array.shape)}"
            )

        state_dict[name] = torch.as_tensor(
            array,
            dtype=current_tensor.dtype,
        )

    model.load_state_dict(state_dict)
    model.eval()

    # Generate predictions and confidence scores
    with torch.no_grad():
        X_tensor = torch.from_numpy(X)

        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1)

        predictions = probabilities.argmax(dim=1).numpy()
        sitting_confidence = probabilities[
            :, SITTING_CLASS
        ].numpy()

    # Overall validation performance
    accuracy = (predictions == y).mean() * 100

    print("\n" + "=" * 55)
    print("VALIDATION RESULTS")
    print("=" * 55)

    print("Validation samples:", len(y))
    print(f"Validation accuracy: {accuracy:.2f}%")

    print("\nClassification report:")

    print(
        classification_report(
            y,
            predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=ACTIVITY_NAMES,
            zero_division=0,
            digits=4,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")

    print(
        confusion_matrix(
            y,
            predictions,
            labels=list(range(NUM_CLASSES)),
        )
    )

    # Analyze Sitting confidence
    actual_sitting = y == SITTING_CLASS
    actual_standing = y == STANDING_CLASS

    print("\n" + "=" * 55)
    print("SITTING CONFIDENCE ANALYSIS")
    print("=" * 55)

    print(
        "Actual Sitting samples:",
        int(actual_sitting.sum()),
    )

    print(
        "Actual Standing samples:",
        int(actual_standing.sum()),
    )

    if actual_sitting.any():
        print(
            "Average Sitting probability for actual Sitting samples:",
            f"{sitting_confidence[actual_sitting].mean() * 100:.2f}%",
        )

    if actual_standing.any():
        print(
            "Average Sitting probability for actual Standing samples:",
            f"{sitting_confidence[actual_standing].mean() * 100:.2f}%",
        )

    print(
        "Highest Sitting probability across validation samples:",
        f"{sitting_confidence.max() * 100:.2f}%",
    )

    print(
        "Validation samples predicted as Sitting:",
        int((predictions == SITTING_CLASS).sum()),
    )

    # Show the most common predictions for actual Sitting samples
    if actual_sitting.any():
        sitting_predictions = predictions[actual_sitting]
        counts = np.bincount(
            sitting_predictions,
            minlength=NUM_CLASSES,
        )

        print("\nPredictions for actual Sitting samples:")

        for class_id, count in enumerate(counts):
            if count > 0:
                print(
                    f"  Predicted as {ACTIVITY_NAMES[class_id]}: "
                    f"{count} samples"
                )

    # Show average predicted probabilities for actual Sitting samples
    if actual_sitting.any():
        mean_probabilities = probabilities.numpy()[
            actual_sitting
        ].mean(axis=0)

        print(
            "\nAverage predicted probabilities "
            "for actual Sitting samples:"
        )

        for class_id, activity_name in enumerate(ACTIVITY_NAMES):
            print(
                f"  {activity_name}: "
                f"{mean_probabilities[class_id] * 100:.2f}%"
            )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()

