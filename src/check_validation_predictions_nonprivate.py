
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models_no_dropout import WearableMLP

DATA_PATH = ROOT / "experiments" / "balanced_data" / "validation.csv"
MODEL_PATH = (
    ROOT / "experiments" / "balanced_fedavg_no_dropout" / "best_parameters.npz"
)

ACTIVITY_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]


def main():
    print("=" * 65)
    print("FEDHEALTH AI - NON-PRIVATE FEDAVG VALIDATION EVALUATION")
    print("=" * 65)

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["activity"]).to_numpy(dtype=np.float32)
    y = df["activity"].to_numpy(dtype=np.int64) - 1

    if X.shape[1] != 561:
        raise ValueError(f"Expected 561 features, found {X.shape[1]}.")

    model = WearableMLP(input_dim=561, num_classes=6)

    saved = np.load(MODEL_PATH, allow_pickle=False)

    model_state = model.state_dict()

    if set(saved.files) != set(model_state.keys()):
        raise ValueError(
            f"Saved parameters do not match model architecture.\n"
            f"Saved: {saved.files}\nExpected: {list(model_state.keys())}"
        )

    with torch.no_grad():
        for name, parameter in model_state.items():
            parameter.copy_(
                torch.as_tensor(saved[name], dtype=parameter.dtype)
            )

    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).numpy()

    predictions = probabilities.argmax(axis=1)

    print(f"\nValidation samples: {len(y)}")
    print(f"Validation accuracy: {accuracy_score(y, predictions) * 100:.2f}%")

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y,
            predictions,
            labels=list(range(6)),
            target_names=ACTIVITY_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y, predictions, labels=list(range(6)))

    print("CONFUSION MATRIX")
    print(pd.DataFrame(
        matrix,
        index=ACTIVITY_NAMES,
        columns=ACTIVITY_NAMES,
    ))

    sitting_index = ACTIVITY_NAMES.index("Sitting")
    actual_sitting = y == sitting_index

    print("\nSITTING ANALYSIS")
    print(f"Actual Sitting samples: {actual_sitting.sum()}")
    print(f"Correct Sitting predictions: {((predictions == sitting_index) & actual_sitting).sum()}")
    print(f"Total predictions of Sitting: {(predictions == sitting_index).sum()}")

    if actual_sitting.any():
        print("\nAverage probabilities for actual Sitting samples:")

        for i, name in enumerate(ACTIVITY_NAMES):
            print(f"{name}: {probabilities[actual_sitting, i].mean() * 100:.2f}%")

        print("\nActual Sitting samples predicted as:")
        for i, name in enumerate(ACTIVITY_NAMES):
            count = int((predictions[actual_sitting] == i).sum())
            print(f"{name}: {count}")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
