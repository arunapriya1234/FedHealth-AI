
import os
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from models_no_dropout import WearableMLP
from data_processing import load_dataset


# ---------------- CONFIGURATION ----------------

OUTPUT_DIR = "experiments/balanced_fedavg_no_dropout"
DEVICE = torch.device("cpu")

ACTIVITIES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]


# ---------------- MAIN EVALUATION ----------------

def main():

    print("=" * 60)
    print("EVALUATING NON-PRIVATE FEDAVG WITHOUT DROPOUT")
    print("=" * 60)

    # Load the official UCI HAR dataset
    X_train, y_train, X_test, y_test, features, activities = (
        load_dataset()
    )

    # Convert labels from 1-6 to 0-5 if necessary
    y_test = np.asarray(y_test).astype(np.int64).reshape(-1)

    if y_test.min() == 1 and y_test.max() == 6:
        y_test = y_test - 1

    X_test = np.asarray(X_test, dtype=np.float32)

    print("Official test samples:", len(y_test))
    print("Input features:", X_test.shape[1])

    # ---------------- LOAD BEST MODEL ----------------

    model_path = os.path.join(
        OUTPUT_DIR,
        "best_parameters.npz"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    saved_parameters = np.load(model_path)

    model = WearableMLP(
        input_dim=X_test.shape[1],
        num_classes=6
    ).to(DEVICE)

    model.load_state_dict({
        key: torch.tensor(saved_parameters[key])
        for key in saved_parameters.files
    })

    model.eval()

    print("Best model loaded successfully.")

    # ---------------- PREDICTIONS ----------------

    with torch.no_grad():

        X_tensor = torch.tensor(
            X_test,
            dtype=torch.float32
        ).to(DEVICE)

        outputs = model(X_tensor)

        predictions = (
            outputs.argmax(dim=1)
            .cpu()
            .numpy()
            .astype(np.int64)
        )

    # Ensure labels and predictions are one-dimensional
    y_test = np.asarray(y_test).reshape(-1).astype(np.int64)
    predictions = np.asarray(predictions).reshape(-1).astype(np.int64)

    # ---------------- METRICS ----------------

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\nOFFICIAL TEST RESULTS")
    print("-" * 40)

    print(
        f"Accuracy: {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro-F1: {macro_f1:.4f} "
        f"({macro_f1 * 100:.2f}%)"
    )

    # ---------------- CLASSIFICATION REPORT ----------------

    print("\nCLASSIFICATION REPORT")

    report_text = classification_report(
        y_test,
        predictions,
        labels=list(range(6)),
        target_names=ACTIVITIES,
        zero_division=0,
        digits=4
    )

    print(report_text)

    # ---------------- CREATE OUTPUT DIRECTORY ----------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Convert NumPy integer labels to activity names safely
    actual_names = [
        ACTIVITIES[int(label)]
        for label in y_test
    ]

    predicted_names = [
        ACTIVITIES[int(label)]
        for label in predictions
    ]

    # ---------------- SAVE PREDICTIONS ----------------

    predictions_df = pd.DataFrame({
        "actual_activity": actual_names,
        "predicted_activity": predicted_names
    })

    predictions_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_results.csv"
        ),
        index=False
    )

    # ---------------- SAVE CLASSIFICATION REPORT ----------------

    report_dict = classification_report(
        y_test,
        predictions,
        labels=list(range(6)),
        target_names=ACTIVITIES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(
            OUTPUT_DIR,
            "classification_report.csv"
        ),
        index=True
    )

    # ---------------- SAVE CONFUSION MATRIX ----------------

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(6))
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=ACTIVITIES,
        columns=ACTIVITIES
    )

    confusion_df.index.name = "Actual Activity"
    confusion_df.columns.name = "Predicted Activity"

    confusion_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "confusion_matrix.csv"
        )
    )

    # ---------------- PREDICTED ACTIVITY DISTRIBUTION ----------------

    print("\nPREDICTED ACTIVITY DISTRIBUTION")
    print("-" * 40)

    for class_id, activity in enumerate(ACTIVITIES):

        count = int(
            np.sum(predictions == class_id)
        )

        print(f"{activity}: {count}")

    # ---------------- FINAL STATUS ----------------

    print("\nALL RESULTS SAVED SUCCESSFULLY")

    print("Output directory:", os.path.abspath(OUTPUT_DIR))

    print("1. test_results.csv")
    print("2. classification_report.csv")
    print("3. confusion_matrix.csv")

    print("=" * 60)


if __name__ == "__main__":
    main()
