
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "experiments" / "balanced_data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "balanced_dp_no_dropout"

sys.path.insert(0, str(SRC_DIR))

from models_no_dropout import WearableMLP
from client_dp_balanced_no_dropout import train_private_local_model
from data_processing import load_dataset


NUM_CLIENTS = 5
ROUNDS = 10
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NOISE_MULTIPLIER = 0.7
MAX_GRAD_NORM = 1.0
DELTA_TOTAL = 1e-5
DELTA_PER_ROUND = 1e-6
DEVICE = torch.device("cpu")

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def get_parameters(model):
    return {
        key: value.detach().cpu().numpy().copy()
        for key, value in model.state_dict().items()
    }


def set_parameters(model, parameters):
    state = model.state_dict()

    for key in state:
        state[key] = torch.tensor(
            parameters[key],
            dtype=state[key].dtype,
        )

    model.load_state_dict(state)


def fedavg(client_parameters, client_sizes):
    total = sum(client_sizes)
    averaged = {}

    for key in client_parameters[0]:
        averaged[key] = sum(
            client_parameters[i][key] * client_sizes[i]
            for i in range(len(client_parameters))
        ) / total

    return averaged


def evaluate(model, X, y):
    model.eval()

    with torch.no_grad():
        predictions = model(X).argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(y, predictions)
    macro_f1 = f1_score(
        y,
        predictions,
        labels=list(range(6)),
        average="macro",
        zero_division=0,
    )

    return accuracy, macro_f1


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    # Load validation data only. The official test set is not used
    # to select the best training round.
    validation_df = pd.read_csv(DATA_DIR / "validation.csv")

    X_val = torch.tensor(
        validation_df.drop(columns=["activity"]).to_numpy(dtype="float32"),
        dtype=torch.float32,
    ).to(DEVICE)

    y_val = (
        validation_df["activity"].to_numpy(dtype="int64") - 1
    )

    global_model = WearableMLP(
        input_dim=X_val.shape[1],
        num_classes=6,
    ).to(DEVICE)

    global_parameters = get_parameters(global_model)

    history = []
    privacy_rows = []
    best_macro_f1 = -1.0
    best_round = 0
    best_parameters = None

    print("=" * 65)
    print("BALANCED DP-FEDAVG WITHOUT DROPOUT")
    print("=" * 65)
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Rounds: {ROUNDS}")
    print(f"Local epochs: {LOCAL_EPOCHS}")
    print(f"Noise multiplier: {NOISE_MULTIPLIER}")
    print(f"Gradient clipping norm: {MAX_GRAD_NORM}")
    print(f"Output directory: {RESULTS_DIR}")

    for round_number in range(1, ROUNDS + 1):
        print(f"\n{'=' * 20} ROUND {round_number}/{ROUNDS} {'=' * 20}")

        client_parameters = []
        client_sizes = []
        client_epsilons = []

        for client_id in range(1, NUM_CLIENTS + 1):
            client_model = WearableMLP(
                input_dim=X_val.shape[1],
                num_classes=6,
            ).to(DEVICE)

            set_parameters(client_model, global_parameters)

            trained_model, epsilon = train_private_local_model(
                client_id=client_id,
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                noise_multiplier=NOISE_MULTIPLIER,
                max_grad_norm=MAX_GRAD_NORM,
                delta=DELTA_PER_ROUND,
                model=client_model,
            )

            client_df = pd.read_csv(
                DATA_DIR / f"client_{client_id}.csv"
            )

            client_parameters.append(get_parameters(trained_model))
            client_sizes.append(len(client_df))
            client_epsilons.append(epsilon)

        global_parameters = fedavg(client_parameters, client_sizes)
        set_parameters(global_model, global_parameters)

        val_accuracy, val_macro_f1 = evaluate(
            global_model,
            X_val,
            y_val,
        )

        history.append({
            "round": round_number,
            "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_macro_f1,
            "noise_multiplier": NOISE_MULTIPLIER,
            "max_grad_norm": MAX_GRAD_NORM,
        })

        privacy_rows.append({
            "round": round_number,
            "noise_multiplier": NOISE_MULTIPLIER,
            "delta_per_local_run": DELTA_PER_ROUND,
            "max_local_epsilon": max(client_epsilons),
            "sum_of_local_epsilon_values": sum(client_epsilons),
            "note": (
                "Local epsilon values are not an end-to-end federated "
                "privacy guarantee. Formal composition and accounting "
                "must be verified separately."
            ),
        })

        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation macro-F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_round = round_number
            best_parameters = get_parameters(global_model)

            np.savez(
                RESULTS_DIR / "best_parameters.npz",
                **best_parameters,
            )

            print("New best validation model saved.")

        np.savez(
            RESULTS_DIR / "global_parameters.npz",
            **get_parameters(global_model),
        )

        pd.DataFrame(history).to_csv(
            RESULTS_DIR / "history.csv",
            index=False,
        )

        pd.DataFrame(privacy_rows).to_csv(
            RESULTS_DIR / "privacy_diagnostics.csv",
            index=False,
        )

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)
    print(f"Best round: {best_round}")
    print(f"Best validation macro-F1: {best_macro_f1:.4f}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
