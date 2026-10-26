
from pathlib import Path
import copy
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(SRC_DIR))

from models import WearableMLP
from client_dp import train_private_local_model


# ---------------- Configuration ----------------

NUM_CLIENTS = 5
ROUNDS = 10
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001

NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM = 1.0

# Overall delta budget divided across rounds.
DELTA_TOTAL = 1e-5
DELTA_PER_ROUND = DELTA_TOTAL / ROUNDS

SEED = 42
DEVICE = torch.device("cpu")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_validation_data():
    """Load the validation split; the official test set remains untouched."""
    path = DATA_DIR / "validation.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Validation file not found: {path}\n"
            "Run your existing client-data preparation script first."
        )

    df = pd.read_csv(path)

    if "activity" not in df.columns or df.empty:
        raise ValueError("validation.csv is empty or missing 'activity'.")

    X = torch.tensor(
        df.drop(columns=["activity"]).to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    y = torch.tensor(
        df["activity"].to_numpy(dtype=np.int64) - 1,
        dtype=torch.long,
    )

    return X, y


def evaluate_validation(model, X, y):
    model.eval()
    with torch.no_grad():
        predictions = model(X.to(DEVICE)).argmax(dim=1).cpu().numpy()

    labels = y.numpy()
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return accuracy, macro_f1


def get_model_parameters(model):
    return {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in model.state_dict().items()
    }


def set_model_parameters(model, parameters):
    state = {
        name: torch.as_tensor(
            parameters[name],
            dtype=value.dtype,
        )
        for name, value in model.state_dict().items()
    }
    model.load_state_dict(state)


def fedavg(client_results):
    """Average model parameters, weighted by each client's sample count."""
    total_samples = sum(sample_count for _, sample_count, _ in client_results)

    if total_samples <= 0:
        raise ValueError("No client samples available for aggregation.")

    aggregated = {}

    for name in client_results[0][0]:
        aggregated[name] = sum(
            parameters[name] * (sample_count / total_samples)
            for parameters, sample_count, _ in client_results
        )

    return aggregated


def main():
    print("=" * 65)
    print("FedHealth AI — Differentially Private Federated Averaging")
    print("=" * 65)
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Rounds: {ROUNDS}")
    print(f"Local epochs per round: {LOCAL_EPOCHS}")
    print(f"Noise multiplier: {NOISE_MULTIPLIER}")
    print(f"Maximum gradient norm: {MAX_GRAD_NORM}")
    print(f"Total delta budget: {DELTA_TOTAL}")
    print(f"Delta per local run: {DELTA_PER_ROUND}")
    print()
    print("Note: this is a simulated federated experiment.")
    print("Each client trains privately; the server aggregates model weights.")
    print()

    X_val, y_val = load_validation_data()

    global_model = WearableMLP().to(DEVICE)
    global_parameters = get_model_parameters(global_model)

    history = []
    client_epsilon_totals = {client_id: 0.0 for client_id in range(1, NUM_CLIENTS + 1)}
    best_macro_f1 = -1.0
    best_round = 0
    best_parameters = copy.deepcopy(global_parameters)

    for round_number in range(1, ROUNDS + 1):
        print("\n" + "=" * 65)
        print(f"DP-FedAvg round {round_number}/{ROUNDS}")
        print("=" * 65)

        client_results = []

        for client_id in range(1, NUM_CLIENTS + 1):
            client_model = WearableMLP().to(DEVICE)
            set_model_parameters(client_model, global_parameters)

            trained_model, round_epsilon = train_private_local_model(
                client_id=client_id,
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                noise_multiplier=NOISE_MULTIPLIER,
                max_grad_norm=MAX_GRAD_NORM,
                delta=DELTA_PER_ROUND,
                model=client_model,
            )

            client_path = DATA_DIR / f"client_{client_id}.csv"
            client_df = pd.read_csv(client_path)
            sample_count = len(client_df)

            client_epsilon_totals[client_id] += round_epsilon

            client_results.append(
                (
                    get_model_parameters(trained_model),
                    sample_count,
                    round_epsilon,
                )
            )

            print(
                f"Client {client_id}: samples={sample_count}, "
                f"this-run epsilon={round_epsilon:.4f}, "
                f"conservative cumulative epsilon="
                f"{client_epsilon_totals[client_id]:.4f}"
            )

        # Aggregate privately trained client models.
        global_parameters = fedavg(client_results)
        set_model_parameters(global_model, global_parameters)

        val_accuracy, val_macro_f1 = evaluate_validation(
            global_model,
            X_val,
            y_val,
        )

        print(
            f"\nRound {round_number} validation: "
            f"accuracy={val_accuracy:.4f}, "
            f"macro-F1={val_macro_f1:.4f}"
        )

        history.append(
            {
                "round": round_number,
                "validation_accuracy": val_accuracy,
                "validation_macro_f1": val_macro_f1,
                "noise_multiplier": NOISE_MULTIPLIER,
                "max_grad_norm": MAX_GRAD_NORM,
                "delta_per_round": DELTA_PER_ROUND,
            }
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_round = round_number
            best_parameters = copy.deepcopy(global_parameters)

    # Save the final-round and best-validation models separately.
    np.savez(
        RESULTS_DIR / "dp_fedavg_global_parameters.npz",
        **global_parameters,
    )
    np.savez(
        RESULTS_DIR / "dp_fedavg_best_validation_parameters.npz",
        **best_parameters,
    )

    pd.DataFrame(history).to_csv(
        RESULTS_DIR / "dp_fedavg_training_history.csv",
        index=False,
    )

    privacy_summary = pd.DataFrame(
        [
            {
                "client_id": client_id,
                "rounds_participated": ROUNDS,
                "noise_multiplier": NOISE_MULTIPLIER,
                "delta_total": DELTA_TOTAL,
                "delta_per_round": DELTA_PER_ROUND,
                "sum_of_per_round_epsilon": epsilon_total,
                "accounting_note": (
                    "Conservative basic composition: sum of per-round "
                    "epsilon values; per-round delta is delta_total / rounds. "
                    "Verify the guarantee against the exact sampling and "
                    "privacy assumptions before making formal claims."
                ),
            }
            for client_id, epsilon_total in client_epsilon_totals.items()
        ]
    )

    privacy_summary.to_csv(
        RESULTS_DIR / "dp_fedavg_privacy_summary.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("DP-FedAvg training completed.")
    print(f"Best validation round: {best_round}")
    print(f"Best validation macro-F1: {best_macro_f1:.4f}")
    print(f"Final round: {ROUNDS}")
    print(f"Saved outputs in: {RESULTS_DIR}")
    print("=" * 65)
    print("\nPer-client privacy accounting summary:")
    print(privacy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
