from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from models import WearableMLP
from client_dp_balanced import train_private_local_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "experiments" / "balanced_data"

# Save Experiment 2 separately to protect Experiment 1.
RESULTS_DIR = PROJECT_ROOT / "experiments" / "balanced_dp_noise07"

NUM_CLIENTS = 5
ROUNDS = 10
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Experiment 2: lower differential privacy noise.
NOISE_MULTIPLIER = 0.7
MAX_GRAD_NORM = 1.0

DELTA_TOTAL = 1e-5
DELTA_PER_ROUND = DELTA_TOTAL / ROUNDS

DEVICE = torch.device("cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_parameters(model):
    return {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in model.state_dict().items()
    }


def set_parameters(model, parameters):
    current_state = model.state_dict()
    new_state = {
        name: torch.as_tensor(
            parameters[name],
            dtype=current_state[name].dtype,
        )
        for name in current_state
    }
    model.load_state_dict(new_state)


def load_validation_data():
    path = DATA_DIR / "validation.csv"

    if not path.exists():
        raise FileNotFoundError(f"Validation file not found: {path}")

    df = pd.read_csv(path)

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
        predictions = model(X).argmax(dim=1).cpu().numpy()

    labels = y.cpu().numpy()

    return (
        accuracy_score(labels, predictions),
        f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
    )


def fedavg(client_results):
    """Sample-count-weighted averaging of client model parameters."""
    total_samples = sum(
        sample_count
        for _, sample_count, _ in client_results
    )

    if total_samples <= 0:
        raise ValueError("No client samples available.")

    return {
        name: sum(
            parameters[name] * (sample_count / total_samples)
            for parameters, sample_count, _ in client_results
        )
        for name in client_results[0][0]
    }


def main():
    print("=" * 65)
    print("FedHealth AI - Experiment 2: Balanced DP-FedAvg")
    print("=" * 65)
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Rounds: {ROUNDS}")
    print(f"Local epochs: {LOCAL_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Noise multiplier: {NOISE_MULTIPLIER}")
    print(f"Maximum gradient norm: {MAX_GRAD_NORM}")
    print(f"Total delta budget: {DELTA_TOTAL}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")

    X_val, y_val = load_validation_data()

    global_model = WearableMLP().to(DEVICE)
    global_parameters = get_parameters(global_model)

    history = []
    epsilon_totals = {
        client_id: 0.0
        for client_id in range(1, NUM_CLIENTS + 1)
    }

    best_macro_f1 = -1.0
    best_round = 0
    best_parameters = copy.deepcopy(global_parameters)

    for round_number in range(1, ROUNDS + 1):
        print(f"\n{'=' * 65}")
        print(f"Round {round_number}/{ROUNDS}")
        print("=" * 65)

        client_results = []

        for client_id in range(1, NUM_CLIENTS + 1):
            client_model = WearableMLP().to(DEVICE)
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

            client_path = DATA_DIR / f"client_{client_id}.csv"

            if not client_path.exists():
                raise FileNotFoundError(
                    f"Client dataset not found: {client_path}"
                )

            sample_count = len(pd.read_csv(client_path))
            epsilon_totals[client_id] += epsilon

            client_results.append(
                (
                    get_parameters(trained_model),
                    sample_count,
                    epsilon,
                )
            )

            print(
                f"Client {client_id}: "
                f"samples={sample_count}, "
                f"run epsilon={epsilon:.4f}"
            )

        global_parameters = fedavg(client_results)
        set_parameters(global_model, global_parameters)

        val_accuracy, val_macro_f1 = evaluate_validation(
            global_model,
            X_val,
            y_val,
        )

        print(
            f"Validation accuracy: {val_accuracy:.4f} | "
            f"Macro-F1: {val_macro_f1:.4f}"
        )

        history.append({
            "round": round_number,
            "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_macro_f1,
            "noise_multiplier": NOISE_MULTIPLIER,
            "max_grad_norm": MAX_GRAD_NORM,
            "delta_per_round": DELTA_PER_ROUND,
        })

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_round = round_number
            best_parameters = copy.deepcopy(global_parameters)

    # Save Experiment 2 files only in its dedicated directory.
    np.savez(
        RESULTS_DIR / "dp_fedavg_balanced_noise07_global_parameters.npz",
        **global_parameters,
    )

    np.savez(
        RESULTS_DIR / "dp_fedavg_balanced_noise07_best_parameters.npz",
        **best_parameters,
    )

    pd.DataFrame(history).to_csv(
        RESULTS_DIR / "dp_fedavg_balanced_noise07_history.csv",
        index=False,
    )

    privacy_summary = pd.DataFrame([
        {
            "client_id": client_id,
            "rounds_participated": ROUNDS,
            "noise_multiplier": NOISE_MULTIPLIER,
            "delta_total": DELTA_TOTAL,
            "delta_per_round": DELTA_PER_ROUND,
            "sum_of_per_round_epsilon": epsilon_totals[client_id],
            "accounting_note": (
                "Sum of per-round epsilon values is a preliminary "
                "accounting summary, not a verified end-to-end privacy "
                "guarantee. Verify accountant assumptions before "
                "making formal privacy claims."
            ),
        }
        for client_id in range(1, NUM_CLIENTS + 1)
    ])

    privacy_summary.to_csv(
        RESULTS_DIR / "dp_fedavg_balanced_noise07_privacy.csv",
        index=False,
    )

    print("\nExperiment 2 completed.")
    print(f"Best validation round: {best_round}")
    print(f"Best validation accuracy: {max(
        row['validation_accuracy'] for row in history
    ):.4f}")
    print(f"Best validation macro-F1: {best_macro_f1:.4f}")
    print(f"Results saved in: {RESULTS_DIR}")

    print("\nPreliminary privacy summary:")
    print(privacy_summary.to_string(index=False))


if __name__ == "__main__":
    main()