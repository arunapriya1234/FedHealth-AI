
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from opacus.accountants import RDPAccountant

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "experiments" / "balanced_data"

# Separate output folder to preserve previous experiment results.
RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_dp_clip2_run"
)

sys.path.insert(0, str(SRC_DIR))

from models_no_dropout import WearableMLP
from client_dp_balanced_no_dropout import train_private_local_model


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

NUM_CLIENTS = 5
ROUNDS = 10
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NOISE_MULTIPLIER = 0.7
MAX_GRAD_NORM = 2.0

# Reporting delta for reference.
# This is not automatically the end-to-end federated delta.
DELTA_TOTAL = 1e-5

# Delta used to calculate each client's cumulative RDP epsilon.
DELTA_PER_CLIENT = 1e-6

DEVICE = torch.device("cpu")


# ============================================================
# MODEL PARAMETER HELPERS
# ============================================================

def get_parameters(model):
    """Return model parameters as independent NumPy copies."""
    return {
        key: value.detach().cpu().numpy().copy()
        for key, value in model.state_dict().items()
    }


def set_parameters(model, parameters):
    """Load NumPy model parameters into a PyTorch model."""
    state = model.state_dict()

    for key in state:
        state[key] = torch.as_tensor(
            parameters[key],
            dtype=state[key].dtype,
        )

    model.load_state_dict(state)


def fedavg(client_parameters, client_sizes):
    """Aggregate client models using sample-count-weighted FedAvg."""
    if not client_parameters:
        raise ValueError("No client parameters were provided.")

    if len(client_parameters) != len(client_sizes):
        raise ValueError(
            "The number of client models and client sizes must match."
        )

    total = sum(client_sizes)

    if total <= 0:
        raise ValueError("Total client sample count must be positive.")

    averaged = {}

    for key in client_parameters[0]:
        averaged[key] = sum(
            client_parameters[i][key] * client_sizes[i]
            for i in range(len(client_parameters))
        ) / total

    return averaged


# ============================================================
# EVALUATION AND PRIVACY ACCOUNTING
# ============================================================

def evaluate(model, X, y):
    """Evaluate a model using accuracy and macro-F1."""
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


def calculate_epsilon(accountant_state, delta):
    """Calculate epsilon from a client's saved RDP accountant state."""
    accountant = RDPAccountant()
    accountant.load_state_dict(accountant_state)
    return accountant.get_epsilon(delta)


def save_accountant_states(accountant_states):
    """Save each client's accountant history for later auditing."""
    for client_id, state in accountant_states.items():
        history = state["history"]

        np.save(
            RESULTS_DIR / f"client_{client_id}_accountant_history.npy",
            np.array(history, dtype=object),
            allow_pickle=True,
        )


# ============================================================
# MAIN FEDERATED TRAINING LOOP
# ============================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    # --------------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------------

    validation_path = DATA_DIR / "validation.csv"

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {validation_path}"
        )

    validation_df = pd.read_csv(validation_path)

    if "activity" not in validation_df.columns:
        raise ValueError(
            "Validation dataset must contain an 'activity' column."
        )

    X_val = torch.tensor(
        validation_df.drop(columns=["activity"]).to_numpy(
            dtype="float32"
        ),
        dtype=torch.float32,
    ).to(DEVICE)

    y_val = (
        validation_df["activity"].to_numpy(dtype="int64") - 1
    )

    if X_val.shape[1] != 561:
        raise ValueError(
            f"Expected 561 features, found {X_val.shape[1]}."
        )

    if len(X_val) != len(y_val) or len(y_val) == 0:
        raise ValueError("Validation dataset is empty or inconsistent.")

    # --------------------------------------------------------
    # Initialize global model
    # --------------------------------------------------------

    global_model = WearableMLP(
        input_dim=X_val.shape[1],
        num_classes=6,
    ).to(DEVICE)

    global_parameters = get_parameters(global_model)

    # --------------------------------------------------------
    # Initialize training histories
    # --------------------------------------------------------

    history = []
    privacy_rows = []

    best_macro_f1 = -1.0
    best_round = 0

    # Each client retains its own cumulative accountant history.
    client_accountant_states = {
        client_id: RDPAccountant().state_dict()
        for client_id in range(1, NUM_CLIENTS + 1)
    }

    # Read client dataset sizes.
    client_sizes = {}

    for client_id in range(1, NUM_CLIENTS + 1):
        client_path = DATA_DIR / f"client_{client_id}.csv"

        if not client_path.exists():
            raise FileNotFoundError(
                f"Client dataset not found: {client_path}"
            )

        client_df = pd.read_csv(client_path)

        if "activity" not in client_df.columns:
            raise ValueError(
                f"Client {client_id} dataset has no activity column."
            )

        if client_df.empty:
            raise ValueError(
                f"Client {client_id} dataset is empty."
            )

        client_sizes[client_id] = len(client_df)

    # --------------------------------------------------------
    # Display configuration
    # --------------------------------------------------------

    print("=" * 70)
    print("FEDHEALTH AI")
    print("BALANCED DP-FEDAVG WITHOUT DROPOUT")
    print("=" * 70)

    print(f"Clients: {NUM_CLIENTS}")
    print(f"Rounds: {ROUNDS}")
    print(f"Local epochs per round: {LOCAL_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Noise multiplier: {NOISE_MULTIPLIER}")
    print(f"Maximum gradient norm: {MAX_GRAD_NORM}")
    print(f"Reporting delta reference: {DELTA_TOTAL}")
    print(f"Per-client accountant delta: {DELTA_PER_CLIENT}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Output directory: {RESULTS_DIR}")

    # --------------------------------------------------------
    # Federated rounds
    # --------------------------------------------------------

    for round_number in range(1, ROUNDS + 1):

        print()
        print("=" * 25 + f" ROUND {round_number}/{ROUNDS} " + "=" * 25)

        round_client_parameters = []
        round_client_sizes = []

        # ----------------------------------------------------
        # Train clients sequentially
        # ----------------------------------------------------

        for client_id in range(1, NUM_CLIENTS + 1):

            print(
                f"\n--- Round {round_number}: "
                f"Client {client_id}/{NUM_CLIENTS} ---"
            )

            client_model = WearableMLP(
                input_dim=X_val.shape[1],
                num_classes=6,
            ).to(DEVICE)

            # Every client starts from the current global model.
            set_parameters(client_model, global_parameters)

            (
                trained_model,
                _local_epsilon,
                updated_accountant_state,
            ) = train_private_local_model(
                client_id=client_id,
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                noise_multiplier=NOISE_MULTIPLIER,
                max_grad_norm=MAX_GRAD_NORM,
                delta=DELTA_PER_CLIENT,
                model=client_model,
                accountant_state=client_accountant_states[client_id],
            )

            # Preserve this client's cumulative accountant state.
            client_accountant_states[client_id] = (
                updated_accountant_state
            )

            round_client_parameters.append(
                get_parameters(trained_model)
            )

            round_client_sizes.append(
                client_sizes[client_id]
            )

            # Release references to local objects before the next client.
            del trained_model
            del client_model

        # ----------------------------------------------------
        # Aggregate client models
        # ----------------------------------------------------

        global_parameters = fedavg(
            round_client_parameters,
            round_client_sizes,
        )

        set_parameters(global_model, global_parameters)

        # ----------------------------------------------------
        # Validate global model
        # ----------------------------------------------------

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
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": BATCH_SIZE,
        })

        # ----------------------------------------------------
        # Per-client cumulative accountant diagnostics
        # ----------------------------------------------------

        cumulative_epsilons = {
            client_id: calculate_epsilon(
                client_accountant_states[client_id],
                DELTA_PER_CLIENT,
            )
            for client_id in range(1, NUM_CLIENTS + 1)
        }

        max_client_epsilon = max(cumulative_epsilons.values())

        privacy_rows.append({
            "round": round_number,
            "noise_multiplier": NOISE_MULTIPLIER,
            "delta_per_client": DELTA_PER_CLIENT,
            "client_1_epsilon": cumulative_epsilons[1],
            "client_2_epsilon": cumulative_epsilons[2],
            "client_3_epsilon": cumulative_epsilons[3],
            "client_4_epsilon": cumulative_epsilons[4],
            "client_5_epsilon": cumulative_epsilons[5],
            "max_client_cumulative_epsilon": max_client_epsilon,
            "note": (
                "Per-client cumulative RDP diagnostics. "
                "Not a verified end-to-end federated privacy guarantee."
            ),
        })

        # ----------------------------------------------------
        # Print round results
        # ----------------------------------------------------

        print("\nROUND RESULTS")
        print("-" * 45)
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation macro-F1: {val_macro_f1:.4f}")

        print(
            f"Maximum client cumulative epsilon "
            f"(delta={DELTA_PER_CLIENT}): "
            f"{max_client_epsilon:.4f}"
        )

        for client_id in range(1, NUM_CLIENTS + 1):
            print(
                f"Client {client_id} cumulative epsilon: "
                f"{cumulative_epsilons[client_id]:.4f}"
            )

        # ----------------------------------------------------
        # Save the best model
        # ----------------------------------------------------

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_round = round_number

            np.savez(
                RESULTS_DIR / "best_parameters.npz",
                **get_parameters(global_model),
            )

            print("New best validation model saved.")

        # ----------------------------------------------------
        # Save current model and experiment histories
        # ----------------------------------------------------

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

        save_accountant_states(client_accountant_states)

        print(f"Round {round_number} results saved.")

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(f"Best round: {best_round}")
    print(f"Best validation macro-F1: {best_macro_f1:.4f}")
    print(f"Results saved to: {RESULTS_DIR}")

    print("\nFinal per-client accountant diagnostics:")

    final_epsilons = {}

    for client_id in range(1, NUM_CLIENTS + 1):
        epsilon = calculate_epsilon(
            client_accountant_states[client_id],
            DELTA_PER_CLIENT,
        )

        final_epsilons[client_id] = epsilon

        print(f"Client {client_id}: epsilon = {epsilon:.4f}")

    print(
        "\nPrivacy interpretation: these are per-client cumulative "
        "RDP accountant estimates under the configured sampling "
        "and noise assumptions. They do not, by themselves, prove "
        "an end-to-end federated privacy guarantee. Document the "
        "adjacency definition, threat model, and composition "
        "assumptions before making formal privacy claims."
    )


if __name__ == "__main__":
    main()

