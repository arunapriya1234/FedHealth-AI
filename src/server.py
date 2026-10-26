
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from models import WearableMLP
from client import train_local_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIENT_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

BATCH_SIZE = 64
RANDOM_SEED = 42


def get_parameters(model):
    """Return independent NumPy copies of model parameters."""
    return [
        tensor.detach().cpu().numpy().copy()
        for tensor in model.state_dict().values()
    ]


def set_parameters(model, parameters):
    """Load NumPy parameters into a PyTorch model."""
    state = model.state_dict()

    if len(parameters) != len(state):
        raise ValueError("Number of parameter arrays does not match model.")

    updated_state = {
        key: torch.as_tensor(
            parameters[index],
            dtype=value.dtype,
            device=value.device,
        )
        for index, (key, value) in enumerate(state.items())
    }

    model.load_state_dict(updated_state)


def fedavg(client_parameters, client_sizes):
    """Perform sample-weighted Federated Averaging."""
    if not client_parameters:
        raise ValueError("No client parameters were supplied.")

    if len(client_parameters) != len(client_sizes):
        raise ValueError("Client parameter count and client size differ.")

    if any(size <= 0 for size in client_sizes):
        raise ValueError("Every client must have positive sample size.")

    if any(
        len(params) != len(client_parameters[0])
        for params in client_parameters
    ):
        raise ValueError("Client model parameter counts differ.")

    total_samples = sum(client_sizes)
    averaged_parameters = []

    for parameter_group in zip(*client_parameters):
        weighted_average = sum(
            parameter * (size / total_samples)
            for parameter, size in zip(parameter_group, client_sizes)
        )
        averaged_parameters.append(weighted_average)

    return averaged_parameters


def get_client_files():
    """Return numbered client CSV files in numeric order."""
    files = []

    for path in CLIENT_DIR.glob("client_*.csv"):
        client_id = path.stem.removeprefix("client_")
        if client_id.isdigit():
            files.append(path)

    return sorted(
        files,
        key=lambda path: int(path.stem.removeprefix("client_")),
    )


def load_official_test_data():
    """Load the official test set for final evaluation only."""
    X_path = DATA_DIR / "test" / "X_test.txt"
    y_path = DATA_DIR / "test" / "y_test.txt"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError("Official UCI HAR test files were not found.")

    X = np.loadtxt(X_path, dtype=np.float32)
    y = np.loadtxt(y_path, dtype=np.int64) - 1

    if X.shape != (2947, 561) or len(y) != len(X):
        raise ValueError(f"Unexpected test-data dimensions: {X.shape}, {y.shape}")

    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


def load_validation_data():
    """Load validation data created from the original training set."""
    validation_path = CLIENT_DIR / "validation.csv"

    if not validation_path.exists():
        raise FileNotFoundError(
            "validation.csv was not found. Run "
            "'python src\\create_clients.py' first."
        )

    df = pd.read_csv(validation_path)

    if df.shape[1] != 562:
        raise ValueError(
            f"Expected 561 features plus activity; found {df.shape[1]} columns."
        )

    X = df.drop(columns=["activity"]).to_numpy(dtype=np.float32)
    y = df["activity"].to_numpy(dtype=np.int64) - 1

    if not np.isfinite(X).all():
        raise ValueError("Validation features contain non-finite values.")

    if not np.isin(y, np.arange(6)).all():
        raise ValueError("Unexpected labels in validation data.")

    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


def evaluate_model(parameters, dataset):
    """Evaluate parameters on a supplied dataset without training."""
    model = WearableMLP()
    set_parameters(model, parameters)
    model.eval()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    predictions = []
    labels_all = []

    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    accuracy = accuracy_score(labels_all, predictions)
    macro_f1 = f1_score(
        labels_all,
        predictions,
        average="macro",
        zero_division=0,
    )

    return accuracy, macro_f1, len(labels_all)


def run_federated_training(rounds=10, local_epochs=2):
    """Train simulated FedAvg and select a model by validation macro-F1."""
    if rounds < 1 or local_epochs < 1:
        raise ValueError("Rounds and local epochs must be positive.")

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    client_files = get_client_files()
    if not client_files:
        raise FileNotFoundError(f"No client CSV files found in {CLIENT_DIR}")

    client_ids = [
        int(path.stem.removeprefix("client_"))
        for path in client_files
    ]

    client_sizes = []
    for path in client_files:
        df = pd.read_csv(path)

        if df.empty or "activity" not in df.columns or len(df.columns) != 562:
            raise ValueError(f"Invalid client dataset: {path}")

        client_sizes.append(len(df))

    validation_dataset = load_validation_data()
    total_samples = sum(client_sizes)

    global_model = WearableMLP()
    global_parameters = get_parameters(global_model)

    best_validation_f1 = -1.0
    best_round = 0
    best_parameters = None
    history = []

    print("FEDHEALTH AI - VALIDATION-BASED FEDERATED LEARNING")
    print("-" * 54)
    print(f"Clients: {len(client_ids)}")
    print(f"Federated training samples: {total_samples}")
    print(f"Validation samples: {len(validation_dataset)}")
    print("Official test set: reserved for final evaluation")
    print(f"Rounds: {rounds} | Local epochs: {local_epochs}")
    print("Aggregation: sample-weighted FedAvg")

    for round_number in range(1, rounds + 1):
        print(f"\n{'=' * 12} ROUND {round_number} {'=' * 12}")

        client_parameters = []

        for client_id, client_size in zip(client_ids, client_sizes):
            local_model = WearableMLP()
            set_parameters(local_model, global_parameters)

            trained_model = train_local_model(
                client_id=client_id,
                epochs=local_epochs,
                model=local_model,
            )
            client_parameters.append(get_parameters(trained_model))

            print(f"Client {client_id}: trained on {client_size} samples")

        global_parameters = fedavg(client_parameters, client_sizes)
        set_parameters(global_model, global_parameters)

        val_accuracy, val_f1, val_count = evaluate_model(
            global_parameters,
            validation_dataset,
        )

        print(f"Validation accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")
        print(f"Validation macro-F1: {val_f1:.4f}")
        print(f"Validation samples: {val_count}")

        history.append({
            "round": round_number,
            "clients": len(client_ids),
            "training_samples": total_samples,
            "validation_samples": val_count,
            "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_f1,
        })

        # Select the best global model using validation macro-F1.
        if val_f1 > best_validation_f1:
            best_validation_f1 = val_f1
            best_round = round_number
            best_parameters = [
                parameter.copy() for parameter in global_parameters
            ]
            print("New best model selected using validation macro-F1.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if best_parameters is None:
        raise RuntimeError("No best model was selected.")

    best_model_path = RESULTS_DIR / "fedavg_best_validation_parameters.npz"
    np.savez(best_model_path, *best_parameters)

    history_path = RESULTS_DIR / "federated_training_history_detailed.csv"
    summary_path = RESULTS_DIR / "fedavg_round_tracking_summary.csv"

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)
    history_df.loc[history_df["round"] == best_round].to_csv(
        summary_path, index=False
    )

    # Keep the last-round parameters separately for reproducibility.
    last_model_path = RESULTS_DIR / "fedavg_global_parameters.npz"
    np.savez(last_model_path, *global_parameters)

    print("\nFEDERATED TRAINING COMPLETED")
    print("-" * 40)
    print(f"Best validation round: {best_round}")
    print(f"Best validation macro-F1: {best_validation_f1:.4f}")
    print(f"Best validation model: {best_model_path}")
    print(f"Last-round model: {last_model_path}")
    print(f"Detailed history: {history_path}")
    print(f"Round summary: {summary_path}")
    print("\nROUND-BY-ROUND VALIDATION RESULTS")
    print(history_df.to_string(index=False))


if __name__ == "__main__":
    run_federated_training(rounds=10, local_epochs=2)
