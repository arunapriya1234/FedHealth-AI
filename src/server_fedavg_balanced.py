from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from models import WearableMLP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIENT_DATA_DIR = PROJECT_ROOT / "experiments" / "balanced_data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "balanced_fedavg"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_CLIENTS = 5
ROUNDS = 10
LOCAL_EPOCHS = 2
SEED = 42


def get_parameters(model):
    """Return independent copies of model parameters."""
    return [
        tensor.detach().cpu().numpy().copy()
        for tensor in model.state_dict().values()
    ]


def set_parameters(model, parameters):
    """Load NumPy parameters into a PyTorch model."""
    state = model.state_dict()

    if len(parameters) != len(state):
        raise ValueError("Parameter count does not match model state.")

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
    if not client_parameters or len(client_parameters) != len(client_sizes):
        raise ValueError("Client parameters and sizes do not match.")

    total_samples = sum(client_sizes)

    if total_samples <= 0 or any(size <= 0 for size in client_sizes):
        raise ValueError("Client datasets must contain samples.")

    averaged_parameters = []

    for parameter_group in zip(*client_parameters):
        weighted_average = sum(
            parameter * (size / total_samples)
            for parameter, size in zip(parameter_group, client_sizes)
        )
        averaged_parameters.append(weighted_average)

    return averaged_parameters


def load_client_data(client_id):
    """Load one balanced client's local training dataset."""
    data_path = CLIENT_DATA_DIR / f"client_{client_id}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Client dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    if df.empty or "activity" not in df.columns:
        raise ValueError(f"Invalid client dataset: {data_path}")

    if len(df.columns) != 562:
        raise ValueError(
            f"Expected 561 features plus activity in {data_path}; "
            f"found {len(df.columns)} columns."
        )

    X = df.drop(columns=["activity"]).to_numpy(dtype=np.float32)
    y = df["activity"].to_numpy(dtype=np.int64) - 1

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite features found in {data_path}")

    if not np.isin(y, np.arange(6)).all():
        raise ValueError(f"Unexpected activity labels in {data_path}")

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )

    return dataset


def train_local_model(client_id, global_parameters, dataset):
    """Train a client model without differential privacy."""
    model = WearableMLP()
    set_parameters(model, global_parameters)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    for epoch in range(LOCAL_EPOCHS):
        total_loss = 0.0
        total_samples = 0

        for features, labels in loader:
            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        print(
            f"Client {client_id} | "
            f"Epoch {epoch + 1}/{LOCAL_EPOCHS} | "
            f"Loss: {total_loss / total_samples:.4f}"
        )

    return get_parameters(model)


def load_validation_data():
    """Load validation data without using the official test set."""
    validation_path = CLIENT_DATA_DIR / "validation.csv"

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation file not found: {validation_path}"
        )

    df = pd.read_csv(validation_path)

    if df.empty or "activity" not in df.columns or len(df.columns) != 562:
        raise ValueError("Invalid balanced validation dataset.")

    X = df.drop(columns=["activity"]).to_numpy(dtype=np.float32)
    y = df["activity"].to_numpy(dtype=np.int64) - 1

    if not np.isfinite(X).all():
        raise ValueError("Validation features contain non-finite values.")

    if not np.isin(y, np.arange(6)).all():
        raise ValueError("Unexpected validation activity labels.")

    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


def evaluate_model(parameters, dataset):
    """Evaluate a global model on validation data."""
    model = WearableMLP()
    set_parameters(model, parameters)
    model.eval()

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    predictions = []
    actual_labels = []

    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(actual_labels, predictions)
    macro_f1 = f1_score(
        actual_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return accuracy, macro_f1


def run_federated_training():
    """Run non-private FedAvg with balanced client data."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not CLIENT_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Balanced data directory not found: {CLIENT_DATA_DIR}"
        )

    client_datasets = {}
    client_sizes = []

    for client_id in range(1, NUM_CLIENTS + 1):
        dataset = load_client_data(client_id)
        client_datasets[client_id] = dataset
        client_sizes.append(len(dataset))

        print(f"Client {client_id}: {len(dataset)} training samples")

    validation_dataset = load_validation_data()

    global_model = WearableMLP()
    global_parameters = get_parameters(global_model)

    best_validation_f1 = -1.0
    best_round = 0
    best_parameters = None
    history = []

    print("\nFEDHEALTH AI - BALANCED NON-PRIVATE FEDAVG")
    print("-" * 48)
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Training samples: {sum(client_sizes)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Rounds: {ROUNDS} | Local epochs: {LOCAL_EPOCHS}")
    print("Differential privacy: Disabled")
    print("Official test set: Reserved for final evaluation")

    for round_number in range(1, ROUNDS + 1):
        print(f"\n{'=' * 12} ROUND {round_number} {'=' * 12}")

        client_parameters = []

        for client_id in range(1, NUM_CLIENTS + 1):
            parameters = train_local_model(
                client_id,
                global_parameters,
                client_datasets[client_id],
            )
            client_parameters.append(parameters)

        global_parameters = fedavg(client_parameters, client_sizes)

        val_accuracy, val_f1 = evaluate_model(
            global_parameters,
            validation_dataset,
        )

        print(f"Validation accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")
        print(f"Validation macro-F1: {val_f1:.4f}")

        history.append({
            "round": round_number,
            "clients": NUM_CLIENTS,
            "training_samples": sum(client_sizes),
            "validation_samples": len(validation_dataset),
            "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_f1,
        })

        if val_f1 > best_validation_f1:
            best_validation_f1 = val_f1
            best_round = round_number
            best_parameters = [
                parameter.copy() for parameter in global_parameters
            ]
            print("New best model selected.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        RESULTS_DIR / "fedavg_balanced_best_parameters.npz",
        *best_parameters,
    )

    np.savez(
        RESULTS_DIR / "fedavg_balanced_global_parameters.npz",
        *global_parameters,
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        RESULTS_DIR / "fedavg_balanced_history.csv",
        index=False,
    )

    print("\nTRAINING COMPLETED")
    print(f"Best validation round: {best_round}")
    print(f"Best validation accuracy: "
          f"{history_df.loc[best_round - 1, 'validation_accuracy']:.4f}")
    print(f"Best validation macro-F1: {best_validation_f1:.4f}")
    print(f"Results saved in: {RESULTS_DIR}")
    print("\nRound-by-round results:")
    print(history_df.to_string(index=False))


if __name__ == "__main__":
    run_federated_training()