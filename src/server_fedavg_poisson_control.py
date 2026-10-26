
import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from models_no_dropout import WearableMLP


# -------------------- CONFIGURATION --------------------

SEED = 42
NUM_CLIENTS = 5
ROUNDS = 10
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001

DATA_DIR = "experiments/balanced_data"
OUTPUT_DIR = "experiments/balanced_fedavg_no_dropout"

DEVICE = torch.device("cpu")


# -------------------- REPRODUCIBILITY --------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------- LOAD CLIENT DATA --------------------

def load_client_data(client_id):
    path = os.path.join(DATA_DIR, f"client_{client_id}.csv")
    df = pd.read_csv(path)

    feature_columns = [col for col in df.columns if col != "activity"]

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["activity"].to_numpy(dtype=np.int64) - 1

    return X, y


def load_validation_data():
    path = os.path.join(DATA_DIR, "validation.csv")
    df = pd.read_csv(path)

    feature_columns = [col for col in df.columns if col != "activity"]

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["activity"].to_numpy(dtype=np.int64) - 1

    return X, y


# -------------------- LOCAL TRAINING --------------------

def train_local_model(global_parameters, client_id):
    X, y = load_client_data(client_id)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    # Reproducible client-specific shuffling
    generator = torch.Generator()
    generator.manual_seed(SEED + client_id)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    model = WearableMLP(input_dim=X.shape[1], num_classes=6).to(DEVICE)
    model.load_state_dict(copy.deepcopy(global_parameters))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    for epoch in range(LOCAL_EPOCHS):
        total_loss = 0.0
        total_samples = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            total_samples += len(batch_y)

    average_loss = total_loss / total_samples

    return model.state_dict(), len(dataset), average_loss


# -------------------- FEDAVG AGGREGATION --------------------

def aggregate_parameters(client_parameters, client_sizes):
    total_samples = sum(client_sizes)
    aggregated = {}

    for key in client_parameters[0]:
        aggregated[key] = sum(
            client_parameters[i][key] * client_sizes[i]
            for i in range(len(client_parameters))
        ) / total_samples

    return aggregated


# -------------------- VALIDATION --------------------

def evaluate_validation(parameters, X_val, y_val):
    model = WearableMLP(
        input_dim=X_val.shape[1],
        num_classes=6
    ).to(DEVICE)

    model.load_state_dict(parameters)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X_val, dtype=torch.float32)
        predictions = model(X_tensor).argmax(dim=1).numpy()

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    return accuracy, macro_f1


# -------------------- MAIN FEDAVG LOOP --------------------

def main():
    print("=" * 60)
    print("BALANCED FEDAVG WITHOUT DROPOUT — NON-PRIVATE")
    print("=" * 60)

    X_val, y_val = load_validation_data()

    print(f"Validation samples: {len(y_val)}")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Rounds: {ROUNDS}")
    print(f"Local epochs: {LOCAL_EPOCHS}")
    print(f"Device: {DEVICE}")

    global_model = WearableMLP(
        input_dim=X_val.shape[1],
        num_classes=6
    ).to(DEVICE)

    global_parameters = global_model.state_dict()

    history = []
    best_macro_f1 = -1.0
    best_round = 0

    for round_number in range(1, ROUNDS + 1):
        print(f"\nRound {round_number}/{ROUNDS}")

        client_parameters = []
        client_sizes = []
        client_losses = []

        for client_id in range(1, NUM_CLIENTS + 1):
            parameters, sample_count, loss = train_local_model(
                global_parameters,
                client_id
            )

            client_parameters.append(parameters)
            client_sizes.append(sample_count)
            client_losses.append(loss)

            print(
                f"Client {client_id}: "
                f"samples={sample_count}, loss={loss:.4f}"
            )

        global_parameters = aggregate_parameters(
            client_parameters,
            client_sizes
        )

        val_accuracy, val_macro_f1 = evaluate_validation(
            global_parameters,
            X_val,
            y_val
        )

        average_loss = float(np.average(
            client_losses,
            weights=client_sizes
        ))

        history.append({
            "round": round_number,
            "average_training_loss": average_loss,
            "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_macro_f1
        })

        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation macro-F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_round = round_number

            np.savez(
                os.path.join(OUTPUT_DIR, "best_parameters.npz"),
                **{
                    key: value.detach().cpu().numpy()
                    for key, value in global_parameters.items()
                }
            )

            print("New best model saved.")

    np.savez(
        os.path.join(OUTPUT_DIR, "global_parameters.npz"),
        **{
            key: value.detach().cpu().numpy()
            for key, value in global_parameters.items()
        }
    )

    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, "history.csv"),
        index=False
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print(f"Best validation macro-F1: {best_macro_f1:.4f}")
    print(f"Best round: {best_round}")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
