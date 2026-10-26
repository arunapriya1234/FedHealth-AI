
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import WearableMLP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIENT_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def train_local_model(
    client_id=1,
    epochs=2,
    batch_size=32,
    learning_rate=0.001,
    model=None,
):
    """Train a client's local model starting from global parameters."""

    data_path = CLIENT_DATA_DIR / f"client_{client_id}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Client dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    X = torch.tensor(
        df.drop(columns=["activity"]).to_numpy(dtype="float32")
    )

    # Convert activity labels 1-6 into class indices 0-5.
    y = torch.tensor(
        df["activity"].to_numpy(dtype="int64") - 1
    )

    dataset = TensorDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Use the global model provided by the server.
    # Initialize a new model only for standalone testing.
    if model is None:
        model = WearableMLP()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in loader:
            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        print(
            f"Client {client_id} | "
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {total_loss / total:.4f} | "
            f"Training accuracy: {correct / total:.4f}"
        )

    print(f"\nClient {client_id} local training completed!")

    return model


if __name__ == "__main__":
    trained_model = train_local_model(client_id=1)

    print(
        "Trainable parameters:",
        sum(
            parameter.numel()
            for parameter in trained_model.parameters()
            if parameter.requires_grad
        ),
    )
