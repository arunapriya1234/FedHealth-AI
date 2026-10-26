
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant

from models_no_dropout import WearableMLP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIENT_DATA_DIR = PROJECT_ROOT / "experiments" / "balanced_data"


def train_private_local_model(
    client_id=1,
    epochs=2,
    batch_size=32,
    learning_rate=0.001,
    noise_multiplier=0.7,
    max_grad_norm=1.0,
    delta=1e-6,
    model=None,
    accountant_state=None,
):
    """Train a client model using DP-SGD and preserve its RDP history."""

    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if noise_multiplier <= 0 or max_grad_norm <= 0:
        raise ValueError(
            "Noise multiplier and gradient norm must be positive."
        )

    if not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1.")

    data_path = CLIENT_DATA_DIR / f"client_{client_id}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Client dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    if df.empty or "activity" not in df.columns:
        raise ValueError(f"Invalid client dataset: {data_path}")

    X = torch.tensor(
        df.drop(columns=["activity"]).to_numpy(dtype="float32"),
        dtype=torch.float32,
    )

    y = torch.tensor(
        df["activity"].to_numpy(dtype="int64") - 1,
        dtype=torch.long,
    )

    if not torch.isfinite(X).all():
        raise ValueError(f"Non-finite features found in {data_path}")

    if y.min().item() < 0 or y.max().item() > 5:
        raise ValueError("Activity labels must map to values 0 through 5.")

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=False,
    )

    if model is None:
        model = WearableMLP(
            input_dim=X.shape[1],
            num_classes=6,
        )

    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    privacy_engine = PrivacyEngine(accountant="rdp")

    if accountant_state is not None:
        accountant = RDPAccountant()
        accountant.load_state_dict(accountant_state)
        privacy_engine.accountant = accountant

    private_model = None

    try:
        private_model, optimizer, private_loader = (
            privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                poisson_sampling=True,
            )
        )

        print(f"\nDP-FedAvg | Client {client_id}")
        print(f"Samples: {len(dataset)}")
        print(f"Noise multiplier: {noise_multiplier}")
        print(f"Maximum gradient norm: {max_grad_norm}")

        for epoch in range(epochs):
            total_loss = 0.0
            total_seen = 0

            for features, labels in private_loader:
                optimizer.zero_grad()

                outputs = private_model(features)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                count = labels.size(0)
                total_loss += loss.item() * count
                total_seen += count

            epsilon = privacy_engine.get_epsilon(delta)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {total_loss / max(total_seen, 1):.4f} | "
                f"Cumulative accountant epsilon: {epsilon:.4f}"
            )

        privacy_epsilon = privacy_engine.get_epsilon(delta)
        updated_accountant_state = (
            privacy_engine.accountant.state_dict()
        )

        # Return the underlying model for aggregation by the server.
        trained_model = private_model._module

    finally:
        # Remove Opacus gradient-sampling hooks before returning.
        if private_model is not None:
            private_model.remove_hooks()

    print(
        f"Client {client_id} epsilon from its preserved accountant: "
        f"{privacy_epsilon:.4f}"
    )

    return (
        trained_model,
        privacy_epsilon,
        updated_accountant_state,
    )


if __name__ == "__main__":
    trained_model, epsilon, accountant_state = (
        train_private_local_model(client_id=1)
    )

    print("\nDP client smoke test completed.")
    print(f"Epsilon for this training run: {epsilon:.4f}")
    print(
        "Accountant history entries:",
        len(accountant_state["history"]),
    )
    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in trained_model.parameters()
            if p.requires_grad
        ),
    )
