from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant

from models_no_dropout import WearableMLP


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIENT_DATA_DIR = PROJECT_ROOT / "experiments" / "balanced_data"


# ============================================================
# PRIVATE LOCAL TRAINING
# ============================================================

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
    """
    Train a client model using DP-SGD.

    Parameters
    ----------
    client_id : int
        Client dataset identifier.

    epochs : int
        Number of local training epochs.

    batch_size : int
        Requested batch size.

    learning_rate : float
        Optimizer learning rate.

    noise_multiplier : float
        Gaussian noise multiplier used by DP-SGD.

    max_grad_norm : float
        Maximum per-sample gradient norm.

    delta : float
        Delta used to calculate the reported epsilon.

    model : torch.nn.Module or None
        Optional model containing the global model's weights.

    accountant_state : dict or None
        Previously saved RDP accountant state for this client.

    Returns
    -------
    trained_model, privacy_epsilon, updated_accountant_state
    """

    # --------------------------------------------------------
    # 1. Validate parameters
    # --------------------------------------------------------

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive.")

    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")

    if not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1.")

    # --------------------------------------------------------
    # 2. Load client dataset
    # --------------------------------------------------------

    data_path = CLIENT_DATA_DIR / f"client_{client_id}.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Client dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError(f"Client dataset is empty: {data_path}")

    if "activity" not in df.columns:
        raise ValueError(
            f"'activity' column is missing from {data_path}"
        )

    # Features
    X = torch.tensor(
        df.drop(columns=["activity"]).to_numpy(dtype="float32"),
        dtype=torch.float32,
    )

    # Activity labels are assumed to be 1 through 6.
    # Convert them to class indices 0 through 5.
    y = torch.tensor(
        df["activity"].to_numpy(dtype="int64") - 1,
        dtype=torch.long,
    )

    if X.ndim != 2 or X.shape[1] != 561:
        raise ValueError(
            f"Expected 561 input features, got {X.shape}"
        )

    if not torch.isfinite(X).all():
        raise ValueError(
            f"Non-finite feature values found in {data_path}"
        )

    if y.min().item() < 0 or y.max().item() > 5:
        raise ValueError(
            "Activity labels must map to values 0 through 5."
        )

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=False,
    )

    print(f"\n{'=' * 55}")
    print(f"DP-FedAvg | Client {client_id}")
    print(f"{'=' * 55}")
    print(f"Dataset samples: {len(dataset)}")
    print(f"Input features: {X.shape[1]}")
    print(f"Local epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Noise multiplier: {noise_multiplier}")
    print(f"Maximum gradient norm: {max_grad_norm}")
    print(f"Delta: {delta}")

    # --------------------------------------------------------
    # 3. Initialize or reuse the model
    # --------------------------------------------------------

    if model is None:
        model = WearableMLP(
            input_dim=X.shape[1],
            num_classes=6,
        )

    model.train()

    # --------------------------------------------------------
    # 4. Create optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # --------------------------------------------------------
    # 5. Initialize the privacy engine
    # --------------------------------------------------------

    privacy_engine = PrivacyEngine(accountant="rdp")

    # Restore this client's cumulative accounting history.
    if accountant_state is not None:
        accountant = RDPAccountant()
        accountant.load_state_dict(accountant_state)

        privacy_engine.accountant = accountant

        print(
            "Restored accountant history entries:",
            len(privacy_engine.accountant.history),
        )
    else:
        print("Starting a new client accountant history.")

    # --------------------------------------------------------
    # 6. Make training private
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 7. Train the model
    # --------------------------------------------------------

    try:
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

            average_loss = total_loss / max(total_seen, 1)

            cumulative_epsilon = privacy_engine.get_epsilon(delta)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {average_loss:.4f} | "
                f"Cumulative epsilon: {cumulative_epsilon:.4f}"
            )

        # ----------------------------------------------------
        # 8. Capture accountant state before cleanup
        # ----------------------------------------------------

        privacy_epsilon = privacy_engine.get_epsilon(delta)

        updated_accountant_state = (
            privacy_engine.accountant.state_dict()
        )

        print("\nClient training completed.")
        print(f"Cumulative epsilon: {privacy_epsilon:.4f}")
        print(
            "Accountant history entries:",
            len(updated_accountant_state["history"]),
        )

        # Save the underlying model, not the Opacus wrapper.
        trained_model = private_model._module

    finally:
        # ----------------------------------------------------
        # 9. Remove Opacus gradient hooks
        # ----------------------------------------------------

        # This allows the returned model to be wrapped again
        # in a later local-training call.
        private_model.remove_hooks()

    return (
        trained_model,
        privacy_epsilon,
        updated_accountant_state,
    )


# ============================================================
# STANDALONE SMOKE TEST
# ============================================================

if __name__ == "__main__":

    trained_model, epsilon, accountant_state = (
        train_private_local_model(
            client_id=1,
            epochs=1,
            batch_size=32,
            noise_multiplier=0.7,
            max_grad_norm=1.0,
            delta=1e-6,
        )
    )

    print("\n" + "=" * 55)
    print("DP CLIENT SMOKE TEST COMPLETED")
    print("=" * 55)

    print(f"Epsilon: {epsilon:.4f}")

    print(
        "Accountant history entries:",
        len(accountant_state["history"]),
    )

    print(
        "Trainable parameters:",
        sum(
            parameter.numel()
            for parameter in trained_model.parameters()
            if parameter.requires_grad
        ),
    )