
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_processing import load_dataset


def create_client_partitions(
    num_clients=5,
    alpha=0.5,
    random_seed=42,
    validation_size=0.20,
):
    """Create non-IID client partitions and a separate validation set."""

    X_train, y_train, _, _, _, activities = load_dataset()

    X = X_train.to_numpy(dtype=np.float32)
    y = y_train["activity"].to_numpy(dtype=np.int64)

    # Reserve part of the official training set for validation.
    all_indices = np.arange(len(y))
    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=validation_size,
        random_state=random_seed,
        stratify=y,
    )

    X_federated = X[train_indices]
    y_federated = y[train_indices]

    rng = np.random.default_rng(random_seed)
    client_indices = [[] for _ in range(num_clients)]

    # Distribute each activity unevenly across clients.
    for label in np.unique(y_federated):
        label_indices = np.where(y_federated == label)[0]
        rng.shuffle(label_indices)

        proportions = rng.dirichlet(
            np.full(num_clients, alpha)
        )

        split_points = (
            np.cumsum(proportions)[:-1] * len(label_indices)
        ).astype(int)

        splits = np.split(label_indices, split_points)

        for client_id, indices in enumerate(splits):
            client_indices[client_id].extend(indices.tolist())

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    distribution_rows = []

    for client_id, indices in enumerate(client_indices):
        indices = np.asarray(indices, dtype=int)
        rng.shuffle(indices)

        client_df = pd.DataFrame(X_federated[indices])
        client_df["activity"] = y_federated[indices]

        client_path = output_dir / f"client_{client_id + 1}.csv"
        client_df.to_csv(client_path, index=False)

        counts = pd.Series(y_federated[indices]).value_counts()

        for _, activity_row in activities.iterrows():
            activity_id = int(activity_row["activity"])

            distribution_rows.append({
                "client": f"client_{client_id + 1}",
                "activity": activity_id,
                "activity_name": activity_row["activity_name"],
                "samples": int(counts.get(activity_id, 0)),
            })

        print(
            f"Client {client_id + 1}: "
            f"{len(indices)} training samples"
        )

    # Save the held-out validation set separately.
    validation_df = pd.DataFrame(X[validation_indices])
    validation_df["activity"] = y[validation_indices]
    validation_df.to_csv(
        output_dir / "validation.csv",
        index=False,
    )

    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(
        output_dir / "client_activity_distribution.csv",
        index=False,
    )

    print("\nClient activity distribution:")
    print(
        distribution.pivot(
            index="client",
            columns="activity_name",
            values="samples",
        ).fillna(0).astype(int)
    )

    print("\nPartition summary:")
    print(f"Total original training samples: {len(y)}")
    print(f"Federated training samples: {len(train_indices)}")
    print(f"Validation samples: {len(validation_indices)}")
    print("Official test samples: 2947 (unchanged)")
    print(f"\nFiles saved to: {output_dir}")


if __name__ == "__main__":
    create_client_partitions()
