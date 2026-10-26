
import torch
import torch.nn as nn


class WearableMLP(nn.Module):
    """Wearable activity classifier without dropout."""

    def __init__(self, input_dim=561, num_classes=6):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    model = WearableMLP()
    dummy_input = torch.randn(8, 561)
    output = model(dummy_input)

    print("No-dropout model created successfully!")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    assert output.shape == (8, 6)
    print("Model validation passed!")
