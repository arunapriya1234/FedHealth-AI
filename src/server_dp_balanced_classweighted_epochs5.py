
from pathlib import Path
import importlib.util
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(
        module_name, file_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load the existing server configuration and FedAvg implementation.
server = load_module(
    "classweighted_base_server",
    SRC_DIR / "server_dp_balanced_clip2.py",
)

# Load the new class-weighted private client implementation.
client = load_module(
    "classweighted_dp_client",
    SRC_DIR / "client_dp_balanced_classweighted.py",
)

# Replace only the training function used by this server instance.
server.train_private_local_model = client.train_private_local_model

# Experiment settings.
server.NUM_CLIENTS = 5
server.ROUNDS = 10
server.LOCAL_EPOCHS = 5
server.BATCH_SIZE = 32
server.LEARNING_RATE = 0.001
server.NOISE_MULTIPLIER = 0.7
server.MAX_GRAD_NORM = 2.0

# Save to a NEW folder; preserve all previous experiment results.
server.RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "balanced_dp_classweighted_epochs5_run"
)

server.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("FedHealth AI - Class-Weighted DP-FedAvg")
print("=" * 65)
print("Clients:", server.NUM_CLIENTS)
print("Federated rounds:", server.ROUNDS)
print("Local epochs:", server.LOCAL_EPOCHS)
print("Batch size:", server.BATCH_SIZE)
print("Noise multiplier:", server.NOISE_MULTIPLIER)
print("Maximum gradient norm:", server.MAX_GRAD_NORM)
print("Results directory:", server.RESULTS_DIR)
print("=" * 65)

server.main()
