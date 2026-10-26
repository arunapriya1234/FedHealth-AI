from pathlib import Path
import importlib.util

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_SERVER = PROJECT_ROOT / "src" / "server_dp_balanced_clip2.py"

# Load the existing DP server without changing the original file
spec = importlib.util.spec_from_file_location(
    "dp_server_epochs5",
    ORIGINAL_SERVER,
)

if spec is None or spec.loader is None:
    raise ImportError(f"Could not load the original server: {ORIGINAL_SERVER}")

dp_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp_server)

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# Change only the number of local training epochs
dp_server.LOCAL_EPOCHS = 5

# Save results separately to preserve previous experiments
dp_server.RESULTS_DIR = (
    PROJECT_ROOT / "experiments" / "balanced_dp_epochs5_run"
)

# Keep the same clipping norm and noise multiplier
dp_server.NOISE_MULTIPLIER = 0.7
dp_server.MAX_GRAD_NORM = 2.0

print("=" * 65)
print("FEDHEALTH AI - DP-FEDAVG WITH 5 LOCAL EPOCHS")
print("=" * 65)
print(f"Clients: {dp_server.NUM_CLIENTS}")
print(f"Communication rounds: {dp_server.ROUNDS}")
print(f"Local epochs: {dp_server.LOCAL_EPOCHS}")
print(f"Batch size: {dp_server.BATCH_SIZE}")
print(f"Learning rate: {dp_server.LEARNING_RATE}")
print(f"Noise multiplier: {dp_server.NOISE_MULTIPLIER}")
print(f"Maximum gradient norm: {dp_server.MAX_GRAD_NORM}")
print(f"Output folder: {dp_server.RESULTS_DIR}")
print("=" * 65)

# Run the existing server with the updated settings
dp_server.main()