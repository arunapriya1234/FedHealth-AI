import sys

sys.path.insert(0, "src")

from opacus.accountants import RDPAccountant
from client_dp_balanced_no_dropout import train_private_local_model


def count_steps(state):
    accountant = RDPAccountant()
    accountant.load_state_dict(state)
    return sum(entry[2] for entry in accountant.history)


print("\n========== FIRST TRAINING CALL ==========")

model, epsilon1, state1 = train_private_local_model(
    client_id=1,
    epochs=1,
    batch_size=32,
    noise_multiplier=0.7,
    max_grad_norm=1.0,
    delta=1e-6,
)

steps1 = count_steps(state1)

print("Steps after first call:", steps1)
print("Epsilon after first call:", round(epsilon1, 4))


print("\n========== SECOND TRAINING CALL ==========")

model2, epsilon2, state2 = train_private_local_model(
    client_id=1,
    epochs=1,
    batch_size=32,
    noise_multiplier=0.7,
    max_grad_norm=1.0,
    delta=1e-6,
    model=model,
    accountant_state=state1,
)

steps2 = count_steps(state2)

print("Steps after second call:", steps2)
print("Epsilon after second call:", round(epsilon2, 4))


if steps2 > steps1 and epsilon2 >= epsilon1:
    print("\nPASS: Cumulative privacy accounting is working.")
else:
    print("\nCHECK REQUIRED: Cumulative accounting needs further investigation.")