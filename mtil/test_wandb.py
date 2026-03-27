"""
Smoke test for wandb integration in trainer_phase3.
Run from mtil/ with: python test_wandb.py
No GPU required — just verifies wandb init/log/finish works offline.
"""
import os
import csv
import tempfile
import wandb

save_dir = tempfile.mkdtemp(prefix="zscl_wandb_smoke_")
print(f"Smoke test dir: {save_dir}")

# Simulate what trainer_phase3 does
wandb.init(
    project="zscl-mtil",
    name="smoke_test_Aircraft_lam0.1_r8",
    config={
        "task": "Aircraft",
        "lambda_rtd": 0.1,
        "lora_r": 8,
        "use_lora": True,
        "replay_budget": 5000,
        "replay_loss_weight": 0.75,
        "lr": 1e-5,
        "iterations": 2000,
    },
    dir=save_dir,
    mode=os.getenv("WANDB_MODE", "offline"),
)

# Simulate a few loss log steps
for iteration in [50, 100, 150]:
    wandb.log({
        "loss/total": 2.5 - iteration * 0.005,
        "loss/ce": 1.2,
        "loss/l2": 0.3,
        "loss/zscl": 0.5,
        "loss/replay_sup": 0.4,
        "loss/replay_teacher": 0.1,
        "replay_buffer_size": 0,
    }, step=iteration)

# Simulate eval metric logging
fake_csv = os.path.join(save_dir, "metrics_Aircraft.csv")
with open(fake_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "top1", "top5"])
    writer.writerow([500, 42.5, 78.3])

with open(fake_csv, newline="") as f:
    rows = list(csv.DictReader(f))
if rows:
    wandb.log({"eval/Aircraft_top1": float(rows[-1]["top1"])}, step=500)

wandb.finish()

print("\n[PASS] wandb smoke test complete.")
print(f"Offline run saved to: {save_dir}/wandb/")
print("To sync after a real job: wandb sync <run_dir>/wandb/")
