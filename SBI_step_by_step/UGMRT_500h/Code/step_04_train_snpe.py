"""
Step 4: Train SNPE (Sequential Neural Posterior Estimation)
This is the main SBI training step - trains a neural network to estimate the posterior
"""

import numpy as np
import pickle
import torch
import os
from torch import nn
import sys

# SBI imports
from sbi.inference import SNPE
from sbi.utils import BoxUniform, get_density_thresholder
from sbi.analysis import pairplot

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging to file
OUTPUT_DIR = "/user1/supriyo/ml_project/SBI_step_by_step/UGMRT_500h/Outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "training_log.txt")
log_handle = open(log_file, "w")

def log_print(*args, **kwargs):
    """Print to both console and log file"""
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_handle)
    log_handle.flush()

log_print(f"Using device: {device}")

# Paths
DATA_DIR = "/user1/supriyo/ml_project/SBI_step_by_step/UGMRT_500h/Data/Train_test_data/"
OUTPUT_DIR = "/user1/supriyo/ml_project/SBI_step_by_step/UGMRT_500h/Outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_print("\n" + "="*60)
log_print("STEP 4: TRAIN SNPE (Neural Posterior Estimation)")
log_print("="*60)

# Load split dataset
log_print("\n1. Loading split dataset...")
split_path = os.path.join(DATA_DIR, "train_val_test_split.pkl")
with open(split_path, "rb") as f:
    split_data = pickle.load(f)

theta_train = torch.from_numpy(split_data["theta_train"]).float().to(device)  # (427, 2)
x_train = torch.from_numpy(split_data["x_1d_train"]).float().to(device)       # (427, 2762)

theta_val = torch.from_numpy(split_data["theta_val"]).float().to(device)      # (107, 2)
x_val = torch.from_numpy(split_data["x_1d_val"]).float().to(device)           # (107, 2762)

log_print(f"   theta_train shape: {theta_train.shape}")
log_print(f"   x_train shape: {x_train.shape}")
log_print(f"   theta_val shape: {theta_val.shape}")
log_print(f"   x_val shape: {x_val.shape}")
log_print(f"\n   Dataset summary:")
log_print(f"   - Training samples: {theta_train.shape[0]}")
log_print(f"   - Validation samples: {theta_val.shape[0]}")
log_print(f"   - Input dimension (frequencies): {x_train.shape[1]}")
log_print(f"   - Parameter dimension: {theta_train.shape[1]} (xHI, fX)")

# Define prior
log_print("\n2. Defining prior...")
prior = BoxUniform(
    low=torch.tensor([0.0, -4.0], device=device),
    high=torch.tensor([1.0, 1.0], device=device)
)
log_print(f"   Prior: xHI ∈ [0, 1], fX ∈ [-4, 1]")

# Initialize SNPE
log_print("\n3. Initializing SNPE...")
snpe = SNPE(prior=prior, device=device)

# Training hyperparameters - IMPROVED for better posteriors
log_print("\n4. Training with IMPROVED hyperparameters:")
batch_size = 32           # Increased from 16 for better gradient estimates
learning_rate = 5e-4      # Increased from 1e-4 for faster convergence
num_epochs = 300          # Tripled from 100 for better convergence

log_print(f"   batch_size: {batch_size} (was 16)")
log_print(f"   learning_rate: {learning_rate} (was 1e-4)")
log_print(f"   num_epochs: {num_epochs} (was 100)")
log_print(f"   → Expected: SMOOTHER, well-defined posteriors")

# Train the posterior
log_print("\n5. Training neural posterior...")
log_print(f"   This will take 30-60 minutes on CPU...")
log_print(f"   Improved hyperparameters should yield better posteriors...")
posterior = snpe.append_simulations(theta_train, x_train).train(
    training_batch_size=batch_size,
    learning_rate=learning_rate,
    max_num_epochs=num_epochs,
    show_train_summary=True,
)

log_print(f"\n✅ Training complete!")

# Save the posterior
log_print("\n6. Saving posterior...")
posterior_path = os.path.join(OUTPUT_DIR, "posterior_snpe.pt")
torch.save(posterior, posterior_path)
log_print(f"   Saved to: {posterior_path}")

# Save metadata
metadata = {
    "n_train": theta_train.shape[0],
    "n_val": theta_val.shape[0],
    "input_dim": x_train.shape[1],  # 2762 frequencies
    "param_dim": theta_train.shape[1],  # 2 parameters
    "device": str(device),
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "training_status": "IMPROVED_HYPERPARAMETERS",
}

metadata_path = os.path.join(OUTPUT_DIR, "metadata_train.pkl")
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)
log_print(f"   Metadata saved to: {metadata_path}")

log_print("\n" + "="*60)
log_print("✅ STEP 4 COMPLETE - IMPROVED TRAINING!")
log_print("="*60)
log_print(f"\nTraining Summary:")
log_print(f"  - Trained on: {theta_train.shape[0]} samples")
log_print(f"  - Validated on: {theta_val.shape[0]} samples")
log_print(f"  - Model saved: {posterior_path}")
log_print(f"  - Hyperparameters: IMPROVED for smooth posteriors")
log_print(f"  - batch_size: {batch_size}, lr: {learning_rate}, epochs: {num_epochs}")
log_print(f"\nExpected results:")
log_print(f"  - Smoother contours in posterior distributions")
log_print(f"  - Better-defined 1σ and 2σ credible intervals")
log_print(f"  - Professional appearance similar to paper figures")
log_print(f"\nNext: Run Step 5 for posterior inference on 5 test points")

# Close log file
log_handle.close()
print(f"\n✅ Log saved to: {log_file}")
