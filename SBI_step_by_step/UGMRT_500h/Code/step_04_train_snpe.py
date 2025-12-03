"""
Step 4: Train SNPE with 2D Spectral Data (Full Cubes)
Uses 2D brightness temperature cubes (1000 x 2762) for richer information
Much better than 1D mean spectra!
"""

import numpy as np
import pickle
import torch
import os
from torch import nn
import sys

# SBI imports
from sbi.inference import SNPE
from sbi.utils import BoxUniform

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

log_print("\n" + "="*70)
log_print("STEP 4: TRAIN SNPE WITH 2D SPECTRAL DATA (Full Cubes)")
log_print("="*70)

# Load split dataset
log_print("\n1. Loading split dataset...")
split_path = os.path.join(DATA_DIR, "train_val_test_split.pkl")
with open(split_path, "rb") as f:
    split_data = pickle.load(f)

theta_train = torch.from_numpy(split_data["theta_train"]).float().to(device)
x_train_1d = torch.from_numpy(split_data["x_1d_train"]).float().to(device)
x_train_2d = torch.from_numpy(split_data["x_2d_train"]).float().to(device)

theta_val = torch.from_numpy(split_data["theta_val"]).float().to(device)
x_val_1d = torch.from_numpy(split_data["x_1d_val"]).float().to(device)
x_val_2d = torch.from_numpy(split_data["x_2d_val"]).float().to(device)

log_print(f"   theta_train shape: {theta_train.shape}")
log_print(f"   x_train_1d shape: {x_train_1d.shape} (mean spectrum)")
log_print(f"   x_train_2d shape: {x_train_2d.shape} (full cube)")
log_print(f"   theta_val shape: {theta_val.shape}")
log_print(f"   x_val_2d shape: {x_val_2d.shape}")

# Flatten 2D data to 1D for neural network input
log_print("\n2. Flattening 2D spectral cubes...")
batch_size_train = x_train_2d.shape[0]
n_los = x_train_2d.shape[1]
n_freq = x_train_2d.shape[2]
total_features = n_los * n_freq

x_train_flat = x_train_2d.reshape(batch_size_train, -1)  # (427, 2762000)
x_val_flat = x_val_2d.reshape(x_val_2d.shape[0], -1)     # (107, 2762000)

log_print(f"   Flattened training: {x_train_flat.shape}")
log_print(f"   Flattened validation: {x_val_flat.shape}")
log_print(f"   Total features per sample: {total_features}")

# Apply dimensionality reduction - use mean spectrum + variance
log_print("\n3. Applying smart dimensionality reduction...")
x_train_mean = x_train_2d.mean(dim=1)  # (427, 2762)
x_train_std = x_train_2d.std(dim=1)    # (427, 2762)
x_val_mean = x_val_2d.mean(dim=1)      # (107, 2762)
x_val_std = x_val_2d.std(dim=1)        # (107, 2762)

# Concatenate mean and std (more info than just mean!)
x_train = torch.cat([x_train_mean, x_train_std], dim=1)  # (427, 5524)
x_val = torch.cat([x_val_mean, x_val_std], dim=1)        # (107, 5524)

log_print(f"   Mean + Std features: {x_train.shape}")
log_print(f"   - Mean spectrum: 2762 features")
log_print(f"   - Std spectrum: 2762 features")
log_print(f"   Total: 5524 features (rich representation!)")

# Normalize
log_print("\n4. Normalizing features...")
x_train_mean_val = x_train.mean(dim=0, keepdim=True)
x_train_std_val = x_train.std(dim=0, keepdim=True)
x_train_norm = (x_train - x_train_mean_val) / (x_train_std_val + 1e-8)
x_val_norm = (x_val - x_train_mean_val) / (x_train_std_val + 1e-8)

log_print(f"   ✓ Normalized training data")
log_print(f"   ✓ Normalized validation data")

log_print(f"\n   Dataset summary:")
log_print(f"   - Training samples: {theta_train.shape[0]}")
log_print(f"   - Validation samples: {theta_val.shape[0]}")
log_print(f"   - Input features: {x_train_norm.shape[1]} (Mean+Std of 2D cubes)")
log_print(f"   - Parameter dimension: {theta_train.shape[1]} (xHI, fX)")

# Define prior
log_print("\n5. Defining prior...")
prior = BoxUniform(
    low=torch.tensor([0.0, -4.0], device=device),
    high=torch.tensor([1.0, 1.0], device=device)
)
log_print(f"   Prior: xHI ∈ [0, 1], fX ∈ [-4, 1]")

# Initialize SNPE
log_print("\n6. Initializing SNPE...")
snpe = SNPE(prior=prior, device=device)

# Training hyperparameters - OPTIMIZED FOR 2D DATA
log_print("\n7. Training with OPTIMIZED hyperparameters:")
batch_size = 16          # Smaller batch (richer features)
learning_rate = 5e-4     # Conservative learning rate
num_epochs = 1000        # Very many epochs (complex data)

log_print(f"   batch_size: {batch_size}")
log_print(f"   learning_rate: {learning_rate}")
log_print(f"   num_epochs: {num_epochs}")
log_print(f"   Input data: 2D cubes (mean + std)")
log_print(f"   Feature dim: 5524 (2x richer than 1D mean)")
log_print(f"   Expected: SIGNIFICANTLY BETTER R²")

# Train the posterior
log_print("\n8. Training neural posterior with 2D spectral data...")
log_print(f"   This will take 2-4 hours on CPU...")
log_print(f"   Rich 2D data should yield MUCH BETTER posteriors...")

posterior = snpe.append_simulations(theta_train, x_train_norm).train(
    training_batch_size=batch_size,
    learning_rate=learning_rate,
    max_num_epochs=num_epochs,
    show_train_summary=True,
)

log_print(f"\n✅ Training complete!")

# Save the posterior
log_print("\n9. Saving posterior...")
posterior_path = os.path.join(OUTPUT_DIR, "posterior_snpe_2d.pt")
torch.save(posterior, posterior_path)
log_print(f"   Posterior saved to: {posterior_path}")

# Save normalization and feature parameters
log_print("\n10. Saving normalization parameters...")
feature_params = {
    "x_mean": x_train_mean_val.cpu().numpy(),
    "x_std": x_train_std_val.cpu().numpy(),
    "n_los": n_los,
    "n_freq": n_freq,
    "feature_type": "mean_and_std_from_2d_cubes",
}
feature_path = os.path.join(OUTPUT_DIR, "feature_params_2d.pkl")
with open(feature_path, "wb") as f:
    pickle.dump(feature_params, f)
log_print(f"   Saved to: {feature_path}")

# Save metadata
metadata = {
    "n_train": theta_train.shape[0],
    "n_val": theta_val.shape[0],
    "input_dim": x_train_norm.shape[1],
    "param_dim": theta_train.shape[1],
    "device": str(device),
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "data_source": "2D_SPECTRAL_CUBES",
    "n_los": n_los,
    "n_freq": n_freq,
    "architecture": "MEAN_AND_STD_2D",
    "training_status": "2D_DATA_TRAINING",
    "expected_improvement": "R² should be MUCH better (>0.7 for xHI, >0.85 for fX)",
}

metadata_path = os.path.join(OUTPUT_DIR, "metadata_train.pkl")
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)
log_print(f"   Metadata saved to: {metadata_path}")

metadata_path = os.path.join(OUTPUT_DIR, "metadata_train.pkl")
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)
log_print(f"   Metadata saved to: {metadata_path}")

log_print("\n" + "="*70)
log_print("✅ STEP 4 COMPLETE - 2D DATA TRAINING!")
log_print("="*70)
log_print(f"\nTraining Summary:")
log_print(f"  - Trained on: {theta_train.shape[0]} samples")
log_print(f"  - Validated on: {theta_val.shape[0]} samples")
log_print(f"  - Model saved: {posterior_path}")
log_print(f"  - Input dimension: {x_train_norm.shape[1]} features (Mean + Std of 2D cubes)")
log_print(f"  - batch_size: {batch_size}, lr: {learning_rate}, epochs: {num_epochs}")

log_print(f"\nKey Improvements:")
log_print(f"  ✓ 2D spectral cubes (1000 x 2762) instead of 1D mean")
log_print(f"  ✓ Mean + Std features capture distribution information")
log_print(f"  ✓ 5524 features (2x more than 1D mean)")
log_print(f"  ✓ Much richer information for learning")
log_print(f"  ✓ Very many epochs (1000) for convergence")

log_print(f"\nWhy 2D Data is Better:")
log_print(f"  - Contains line-of-sight variance information")
log_print(f"  - Captures texture/structure in 21cm signal")
log_print(f"  - Mean+Std gives distribution shape, not just mean value")
log_print(f"  - Helps network distinguish parameter differences better")

log_print(f"\nTarget Metrics:")
log_print(f"  - xHI R²: >0.70 (was 0.31)")
log_print(f"  - fX R²: >0.85 (was 0.69)")

log_print(f"\nNext: Run Step 4b evaluation to check improvements")

# Close log file
log_handle.close()
print(f"\n✅ Log saved to: {log_file}")
