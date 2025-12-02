"""
Step 3b: Split Dataset into Train/Validation/Test
Load the saved dataset and divide it into separate sets.
Test set: 5 specific diverse points (like in the paper)
Validation set: Random 20% of remaining
Training set: 80% of remaining
"""

import numpy as np
import pickle
import os

# Paths
DATA_DIR = "/user1/supriyo/ml_project/SBI_step_by_step/UGMRT_500h/Data/Full_data_set"
OUTPUT_DIR = "/user1/supriyo/ml_project/SBI_step_by_step/UGMRT_500h/Data/Train_test_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Open log file
log_file = os.path.join(OUTPUT_DIR, "split_data_log.txt")
log_fp = open(log_file, "w")

def log_print(msg):
    """Print to both console and log file"""
    print(msg)
    log_fp.write(msg + "\n")
    log_fp.flush()

log_print("=" * 60)
log_print("STEP 3b: SPLIT DATA INTO TRAIN/VAL/TEST")
log_print("=" * 60)

# Load dataset
log_print("\n1. Loading dataset...")
with open(os.path.join(DATA_DIR, "dataset.pkl"), "rb") as f:
    data = pickle.load(f)

theta = data["theta"]  # (539, 2)
x_1d = data["x_1d"]    # (539, 2762)
x_2d = data["x_2d"]    # (539, 1000, 2762)

log_print(f"   Total samples: {len(theta)}")
log_print(f"   theta shape: {theta.shape}")
log_print(f"   x_1d shape: {x_1d.shape}")
log_print(f"   x_2d shape: {x_2d.shape}")

log_print(f"\n   Data ranges:")
log_print(f"     xHI: [{theta[:, 0].min():.4f}, {theta[:, 0].max():.4f}]")
log_print(f"     fX:  [{theta[:, 1].min():.4f}, {theta[:, 1].max():.4f}]")

# Step 2: Select EXACT 5 test points from the paper
# From Figure 2: {(0.11, -3.0), (0.11, -1.0), (0.52, -2.0), (0.80, -1.0), (0.80, -3.0)}
log_print("\n2. Selecting EXACT 5 test points from paper...")

# These are the EXACT target points from the paper
target_test_points = np.array([
    [0.11, -3.0],
    [0.11, -1.0],
    [0.52, -2.0],
    [0.80, -1.0],
    [0.80, -3.0]
], dtype=np.float32)

log_print(f"\n   Target points from paper:")
for i, pt in enumerate(target_test_points, 1):
    log_print(f"     {i}. xHI={pt[0]:.2f}, fX={pt[1]:.1f}")

# Find EXACT matches in the dataset
test_indices = []
test_matches_info = []

log_print(f"\n   Finding EXACT matches in dataset...")
for target_pt in target_test_points:
    # Calculate Euclidean distance to all points
    distances = np.sqrt((theta[:, 0] - target_pt[0])**2 + (theta[:, 1] - target_pt[1])**2)
    closest_idx = np.argmin(distances)
    distance = distances[closest_idx]
    closest_pt = theta[closest_idx]
    
    test_indices.append(closest_idx)
    test_matches_info.append({
        'target': target_pt,
        'found': closest_pt,
        'distance': distance,
        'index': closest_idx
    })
    
    log_print(f"\n   Target: xHI={target_pt[0]:.2f}, fX={target_pt[1]:.1f}")
    log_print(f"   Found:  xHI={closest_pt[0]:.4f}, fX={closest_pt[1]:.4f} (idx={closest_idx}, diff={distance:.6f})")

test_indices = np.array(test_indices)
log_print(f"\n   Final test indices: {test_indices}")

# Step 3: Separate train/val/test
log_print("\n3. Creating train/validation/test split...")

# Get all indices except test indices
all_indices = np.arange(len(theta))
train_val_indices = np.delete(all_indices, test_indices)
n_train_val = len(train_val_indices)

# Split train_val into 80% train, 20% validation
np.random.seed(42)  # For reproducibility
np.random.shuffle(train_val_indices)
n_train = int(0.8 * n_train_val)
train_indices = train_val_indices[:n_train]
val_indices = train_val_indices[n_train:]

# Extract data
theta_train = theta[train_indices]
x_1d_train = x_1d[train_indices]
x_2d_train = x_2d[train_indices]

theta_val = theta[val_indices]
x_1d_val = x_1d[val_indices]
x_2d_val = x_2d[val_indices]

theta_test = theta[test_indices]
x_1d_test = x_1d[test_indices]
x_2d_test = x_2d[test_indices]

log_print(f"\n   Train set:      {len(train_indices)} samples ({100*len(train_indices)/len(theta):.1f}%)")
log_print(f"   Validation set: {len(val_indices)} samples ({100*len(val_indices)/len(theta):.1f}%)")
log_print(f"   Test set:       {len(test_indices)} samples ({100*len(test_indices)/len(theta):.1f}%)")
log_print(f"   Total:          {len(train_indices) + len(val_indices) + len(test_indices)} samples")

# Step 4: Save split data
log_print("\n4. Saving split data...")

split_data = {
    "theta_train": theta_train,
    "x_1d_train": x_1d_train,
    "x_2d_train": x_2d_train,
    "theta_val": theta_val,
    "x_1d_val": x_1d_val,
    "x_2d_val": x_2d_val,
    "theta_test": theta_test,
    "x_1d_test": x_1d_test,
    "x_2d_test": x_2d_test,
    "train_indices": train_indices,
    "val_indices": val_indices,
    "test_indices": test_indices,
}

split_path = os.path.join(OUTPUT_DIR, "train_val_test_split.pkl")
with open(split_path, "wb") as f:
    pickle.dump(split_data, f)

log_print(f"   Saved to: {split_path}")
log_print(f"   File size: {os.path.getsize(split_path) / 1e9:.2f} GB")

# Also save as individual numpy files
log_print("\n   Saving individual .npy files...")
np.save(os.path.join(OUTPUT_DIR, "theta_train.npy"), theta_train)
np.save(os.path.join(OUTPUT_DIR, "x_1d_train.npy"), x_1d_train)
np.save(os.path.join(OUTPUT_DIR, "x_2d_train.npy"), x_2d_train)

np.save(os.path.join(OUTPUT_DIR, "theta_val.npy"), theta_val)
np.save(os.path.join(OUTPUT_DIR, "x_1d_val.npy"), x_1d_val)
np.save(os.path.join(OUTPUT_DIR, "x_2d_val.npy"), x_2d_val)

np.save(os.path.join(OUTPUT_DIR, "theta_test.npy"), theta_test)
np.save(os.path.join(OUTPUT_DIR, "x_1d_test.npy"), x_1d_test)
np.save(os.path.join(OUTPUT_DIR, "x_2d_test.npy"), x_2d_test)

log_print(f"   Individual files saved!")

# Summary table
log_print("\n" + "=" * 60)
log_print("SUMMARY: Train/Val/Test Split (with 1D and 2D data)")
log_print("=" * 60)

log_print(f"\nTraining Set ({len(train_indices)} samples):")
log_print(f"  theta: {theta_train.shape}")
log_print(f"  x_1d: {x_1d_train.shape}")
log_print(f"  x_2d: {x_2d_train.shape}")
log_print(f"  xHI range: [{theta_train[:, 0].min():.4f}, {theta_train[:, 0].max():.4f}]")
log_print(f"  fX range:  [{theta_train[:, 1].min():.4f}, {theta_train[:, 1].max():.4f}]")

log_print(f"\nValidation Set ({len(val_indices)} samples):")
log_print(f"  theta: {theta_val.shape}")
log_print(f"  x_1d: {x_1d_val.shape}")
log_print(f"  x_2d: {x_2d_val.shape}")
log_print(f"  xHI range: [{theta_val[:, 0].min():.4f}, {theta_val[:, 0].max():.4f}]")
log_print(f"  fX range:  [{theta_val[:, 1].min():.4f}, {theta_val[:, 1].max():.4f}]")

log_print("\nTest Set ({} samples):".format(len(test_indices)))
for i, match_info in enumerate(test_matches_info, 1):
    found_pt = match_info['found']
    log_print(f"  Point {i}: xHI={found_pt[0]:.4f}, fX={found_pt[1]:.4f}")

log_print(f"\n✅ SPLIT COMPLETE!")
log_print(f"   Ready for training with train_val_test_split.pkl")

# Close log file
log_fp.close()
print(f"\n✅ Log saved to: {log_file}")
