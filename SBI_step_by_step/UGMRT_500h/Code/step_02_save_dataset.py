"""
Step 2: Load data and save to disk (1D spectra, 2D cubes, and parameters)
This way we don't need to load from .dat files every time
"""

import numpy as np
import pickle
import os
from glob import glob

# Paths
DATA_ROOT = "/user1/supriyo/ml_project/21cmFAST_los/F21_noisy/"
OUTPUT_DIR = "/user1/supriyo/ml_project/SBI_step_by_step/UGMRT_500h/Data/"
FILE_PATTERN = "F21_noisy*uGMRT_8kHz_t500h*.dat"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_one_file(path):
    """Load a single .dat file and extract parameters + spectra"""
    with open(path, "rb") as f:
        # Read header (5 floats)
        header = np.fromfile(f, dtype=np.float32, count=5)
        z, xHI, fX, N_los_f, N_freq_f = header
        N_los, N_freq = int(N_los_f), int(N_freq_f)
        
        # Read frequency axis (2762 floats)
        freqs = np.fromfile(f, dtype=np.float32, count=N_freq)
        
        # Read data cube flattened (1000 * 2762 floats)
        data_flat = np.fromfile(f, dtype=np.float32, count=N_los * N_freq)
    
    # Reshape to 2D cube
    cube = data_flat.reshape(N_los, N_freq)
    
    # Compute 1D mean spectrum
    mean_spec = cube.mean(axis=0).astype(np.float32)
    
    return float(xHI), float(fX), mean_spec, cube.astype(np.float32)

# Find all files
files = sorted(glob(os.path.join(DATA_ROOT, FILE_PATTERN)))
n_files = len(files)
print(f"Found {n_files} files")

# Arrays to store data
theta_list = []  # Parameters: [xHI, fX]
x_1d_list = []   # 1D mean spectra
x_2d_list = []   # 2D full cubes

# Load all files
print("\nLoading files...")
for i, filepath in enumerate(files):
    if (i+1) % 100 == 0 or i == 0:
        print(f"  Loaded {i+1}/{n_files}")
    
    xHI, fX, mean_spec, cube = load_one_file(filepath)
    theta_list.append([xHI, fX])
    x_1d_list.append(mean_spec)
    x_2d_list.append(cube)

print(f"Done loading {n_files} files!\n")

# Convert to numpy arrays
theta = np.array(theta_list, dtype=np.float32)
x_1d = np.array(x_1d_list, dtype=np.float32)
x_2d = np.array(x_2d_list, dtype=np.float32)

print("Data shapes:")
print(f"  theta: {theta.shape}")
print(f"  x_1d (mean spectra): {x_1d.shape}")
print(f"  x_2d (full cubes): {x_2d.shape}")

print(f"\nData ranges:")
print(f"  xHI: [{theta[:, 0].min():.4f}, {theta[:, 0].max():.4f}]")
print(f"  fX:  [{theta[:, 1].min():.4f}, {theta[:, 1].max():.4f}]")

# Save everything as pickle (easy to load back)
output_file = os.path.join(OUTPUT_DIR, "dataset.pkl")
data_dict = {
    "theta": theta,
    "x_1d": x_1d,  # 1D mean spectra (539, 2762)
    "x_2d": x_2d,  # 2D full cubes (539, 1000, 2762)
}

with open(output_file, "wb") as f:
    pickle.dump(data_dict, f)

print(f"\n✅ Saved to: {output_file}")
print(f"   File size: {os.path.getsize(output_file) / 1e9:.2f} GB")

# Also save as individual numpy files for flexibility
np.save(os.path.join(OUTPUT_DIR, "theta.npy"), theta)
np.save(os.path.join(OUTPUT_DIR, "x_1d.npy"), x_1d)
np.save(os.path.join(OUTPUT_DIR, "x_2d.npy"), x_2d)

print(f"\n✅ Also saved individual .npy files:")
print(f"   - theta.npy ({os.path.getsize(os.path.join(OUTPUT_DIR, 'theta.npy')) / 1e6:.2f} MB)")
print(f"   - x_1d.npy ({os.path.getsize(os.path.join(OUTPUT_DIR, 'x_1d.npy')) / 1e6:.2f} MB)")
print(f"   - x_2d.npy ({os.path.getsize(os.path.join(OUTPUT_DIR, 'x_2d.npy')) / 1e6:.2f} MB)")

print(f"\n✅ DATASET SAVED SUCCESSFULLY!")
print(f"\nTo load in future:")
print(f'  import pickle')
print(f'  with open("{output_file}", "rb") as f:')
print(f'      data = pickle.load(f)')
print(f'  theta, x_1d, x_2d = data["theta"], data["x_1d"], data["x_2d"]')
