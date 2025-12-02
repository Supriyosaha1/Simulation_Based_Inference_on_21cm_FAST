"""
STEP 1: Load uGMRT_500h dataset
Simple and clear - no fancy stuff

File structure:
- Header (5 floats): z, xHI, fX, N_los, N_freq
- Frequency axis (N_freq floats): frequencies in Hz
- Data (N_los × N_freq floats): brightness temperature cube flattened
"""

import numpy as np
import glob
import os

# ============================================================================
# PATHS
# ============================================================================
RAW_DATA_DIR = "/user1/supriyo/ml_project/21cmFAST_los/F21_noisy"
FILE_PATTERN = "F21_noisy*uGMRT_8kHz_t500h*.dat"  # Only F21_noisy files, not 1DPS

# ============================================================================
# FUNCTION: Load one file
# ============================================================================

def load_one_file(path):
    """
    Load one .dat file and extract parameters + mean spectrum
    
    File structure:
      [5 floats header] + [N_freq floats frequencies] + [N_los × N_freq floats data]
    """
    
    with open(path, "rb") as f:
        # Read header (5 floats)
        header = np.fromfile(f, dtype=np.float32, count=5)
        z, xHI, fX, N_los_f, N_freq_f = header
        N_los = int(N_los_f)   # Number of lines of sight (1000)
        N_freq = int(N_freq_f) # Number of frequencies (2762)
        
        # Read frequency axis (N_freq floats)
        freqs = np.fromfile(f, dtype=np.float32, count=N_freq)
        
        # Read brightness temperature data (N_los × N_freq floats)
        data_flat = np.fromfile(f, dtype=np.float32, count=N_los * N_freq)
    
    # Reshape data into cube: (N_los, N_freq)
    cube = data_flat.reshape(N_los, N_freq)
    
    # Average across all lines of sight to get 1D spectrum
    mean_spec = cube.mean(axis=0).astype(np.float32)
    
    return float(xHI), float(fX), mean_spec


# ============================================================================
# MAIN
# ============================================================================

print("\n" + "="*70)
print("STEP 1: LOAD DATA")
print("="*70)

# Find all files
all_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, FILE_PATTERN)))
print(f"\nFound {len(all_files)} files\n")

if len(all_files) == 0:
    print("ERROR: No files found!")
    exit()

# Load all files
print("Loading files...")
theta_list = []
x_list = []

for i, path in enumerate(all_files):
    xHI, fX, mean_spec = load_one_file(path)
    theta_list.append([xHI, fX])
    x_list.append(mean_spec)
    
    if (i+1) % 100 == 0:
        print(f"  Loaded {i+1}/{len(all_files)}")

# Convert to numpy arrays
theta = np.array(theta_list, dtype=np.float32)
x = np.array(x_list, dtype=np.float32)

print(f"\nDone!\n")

# ============================================================================
# SHOW RESULTS
# ============================================================================

print("="*70)
print("DATASET LOADED")
print("="*70)

print(f"\nTotal files: {len(theta)}")
print(f"theta shape: {theta.shape}  (parameters: xHI, fX)")
print(f"x shape: {x.shape}  (spectra: mean brightness)")

print(f"\nxHI range: [{theta[:,0].min():.4f}, {theta[:,0].max():.4f}]")
print(f"fX range:  [{theta[:,1].min():.4f}, {theta[:,1].max():.4f}]")

print(f"\nFirst 5 samples:")
print(f"{'#':<5} {'xHI':<12} {'fX':<12}")
for i in range(min(5, len(theta))):
    print(f"{i:<5} {theta[i,0]:<12.4f} {theta[i,1]:<12.4f}")

print(f"\n" + "="*70)
print("✅ DATA LOADED SUCCESSFULLY")
print("="*70 + "\n")


