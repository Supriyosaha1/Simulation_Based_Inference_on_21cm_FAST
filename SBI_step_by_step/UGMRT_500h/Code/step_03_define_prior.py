"""
Step 3: Define the Prior for SBI
The prior defines the range of parameters we're interested in
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sbi.utils import BoxUniform
import torch
import os
from config import ROOT_DIR

# Paths
OUTPUTS_DIR = os.path.join(ROOT_DIR, "UGMRT_500h", "Outputs")
PRIOR_DIR = os.path.join(OUTPUTS_DIR, "Prior_distribution")
PRIOR_FILE = os.path.join(PRIOR_DIR, "prior.pkl")
os.makedirs(PRIOR_DIR, exist_ok=True)

print("=" * 60)
print("STEP 3: DEFINE PRIOR")
print("=" * 60)

# Our parameters: [xHI, fX]
# xHI = Neutral fraction (ionization state)
# fX = log10(X-ray heating efficiency)

# Define the prior as a BoxUniform (uniform distribution in a box)
# We'll use slightly wider bounds than observed to allow for some extrapolation
prior_xhi_min = 0.0      # Can go from completely ionized
prior_xhi_max = 1.0      # To completely neutral
prior_fx_min = -4.0      # Min X-ray heating
prior_fx_max = 1.0       # Max X-ray heating

# Create prior using SBI's BoxUniform
low = torch.tensor([prior_xhi_min, prior_fx_min], dtype=torch.float32)
high = torch.tensor([prior_xhi_max, prior_fx_max], dtype=torch.float32)

prior = BoxUniform(low=low, high=high)

print("\nPrior definition:")
print(f"  xHI: [{prior_xhi_min}, {prior_xhi_max}]")
print(f"  fX:  [{prior_fx_min}, {prior_fx_max}]")

# Sample from prior to verify
prior_samples = prior.sample((10000,))
prior_samples = prior_samples.numpy()

print(f"\nPrior samples (10,000 samples):")
print(f"  xHI: [{prior_samples[:, 0].min():.4f}, {prior_samples[:, 0].max():.4f}]")
print(f"  fX:  [{prior_samples[:, 1].min():.4f}, {prior_samples[:, 1].max():.4f}]")

# Visualization: Prior distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: xHI histogram
axes[0].hist(prior_samples[:, 0], bins=30, alpha=0.6, label='Prior', color='red', density=True)
axes[0].axvline(prior_xhi_min, color='darkred', linestyle='--', linewidth=2, label='Prior bounds')
axes[0].axvline(prior_xhi_max, color='darkred', linestyle='--', linewidth=2)
axes[0].set_xlabel('xHI (Neutral Fraction)', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('Prior: xHI', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: fX histogram
axes[1].hist(prior_samples[:, 1], bins=30, alpha=0.6, label='Prior', color='red', density=True)
axes[1].axvline(prior_fx_min, color='darkred', linestyle='--', linewidth=2, label='Prior bounds')
axes[1].axvline(prior_fx_max, color='darkred', linestyle='--', linewidth=2)
axes[1].set_xlabel('fX (log10 X-ray heating)', fontsize=11)
axes[1].set_ylabel('Density', fontsize=11)
axes[1].set_title('Prior: fX', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PRIOR_DIR, 'prior_coverage.png'), dpi=150, bbox_inches='tight')

print("\n✅ Saved visualization: prior_coverage.png")

# Save the prior for later use
prior_dict = {
    "prior": prior,
    "bounds": {
        "xhi": [prior_xhi_min, prior_xhi_max],
        "fx": [prior_fx_min, prior_fx_max]
    }
}

with open(PRIOR_FILE, "wb") as f:
    pickle.dump(prior_dict, f)

print(f"\n✅ Saved prior: {PRIOR_FILE}")

print("\n" + "=" * 60)
print("✅ PRIOR DEFINED!")
print("=" * 60)
print("\nNext step: Step 4 - Train SNPE model")
print("  We'll train the neural network to learn the posterior")
