"""
Step 3: Define the Prior for SBI
The prior defines the range of parameters we're interested in
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sbi.utils import BoxUniform
import torch

# Load the saved dataset to check data ranges
data_path = "/user1/supriyo/ml_project/SBI_step_by_step/UGMRT_500h/Data/dataset.pkl"
with open(data_path, "rb") as f:
    data = pickle.load(f)

theta = data["theta"]

print("=" * 60)
print("STEP 3: DEFINE PRIOR")
print("=" * 60)

# Our parameters: [xHI, fX]
# xHI = Neutral fraction (ionization state)
# fX = log10(X-ray heating efficiency)

print("\nData ranges observed:")
print(f"  xHI: [{theta[:, 0].min():.4f}, {theta[:, 0].max():.4f}]")
print(f"  fX:  [{theta[:, 1].min():.4f}, {theta[:, 1].max():.4f}]")

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

# Visualization: Compare prior coverage vs actual data
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: xHI histogram
axes[0].hist(theta[:, 0], bins=30, alpha=0.6, label='Data', color='steelblue', density=True)
axes[0].hist(prior_samples[:, 0], bins=30, alpha=0.4, label='Prior', color='red', density=True)
axes[0].axvline(prior_xhi_min, color='red', linestyle='--', linewidth=2, label='Prior bounds')
axes[0].axvline(prior_xhi_max, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('xHI (Neutral Fraction)', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('Prior Coverage: xHI', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: fX histogram
axes[1].hist(theta[:, 1], bins=30, alpha=0.6, label='Data', color='coral', density=True)
axes[1].hist(prior_samples[:, 1], bins=30, alpha=0.4, label='Prior', color='red', density=True)
axes[1].axvline(prior_fx_min, color='red', linestyle='--', linewidth=2, label='Prior bounds')
axes[1].axvline(prior_fx_max, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('fX (log10 X-ray heating)', fontsize=11)
axes[1].set_ylabel('Density', fontsize=11)
axes[1].set_title('Prior Coverage: fX', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Plot 3: 2D scatter
axes[2].scatter(prior_samples[:, 0], prior_samples[:, 1], alpha=0.1, s=10, color='red', label='Prior samples')
axes[2].scatter(theta[:, 0], theta[:, 1], alpha=0.6, s=30, color='steelblue', label='Data', edgecolors='black', linewidth=0.5)
axes[2].set_xlabel('xHI (Neutral Fraction)', fontsize=11)
axes[2].set_ylabel('fX (log10 X-ray heating)', fontsize=11)
axes[2].set_title('Parameter Space: Data vs Prior', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)
axes[2].set_xlim([prior_xhi_min, prior_xhi_max])
axes[2].set_ylim([prior_fx_min, prior_fx_max])

plt.tight_layout()
plt.savefig('/user1/supriyo/ml_project/SBI_step_by_step/UGMRT_500h/Outputs/prior_coverage.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Saved visualization: prior_coverage.png")

# Save the prior for later use
prior_dict = {
    "prior": prior,
    "bounds": {
        "xhi": [prior_xhi_min, prior_xhi_max],
        "fx": [prior_fx_min, prior_fx_max]
    }
}

import pickle
with open("/user1/supriyo/ml_project/SBI_step_by_step/UGMRT_500h/Data/prior.pkl", "wb") as f:
    pickle.dump(prior_dict, f)

print(f"\n✅ Saved prior: /user1/supriyo/ml_project/SBI_step_by_step/UGMRT_500h/Data/prior.pkl")

print("\n" + "=" * 60)
print("✅ PRIOR DEFINED!")
print("=" * 60)
print("\nNext step: Step 4 - Train SNPE model")
print("  We'll train the neural network to learn the posterior")
