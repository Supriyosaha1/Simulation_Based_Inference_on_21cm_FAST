"""
Step 5: Posterior Inference on 5 Test Points
Sample from the trained posterior and analyze results
"""

import numpy as np
import pickle
import torch
import os
import matplotlib.pyplot as plt
from config import ROOT_DIR

# Setup paths
DATA_DIR = os.path.join(ROOT_DIR, "UGMRT_500h", "Data", "Train_test_data")
TRAIN_OUTPUTS_DIR = os.path.join(ROOT_DIR, "UGMRT_500h", "Outputs", "Train_outputs")
TEST_OUTPUTS_DIR = os.path.join(ROOT_DIR, "UGMRT_500h", "Outputs", "Test_outputs")
os.makedirs(TEST_OUTPUTS_DIR, exist_ok=True)

# Setup logging
log_file = os.path.join(TEST_OUTPUTS_DIR, "inference_log.txt")
log_fp = open(log_file, "w")

def log_print(msg):
    """Print to both console and log file"""
    print(msg)
    log_fp.write(msg + "\n")
    log_fp.flush()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_print("="*60)
log_print("STEP 5: POSTERIOR INFERENCE ON 5 TEST POINTS")
log_print("="*60)

# Load trained posterior
log_print("\n2. Loading trained posterior...")
posterior_path = os.path.join(TRAIN_OUTPUTS_DIR, "posterior_snpe_2d.pt")
posterior = torch.load(posterior_path, map_location=device, weights_only=False)
log_print(f"   Loaded from: {posterior_path}")

# Load test set (5 EXACT points)
log_print("\n2. Loading test set (5 EXACT points from paper)...")
split_path = os.path.join(DATA_DIR, "train_val_test_split.pkl")
with open(split_path, "rb") as f:
    split_data = pickle.load(f)

theta_test = split_data["theta_test"]  # (5, 2)
x_test = torch.from_numpy(split_data["x_1d_test"]).float().to(device)  # (5, 2762)

log_print(f"   Test set size: {theta_test.shape[0]}")
log_print(f"   theta_test shape: {theta_test.shape}")
log_print(f"   x_test shape: {x_test.shape}")

log_print(f"\n   Test points:")
for i in range(len(theta_test)):
    log_print(f"     {i+1}. xHI={theta_test[i, 0]:.4f}, fX={theta_test[i, 1]:.4f}")

# Load feature normalization parameters and extract 2D features
log_print("\n   Loading feature parameters for 2D normalization...")
feature_path = os.path.join(TRAIN_OUTPUTS_DIR, "feature_params_2d.pkl")
with open(feature_path, "rb") as f:
    feature_params = pickle.load(f)
x_mean = torch.from_numpy(feature_params["x_mean"]).float().to(device)
x_std = torch.from_numpy(feature_params["x_std"]).float().to(device)

# Load 2D test data and extract mean+std features
x_test_2d = torch.from_numpy(split_data["x_2d_test"]).float().to(device)  # (5, 1000, 2762)
x_test_mean = x_test_2d.mean(dim=1)  # (5, 2762)
x_test_std = x_test_2d.std(dim=1)    # (5, 2762)
x_test_features = torch.cat([x_test_mean, x_test_std], dim=1)  # (5, 5524)
x_test = (x_test_features - x_mean) / (x_std + 1e-8)  # Normalize
log_print(f"   Using 2D features (mean+std): shape {x_test.shape}")

# Sample from posterior on test set
log_print("\n3. Sampling from posterior on 5 test points...")
n_samples = 2000
log_print(f"   Drawing {n_samples} samples per test point...")

posterior_samples = []
posterior_means = []
posterior_stds = []

for i in range(len(theta_test)):
    x_condition = x_test[i:i+1]  # (1, 2762)
    
    try:
        samples = posterior.sample((n_samples,), x=x_condition)
    except TypeError:
        samples = posterior.sample((n_samples,), condition=x_condition)
    
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    samples = samples.reshape(-1, 2)  # (n_samples, 2)
    
    posterior_samples.append(samples)
    posterior_means.append(samples.mean(axis=0))
    posterior_stds.append(samples.std(axis=0))
    
    log_print(f"   Point {i+1}: Posterior mean xHI={samples[:, 0].mean():.4f}, fX={samples[:, 1].mean():.4f}")

posterior_means = np.array(posterior_means)  # (5, 2)
posterior_stds = np.array(posterior_stds)    # (5, 2)

# Compute statistics
log_print("\n4. Computing inference statistics...")

log_print("\n" + "="*60)
log_print("INFERENCE RESULTS FOR 5 TEST POINTS")
log_print("="*60)

for i in range(len(theta_test)):
    true_pt = theta_test[i]
    post_mean = posterior_means[i]
    post_std = posterior_stds[i]
    samples = posterior_samples[i]
    
    # Compute 1-sigma (68%) credible intervals
    ci_1sig_low_xhi = np.percentile(samples[:, 0], 16)
    ci_1sig_high_xhi = np.percentile(samples[:, 0], 84)
    ci_1sig_low_fx = np.percentile(samples[:, 1], 16)
    ci_1sig_high_fx = np.percentile(samples[:, 1], 84)
    
    # Compute 2-sigma (95%) credible intervals
    ci_2sig_low_xhi = np.percentile(samples[:, 0], 2.5)
    ci_2sig_high_xhi = np.percentile(samples[:, 0], 97.5)
    ci_2sig_low_fx = np.percentile(samples[:, 1], 2.5)
    ci_2sig_high_fx = np.percentile(samples[:, 1], 97.5)
    
    # Compute z-scores
    z_xhi = (true_pt[0] - post_mean[0]) / (post_std[0] + 1e-8)
    z_fx = (true_pt[1] - post_mean[1]) / (post_std[1] + 1e-8)
    
    log_print(f"\n{'='*60}")
    log_print(f"Test Point {i+1}")
    log_print(f"{'='*60}")
    log_print(f"True parameters:      xHI={true_pt[0]:.4f}, fX={true_pt[1]:.4f}")
    log_print(f"Posterior mean:       xHI={post_mean[0]:.4f}, fX={post_mean[1]:.4f}")
    log_print(f"Posterior std:        xHI={post_std[0]:.4f}, fX={post_std[1]:.4f}")
    log_print(f"68% CI (1σ) xHI: [{ci_1sig_low_xhi:.4f}, {ci_1sig_high_xhi:.4f}]")
    log_print(f"68% CI (1σ) fX:  [{ci_1sig_low_fx:.4f}, {ci_1sig_high_fx:.4f}]")
    log_print(f"95% CI (2σ) xHI: [{ci_2sig_low_xhi:.4f}, {ci_2sig_high_xhi:.4f}]")
    log_print(f"95% CI (2σ) fX:  [{ci_2sig_low_fx:.4f}, {ci_2sig_high_fx:.4f}]")
    log_print(f"Z-scores:             xHI={z_xhi:.4f}, fX={z_fx:.4f}")
    
    # Check if true value is within credible intervals
    xhi_in_1sig = ci_1sig_low_xhi <= true_pt[0] <= ci_1sig_high_xhi
    fx_in_1sig = ci_1sig_low_fx <= true_pt[1] <= ci_1sig_high_fx
    xhi_in_2sig = ci_2sig_low_xhi <= true_pt[0] <= ci_2sig_high_xhi
    fx_in_2sig = ci_2sig_low_fx <= true_pt[1] <= ci_2sig_high_fx
    log_print(f"True value in 68% CI: xHI={xhi_in_1sig}, fX={fx_in_1sig}")
    log_print(f"True value in 95% CI: xHI={xhi_in_2sig}, fX={fx_in_2sig}")

# Create comprehensive visualization
log_print("\n5. Creating inference visualization...")

# First compute all credible intervals for plotting
ci_1sig_xhi = np.array([[np.percentile(s[:, 0], 16), np.percentile(s[:, 0], 84)] for s in posterior_samples])
ci_2sig_xhi = np.array([[np.percentile(s[:, 0], 2.5), np.percentile(s[:, 0], 97.5)] for s in posterior_samples])
ci_1sig_fx = np.array([[np.percentile(s[:, 1], 16), np.percentile(s[:, 1], 84)] for s in posterior_samples])
ci_2sig_fx = np.array([[np.percentile(s[:, 1], 2.5), np.percentile(s[:, 1], 97.5)] for s in posterior_samples])

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: True vs Posterior mean xHI with 1σ and 2σ
x_points = np.array(range(1, 6))
# 2σ first (background)
axes[0, 0].errorbar(x_points, posterior_means[:, 0], 
                    yerr=[posterior_means[:, 0] - ci_2sig_xhi[:, 0], ci_2sig_xhi[:, 1] - posterior_means[:, 0]], 
                    fmt='none', ecolor='lightblue', alpha=0.8, capsize=8, capthick=2, elinewidth=6, label='95% CI (2σ)')
# 1σ on top
axes[0, 0].errorbar(x_points, posterior_means[:, 0], 
                    yerr=[posterior_means[:, 0] - ci_1sig_xhi[:, 0], ci_1sig_xhi[:, 1] - posterior_means[:, 0]], 
                    fmt='none', ecolor='blue', alpha=0.8, capsize=5, capthick=2, elinewidth=3, label='68% CI (1σ)')
axes[0, 0].scatter(x_points, theta_test[:, 0], color='red', s=150, label='True', marker='*', zorder=5)
axes[0, 0].scatter(x_points, posterior_means[:, 0], color='blue', s=80, label='Posterior mean', marker='o', zorder=5)
axes[0, 0].set_xlabel('Test Point', fontsize=11)
axes[0, 0].set_ylabel('xHI', fontsize=11)
axes[0, 0].set_title('xHI: True vs Posterior', fontweight='bold')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend(loc='best')
axes[0, 0].set_xticks(range(1, 6))

# Plot 2: True vs Posterior mean fX with 1σ and 2σ
# 2σ first (background)
axes[0, 1].errorbar(x_points, posterior_means[:, 1], 
                    yerr=[posterior_means[:, 1] - ci_2sig_fx[:, 0], ci_2sig_fx[:, 1] - posterior_means[:, 1]], 
                    fmt='none', ecolor='moccasin', alpha=0.8, capsize=8, capthick=2, elinewidth=6, label='95% CI (2σ)')
# 1σ on top
axes[0, 1].errorbar(x_points, posterior_means[:, 1], 
                    yerr=[posterior_means[:, 1] - ci_1sig_fx[:, 0], ci_1sig_fx[:, 1] - posterior_means[:, 1]], 
                    fmt='none', ecolor='orange', alpha=0.8, capsize=5, capthick=2, elinewidth=3, label='68% CI (1σ)')
axes[0, 1].scatter(x_points, theta_test[:, 1], color='red', s=150, label='True', marker='*', zorder=5)
axes[0, 1].scatter(x_points, posterior_means[:, 1], color='orange', s=80, label='Posterior mean', marker='o', zorder=5)
axes[0, 1].set_xlabel('Test Point', fontsize=11)
axes[0, 1].set_ylabel('fX', fontsize=11)
axes[0, 1].set_title('fX: True vs Posterior', fontweight='bold')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].legend(loc='best')
axes[0, 1].set_xticks(range(1, 6))

# Plot 3: Parameter space with posteriors
for i in range(len(theta_test)):
    axes[0, 2].scatter(posterior_samples[i][:, 0], posterior_samples[i][:, 1], alpha=0.1, s=5, color='steelblue')
axes[0, 2].scatter(theta_test[:, 0], theta_test[:, 1], color='red', s=150, marker='*', label='True', zorder=5)
axes[0, 2].scatter(posterior_means[:, 0], posterior_means[:, 1], color='orange', s=80, marker='o', label='Posterior mean', zorder=5)
axes[0, 2].set_xlabel('xHI', fontsize=11)
axes[0, 2].set_ylabel('fX', fontsize=11)
axes[0, 2].set_title('Parameter Space Coverage', fontweight='bold')
axes[0, 2].grid(alpha=0.3)
axes[0, 2].legend()

# Plot 4: Posterior distributions for xHI with CI shading
for i in range(min(3, len(theta_test))):
    samples_xhi = posterior_samples[i][:, 0]
    axes[1, 0].hist(samples_xhi, bins=30, alpha=0.5, label=f'Point {i+1}')
    # Add vertical lines for 1σ and 2σ for first point
    if i == 0:
        axes[1, 0].axvline(ci_1sig_xhi[0, 0], color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[1, 0].axvline(ci_1sig_xhi[0, 1], color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='68% CI')
        axes[1, 0].axvline(ci_2sig_xhi[0, 0], color='lightblue', linestyle=':', linewidth=1.5, alpha=0.7)
        axes[1, 0].axvline(ci_2sig_xhi[0, 1], color='lightblue', linestyle=':', linewidth=1.5, alpha=0.7, label='95% CI')
axes[1, 0].axvline(theta_test[0, 0], color='red', linestyle='-', linewidth=2, label='True (Point 1)')
axes[1, 0].set_xlabel('xHI', fontsize=11)
axes[1, 0].set_ylabel('Count', fontsize=11)
axes[1, 0].set_title('Posterior Distribution: xHI', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Posterior distributions for fX with CI shading
for i in range(min(3, len(theta_test))):
    samples_fx = posterior_samples[i][:, 1]
    axes[1, 1].hist(samples_fx, bins=30, alpha=0.5, label=f'Point {i+1}')
    # Add vertical lines for 1σ and 2σ for first point
    if i == 0:
        axes[1, 1].axvline(ci_1sig_fx[0, 0], color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[1, 1].axvline(ci_1sig_fx[0, 1], color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='68% CI')
        axes[1, 1].axvline(ci_2sig_fx[0, 0], color='moccasin', linestyle=':', linewidth=1.5, alpha=0.7)
        axes[1, 1].axvline(ci_2sig_fx[0, 1], color='moccasin', linestyle=':', linewidth=1.5, alpha=0.7, label='95% CI')
axes[1, 1].axvline(theta_test[0, 1], color='red', linestyle='-', linewidth=2, label='True (Point 1)')
axes[1, 1].set_xlabel('fX', fontsize=11)
axes[1, 1].set_ylabel('Count', fontsize=11)
axes[1, 1].set_title('Posterior Distribution: fX', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Errors and residuals
errors_xhi = np.abs(theta_test[:, 0] - posterior_means[:, 0])
errors_fx = np.abs(theta_test[:, 1] - posterior_means[:, 1])
axes[1, 2].bar(range(1, 6), errors_xhi, alpha=0.6, label='xHI error', width=0.4, align='edge')
axes[1, 2].bar([x+0.4 for x in range(1, 6)], errors_fx/3, alpha=0.6, label='fX error/3', width=0.4, align='edge')
axes[1, 2].set_xlabel('Test Point', fontsize=11)
axes[1, 2].set_ylabel('Absolute Error', fontsize=11)
axes[1, 2].set_title('Inference Errors', fontweight='bold')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)
axes[1, 2].set_xticks(range(1, 6))

plt.tight_layout()
inference_plot_path = os.path.join(TEST_OUTPUTS_DIR, "inference_results.png")
plt.savefig(inference_plot_path, dpi=150, bbox_inches='tight')
log_print(f"   Plot saved to: {inference_plot_path}")
plt.close()

# Save detailed results
log_print("\n6. Saving inference results...")
inference_results = {
    "theta_test": theta_test,
    "posterior_means": posterior_means,
    "posterior_stds": posterior_stds,
    "posterior_samples": posterior_samples,
}

results_path = os.path.join(TEST_OUTPUTS_DIR, "inference_results.pkl")
with open(results_path, "wb") as f:
    pickle.dump(inference_results, f)
log_print(f"   Results saved to: {results_path}")

log_print("\n" + "="*60)
log_print("✅ STEP 5 COMPLETE!")
log_print("="*60)
log_print(f"\nInference Summary:")
log_print(f"  - Sampled on 5 test points")
log_print(f"  - Drew {n_samples} samples per point")
log_print(f"  - Posterior means computed")
log_print(f"  - 95% credible intervals calculated")
log_print(f"  - Visualization created")

# Close log
log_fp.close()
print(f"\n✅ Inference complete! Log saved to: {log_file}")
