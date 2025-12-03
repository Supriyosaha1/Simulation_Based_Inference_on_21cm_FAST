"""
Step 4b: Evaluate 2D-Data Trained Posterior
Check if 2D spectral cubes training improves posterior quality significantly
"""

import numpy as np
import pickle
import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from config import ROOT_DIR

# Setup paths
DATA_DIR = os.path.join(ROOT_DIR, "UGMRT_500h", "Data", "Train_test_data")
TRAIN_OUTPUTS_DIR = os.path.join(ROOT_DIR, "UGMRT_500h", "Outputs", "Train_outputs")
VAL_OUTPUTS_DIR = os.path.join(ROOT_DIR, "UGMRT_500h", "Outputs", "Val_outputs")
os.makedirs(VAL_OUTPUTS_DIR, exist_ok=True)

# Setup logging
log_file = os.path.join(VAL_OUTPUTS_DIR, "evaluation_log.txt")
log_fp = open(log_file, "w")

def log_print(msg):
    """Print to both console and log file"""
    print(msg)
    log_fp.write(msg + "\n")
    log_fp.flush()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_print("="*60)
log_print("STEP 4b: EVALUATE 2D-DATA TRAINED POSTERIOR")
log_print("="*60)

# Load feature parameters
log_print("\n1. Loading feature parameters...")
feature_path = os.path.join(TRAIN_OUTPUTS_DIR, "feature_params_2d.pkl")
if os.path.exists(feature_path):
    with open(feature_path, "rb") as f:
        feature_params = pickle.load(f)
    x_mean = torch.from_numpy(feature_params["x_mean"]).float().to(device)
    x_std = torch.from_numpy(feature_params["x_std"]).float().to(device)
    n_los = feature_params["n_los"]
    n_freq = feature_params["n_freq"]
    log_print(f"   Loaded feature params: n_los={n_los}, n_freq={n_freq}")
else:
    log_print("   Feature params not found, computing from data...")

# Load validation data
log_print("\n2. Loading validation data...")
split_path = os.path.join(DATA_DIR, "train_val_test_split.pkl")
with open(split_path, "rb") as f:
    split_data = pickle.load(f)

theta_val = split_data["theta_val"]  # (107, 2)
x_val_2d = torch.from_numpy(split_data["x_2d_val"]).float().to(device)  # (107, 1000, 2762)

log_print(f"   Validation set size: {theta_val.shape[0]}")
log_print(f"   theta_val shape: {theta_val.shape}")
log_print(f"   x_val_2d shape: {x_val_2d.shape}")

# Extract mean and std from 2D data
log_print("\n3. Extracting mean and std from 2D cubes...")
x_val_mean = x_val_2d.mean(dim=1)  # (107, 2762)
x_val_std = x_val_2d.std(dim=1)    # (107, 2762)

log_print(f"   x_val_mean shape: {x_val_mean.shape}")
log_print(f"   x_val_std shape: {x_val_std.shape}")

# Concatenate and normalize
x_val = torch.cat([x_val_mean, x_val_std], dim=1)  # (107, 5524)
x_val_norm = (x_val - x_mean) / (x_std + 1e-8)

log_print(f"   x_val concatenated shape: {x_val.shape}")
log_print(f"   x_val normalized shape: {x_val_norm.shape}")

# Load trained posterior
log_print("\n4. Loading 2D-data trained posterior...")
posterior_path = os.path.join(TRAIN_OUTPUTS_DIR, "posterior_snpe_2d.pt")
posterior = torch.load(posterior_path, map_location=device)
log_print(f"   Loaded from: {posterior_path}")

# Sample from posterior on validation set
log_print("\n5. Sampling from posterior on validation set...")
n_samples = 500
log_print(f"   Drawing {n_samples} samples per observation...")

posterior_means = []
posterior_stds = []

for i in range(len(theta_val)):
    if (i+1) % 20 == 0:
        log_print(f"   Processed {i+1}/{len(theta_val)}")
    
    x_condition = x_val_norm[i:i+1]  # (1, 5524)
    
    try:
        samples = posterior.sample((n_samples,), x=x_condition)
    except TypeError:
        samples = posterior.sample((n_samples,), condition=x_condition)
    
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    samples = samples.reshape(-1, 2)
    
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    
    posterior_means.append(mean)
    posterior_stds.append(std)

posterior_means = np.array(posterior_means)  # (107, 2)
posterior_stds = np.array(posterior_stds)    # (107, 2)

log_print(f"\n   Posterior means shape: {posterior_means.shape}")
log_print(f"   Posterior stds shape: {posterior_stds.shape}")

# Calculate metrics
log_print("\n4. Computing evaluation metrics...")

# R2 Score for each parameter
r2_xhi = r2_score(theta_val[:, 0], posterior_means[:, 0])
r2_fx = r2_score(theta_val[:, 1], posterior_means[:, 1])

# RMSE for each parameter
rmse_xhi = np.sqrt(mean_squared_error(theta_val[:, 0], posterior_means[:, 0]))
rmse_fx = np.sqrt(mean_squared_error(theta_val[:, 1], posterior_means[:, 1]))

# Z-score (normalized residuals)
z_scores_xhi = (theta_val[:, 0] - posterior_means[:, 0]) / (posterior_stds[:, 0] + 1e-8)
z_scores_fx = (theta_val[:, 1] - posterior_means[:, 1]) / (posterior_stds[:, 1] + 1e-8)

z_mean_xhi = z_scores_xhi.mean()
z_std_xhi = z_scores_xhi.std()
z_mean_fx = z_scores_fx.mean()
z_std_fx = z_scores_fx.std()

log_print("\n" + "="*60)
log_print("EVALUATION RESULTS")
log_print("="*60)

log_print("\nxHI (Neutral Fraction) Metrics:")
log_print(f"  R² Score: {r2_xhi:.4f} (ideal: 1.0, bad: <0.5)")
log_print(f"  RMSE: {rmse_xhi:.6f} (lower is better)")
log_print(f"  Z-score mean: {z_mean_xhi:.4f} (ideal: 0.0)")
log_print(f"  Z-score std:  {z_std_xhi:.4f} (ideal: 1.0)")

log_print("\nfX (log10 X-ray heating) Metrics:")
log_print(f"  R² Score: {r2_fx:.4f} (ideal: 1.0, bad: <0.5)")
log_print(f"  RMSE: {rmse_fx:.6f} (lower is better)")
log_print(f"  Z-score mean: {z_mean_fx:.4f} (ideal: 0.0)")
log_print(f"  Z-score std:  {z_std_fx:.4f} (ideal: 1.0)")

# Overall assessment
log_print("\n" + "="*60)
log_print("TRAINING QUALITY ASSESSMENT")
log_print("="*60)

quality_score = 0
assessments = []

# Check R2
if r2_xhi > 0.8 and r2_fx > 0.8:
    assessments.append("✅ R² Scores: EXCELLENT (>0.8)")
    quality_score += 3
elif r2_xhi > 0.6 and r2_fx > 0.6:
    assessments.append("✅ R² Scores: GOOD (>0.6)")
    quality_score += 2
elif r2_xhi > 0.3 and r2_fx > 0.3:
    assessments.append("⚠️  R² Scores: FAIR (>0.3)")
    quality_score += 1
else:
    assessments.append("❌ R² Scores: POOR (<0.3)")

# Check Z-scores
if abs(z_mean_xhi) < 0.2 and abs(z_mean_fx) < 0.2:
    assessments.append("✅ Z-score means: CENTERED (close to 0)")
    quality_score += 3
elif abs(z_mean_xhi) < 0.5 and abs(z_mean_fx) < 0.5:
    assessments.append("✅ Z-score means: REASONABLE")
    quality_score += 2
else:
    assessments.append("⚠️  Z-score means: SHIFTED (far from 0)")
    quality_score += 1

if 0.8 < z_std_xhi < 1.2 and 0.8 < z_std_fx < 1.2:
    assessments.append("✅ Z-score std: CALIBRATED (close to 1)")
    quality_score += 3
elif 0.5 < z_std_xhi < 1.5 and 0.5 < z_std_fx < 1.5:
    assessments.append("✅ Z-score std: ACCEPTABLE")
    quality_score += 2
else:
    assessments.append("⚠️  Z-score std: MISCALIBRATED")
    quality_score += 1

for assessment in assessments:
    log_print(f"\n{assessment}")

log_print(f"\n{'='*60}")
if quality_score >= 8:
    log_print("🎉 OVERALL: TRAINING IS GOOD! Ready for inference")
elif quality_score >= 5:
    log_print("✅ OVERALL: TRAINING IS ACCEPTABLE. Proceed cautiously")
else:
    log_print("⚠️  OVERALL: TRAINING MAY NEED IMPROVEMENT. Consider retraining")
log_print(f"{'='*60}")

# Create visualization
log_print("\n5. Creating evaluation plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: True vs Predicted xHI
axes[0, 0].scatter(theta_val[:, 0], posterior_means[:, 0], alpha=0.6, s=50)
axes[0, 0].errorbar(theta_val[:, 0], posterior_means[:, 0], 
                    yerr=posterior_stds[:, 0], fmt='none', alpha=0.3, elinewidth=1)
axes[0, 0].plot([0, 1], [0, 1], 'r--', label='Perfect', linewidth=2)
axes[0, 0].set_xlabel('True xHI', fontsize=11)
axes[0, 0].set_ylabel('Posterior Mean xHI', fontsize=11)
axes[0, 0].set_title(f'xHI: R²={r2_xhi:.3f}', fontweight='bold')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend()

# Plot 2: True vs Predicted fX
axes[0, 1].scatter(theta_val[:, 1], posterior_means[:, 1], alpha=0.6, s=50, color='orange')
axes[0, 1].errorbar(theta_val[:, 1], posterior_means[:, 1], 
                    yerr=posterior_stds[:, 1], fmt='none', alpha=0.3, elinewidth=1)
axes[0, 1].plot([-4, 1], [-4, 1], 'r--', label='Perfect', linewidth=2)
axes[0, 1].set_xlabel('True fX', fontsize=11)
axes[0, 1].set_ylabel('Posterior Mean fX', fontsize=11)
axes[0, 1].set_title(f'fX: R²={r2_fx:.3f}', fontweight='bold')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].legend()

# Plot 3: Z-score distribution xHI
axes[1, 0].hist(z_scores_xhi, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Mean')
axes[1, 0].set_xlabel('Z-score', fontsize=11)
axes[1, 0].set_ylabel('Count', fontsize=11)
axes[1, 0].set_title(f'Z-score xHI: μ={z_mean_xhi:.3f}, σ={z_std_xhi:.3f}', fontweight='bold')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].legend()

# Plot 4: Z-score distribution fX
axes[1, 1].hist(z_scores_fx, bins=20, edgecolor='black', alpha=0.7, color='coral')
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Mean')
axes[1, 1].set_xlabel('Z-score', fontsize=11)
axes[1, 1].set_ylabel('Count', fontsize=11)
axes[1, 1].set_title(f'Z-score fX: μ={z_mean_fx:.3f}, σ={z_std_fx:.3f}', fontweight='bold')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
eval_plot_path = os.path.join(VAL_OUTPUTS_DIR, "evaluation_metrics.png")
plt.savefig(eval_plot_path, dpi=150, bbox_inches='tight')
log_print(f"   Plot saved to: {eval_plot_path}")
plt.close()

# Save metrics to file
metrics_dict = {
    "r2_xhi": r2_xhi,
    "r2_fx": r2_fx,
    "rmse_xhi": rmse_xhi,
    "rmse_fx": rmse_fx,
    "z_mean_xhi": z_mean_xhi,
    "z_std_xhi": z_std_xhi,
    "z_mean_fx": z_mean_fx,
    "z_std_fx": z_std_fx,
    "quality_score": quality_score,
}

metrics_path = os.path.join(VAL_OUTPUTS_DIR, "evaluation_metrics.pkl")
with open(metrics_path, "wb") as f:
    pickle.dump(metrics_dict, f)
log_print(f"   Metrics saved to: {metrics_path}")

log_print("\n" + "="*60)
log_print("✅ EVALUATION COMPLETE!")
log_print("="*60)

# Close log
log_fp.close()
print(f"\n✅ Log saved to: {log_file}")
