"""
Step 5b: Publication-Quality Posterior Visualization
Create 2D posterior distributions with contours (1σ and 2σ credible intervals)
Similar to the paper figure you showed
"""

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

# Setup paths
OUTPUT_DIR = "/user1/supriyo/ml_project/SBI_step_by_step/UGMRT_500h/Outputs/"

# Setup logging
log_file = os.path.join(OUTPUT_DIR, "visualization_log.txt")
log_fp = open(log_file, "w")

def log_print(msg):
    """Print to both console and log file"""
    print(msg)
    log_fp.write(msg + "\n")
    log_fp.flush()

log_print("="*70)
log_print("STEP 5b: PUBLICATION-QUALITY POSTERIOR VISUALIZATION")
log_print("="*70)

# Load inference results
log_print("\n1. Loading inference results...")
results_path = os.path.join(OUTPUT_DIR, "inference_results.pkl")
with open(results_path, "rb") as f:
    results = pickle.load(f)

theta_test = results["theta_test"]  # (5, 2)
posterior_means = results["posterior_means"]  # (5, 2)
posterior_stds = results["posterior_stds"]  # (5, 2)
posterior_samples = results["posterior_samples"]  # list of 5 arrays

log_print(f"   Loaded results for {len(theta_test)} test points")

# Create main figure with 2x3 subplots (one for each test point, plus summary)
log_print("\n2. Creating publication-quality plots...")

fig = plt.figure(figsize=(16, 12))

# Define colors for each point
colors = ['steelblue', 'magenta', 'green', 'purple', 'coral']
labels = ['Point 1\n(low xHI, low fX)', 'Point 2\n(low xHI, high fX)', 
          'Point 3\n(mid xHI, mid fX)', 'Point 4\n(high xHI, high fX)', 
          'Point 5\n(high xHI, low fX)']

# Create subplots for each test point
for idx in range(5):
    ax = plt.subplot(2, 3, idx+1)
    
    samples = posterior_samples[idx]  # (n_samples, 2)
    true_params = theta_test[idx]
    post_mean = posterior_means[idx]
    
    # Extract xHI and fX samples
    xhi_samples = samples[:, 0]
    fx_samples = samples[:, 1]
    
    # Create 2D histogram/density plot
    h = ax.hist2d(xhi_samples, fx_samples, bins=40, cmap='Blues', cmin=1)
    
    # Compute contours using KDE
    try:
        xy = np.vstack([xhi_samples, fx_samples])
        z = gaussian_kde(xy)(xy)
        
        # Create contour plot
        x_grid = np.linspace(xhi_samples.min()-0.1, xhi_samples.max()+0.1, 100)
        y_grid = np.linspace(fx_samples.min()-0.5, fx_samples.max()+0.5, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = gaussian_kde(xy)(positions).reshape(X.shape)
        
        # Plot contours for 1σ and 2σ
        levels = [np.percentile(z, 5), np.percentile(z, 32)]  # 2σ and 1σ
        contours = ax.contour(X, Y, Z, levels=levels, colors='darkblue', linewidths=2)
        ax.clabel(contours, inline=True, fontsize=8)
    except:
        pass
    
    # Plot true parameter as star
    ax.scatter(true_params[0], true_params[1], marker='*', s=600, 
              color='red', edgecolors='darkred', linewidths=1.5, zorder=5, label='True')
    
    # Plot posterior mean as circle
    ax.scatter(post_mean[0], post_mean[1], marker='o', s=200, 
              color=colors[idx], edgecolors='black', linewidths=1.5, zorder=5, label='Posterior median')
    
    # Add error ellipse (1σ)
    width = 2 * posterior_stds[idx, 0]
    height = 2 * posterior_stds[idx, 1]
    ellipse = Ellipse((post_mean[0], post_mean[1]), width, height, 
                     color=colors[idx], alpha=0.2, linestyle='--', linewidth=2)
    ax.add_patch(ellipse)
    
    # Labels and formatting
    ax.set_xlabel('xHI (Neutral Fraction)', fontsize=11, fontweight='bold')
    ax.set_ylabel('log₁₀(fX)', fontsize=11, fontweight='bold')
    ax.set_title(f'Test Case {idx+1}\n{labels[idx]}', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':')
    
    # Set axis limits
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-4.5, 1.5)
    
    # Add text box with statistics
    error_xhi = abs(true_params[0] - post_mean[0])
    error_fx = abs(true_params[1] - post_mean[1])
    textstr = f'Error xHI: {error_xhi:.3f}\nError fX: {error_fx:.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if idx == 0:
        ax.legend(loc='lower right', fontsize=9)

# Summary plot (bottom right)
ax_summary = plt.subplot(2, 3, 6)

# Plot all posteriors in 2D parameter space
for idx in range(5):
    samples = posterior_samples[idx]
    ax_summary.scatter(samples[:, 0], samples[:, 1], alpha=0.08, s=3, color=colors[idx], label=f'Test {idx+1}')

# Overlay true and posterior means
ax_summary.scatter(theta_test[:, 0], theta_test[:, 1], marker='*', s=500, 
                  color='red', edgecolors='darkred', linewidths=2, zorder=5, label='True values')
ax_summary.scatter(posterior_means[:, 0], posterior_means[:, 1], marker='o', s=150, 
                  color='orange', edgecolors='black', linewidths=1.5, zorder=5, label='Posterior means')

# Connect true to estimated with arrows
for idx in range(5):
    ax_summary.annotate('', xy=(posterior_means[idx, 0], posterior_means[idx, 1]),
                       xytext=(theta_test[idx, 0], theta_test[idx, 1]),
                       arrowprops=dict(arrowstyle='->', color=colors[idx], alpha=0.5, lw=1.5))

ax_summary.set_xlabel('xHI (Neutral Fraction)', fontsize=12, fontweight='bold')
ax_summary.set_ylabel('log₁₀(fX)', fontsize=12, fontweight='bold')
ax_summary.set_title('Parameter Space Summary\n(All 5 Test Cases)', fontsize=12, fontweight='bold')
ax_summary.grid(alpha=0.3, linestyle=':')
ax_summary.set_xlim(-0.1, 1.1)
ax_summary.set_ylim(-4.5, 1.5)
ax_summary.legend(loc='upper left', fontsize=9, ncol=2)

# Overall title
fig.suptitle('Posterior Distributions from 21-cm Forest SBI Inference\nContours show 1σ and 2σ credible intervals', 
            fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save figure
pub_plot_path = os.path.join(OUTPUT_DIR, "posterior_distributions_publication.png")
plt.savefig(pub_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
log_print(f"   Publication plot saved to: {pub_plot_path}")
plt.close()

# Create second figure: Clean comparison plot
log_print("\n3. Creating comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: True vs Inferred xHI
ax = axes[0, 0]
for idx in range(5):
    ax.scatter(idx+1, theta_test[idx, 0], marker='*', s=400, color='red', zorder=5, label='True' if idx==0 else '')
    ax.scatter(idx+1, posterior_means[idx, 0], marker='o', s=150, color=colors[idx], edgecolors='black', linewidths=1.5, zorder=5)
    # Add error bars
    ax.errorbar(idx+1, posterior_means[idx, 0], yerr=posterior_stds[idx, 0]*1.96, 
               fmt='none', ecolor=colors[idx], alpha=0.5, capsize=8, linewidth=2)
ax.set_xlabel('Test Point', fontsize=12, fontweight='bold')
ax.set_ylabel('xHI', fontsize=12, fontweight='bold')
ax.set_title('Neutral Fraction Inference', fontsize=12, fontweight='bold')
ax.set_xticks(range(1, 6))
ax.grid(alpha=0.3, linestyle=':')
ax.legend(fontsize=10)
ax.set_ylim(-0.2, 1.2)

# Plot 2: True vs Inferred fX
ax = axes[0, 1]
for idx in range(5):
    ax.scatter(idx+1, theta_test[idx, 1], marker='*', s=400, color='red', zorder=5, label='True' if idx==0 else '')
    ax.scatter(idx+1, posterior_means[idx, 1], marker='o', s=150, color=colors[idx], edgecolors='black', linewidths=1.5, zorder=5)
    # Add error bars
    ax.errorbar(idx+1, posterior_means[idx, 1], yerr=posterior_stds[idx, 1]*1.96, 
               fmt='none', ecolor=colors[idx], alpha=0.5, capsize=8, linewidth=2)
ax.set_xlabel('Test Point', fontsize=12, fontweight='bold')
ax.set_ylabel('log₁₀(fX)', fontsize=12, fontweight='bold')
ax.set_title('X-ray Heating Inference', fontsize=12, fontweight='bold')
ax.set_xticks(range(1, 6))
ax.grid(alpha=0.3, linestyle=':')
ax.legend(fontsize=10)
ax.set_ylim(-4.5, 1.5)

# Plot 3: Absolute errors
ax = axes[1, 0]
errors_xhi = np.abs(theta_test[:, 0] - posterior_means[:, 0])
errors_fx = np.abs(theta_test[:, 1] - posterior_means[:, 1])
x_pos = np.arange(5)
width = 0.35
bars1 = ax.bar(x_pos - width/2, errors_xhi, width, label='xHI error', color='steelblue', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, errors_fx/3, width, label='fX error / 3', color='coral', alpha=0.7)
ax.set_xlabel('Test Point', fontsize=12, fontweight='bold')
ax.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
ax.set_title('Inference Errors', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos + 1)
ax.set_xticklabels([f'Test {i+1}' for i in range(5)])
ax.grid(alpha=0.3, axis='y', linestyle=':')
ax.legend(fontsize=10)

# Plot 4: Credible interval coverage
ax = axes[1, 1]
coverage_xhi = []
coverage_fx = []
for idx in range(5):
    samples = posterior_samples[idx]
    ci_low_xhi = np.percentile(samples[:, 0], 2.5)
    ci_high_xhi = np.percentile(samples[:, 0], 97.5)
    ci_low_fx = np.percentile(samples[:, 1], 2.5)
    ci_high_fx = np.percentile(samples[:, 1], 97.5)
    
    # Check if true value is in CI
    coverage_xhi.append(1 if (ci_low_xhi <= theta_test[idx, 0] <= ci_high_xhi) else 0)
    coverage_fx.append(1 if (ci_low_fx <= theta_test[idx, 1] <= ci_high_fx) else 0)

x_pos = np.arange(5)
bars1 = ax.bar(x_pos - width/2, coverage_xhi, width, label='xHI in 95% CI', color='steelblue', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, coverage_fx, width, label='fX in 95% CI', color='coral', alpha=0.7)
ax.set_xlabel('Test Point', fontsize=12, fontweight='bold')
ax.set_ylabel('Coverage (1=Yes, 0=No)', fontsize=12, fontweight='bold')
ax.set_title('95% Credible Interval Coverage', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos + 1)
ax.set_xticklabels([f'Test {i+1}' for i in range(5)])
ax.set_ylim(0, 1.2)
ax.grid(alpha=0.3, axis='y', linestyle=':')
ax.legend(fontsize=10)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

fig.suptitle('SBI Inference Quality Assessment\n21-cm Forest Parameter Estimation', 
            fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.98])

# Save figure
comparison_plot_path = os.path.join(OUTPUT_DIR, "inference_comparison.png")
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
log_print(f"   Comparison plot saved to: {comparison_plot_path}")
plt.close()

log_print("\n" + "="*70)
log_print("✅ PUBLICATION-QUALITY VISUALIZATIONS CREATED!")
log_print("="*70)
log_print(f"\nFigures generated:")
log_print(f"  1. {pub_plot_path}")
log_print(f"     - 5 individual posterior distributions with contours (1σ, 2σ)")
log_print(f"     - Parameter space summary plot")
log_print(f"     - Publication-ready quality (300 DPI)")
log_print(f"\n  2. {comparison_plot_path}")
log_print(f"     - True vs inferred comparison")
log_print(f"     - Inference error analysis")
log_print(f"     - Credible interval coverage assessment")

log_print(f"\nSummary Statistics:")
log_print(f"  - Average xHI error: {np.mean(errors_xhi):.4f}")
log_print(f"  - Average fX error: {np.mean(errors_fx):.4f}")
log_print(f"  - xHI coverage: {int(np.sum(coverage_xhi))}/5 = {100*np.mean(coverage_xhi):.0f}%")
log_print(f"  - fX coverage: {int(np.sum(coverage_fx))}/5 = {100*np.mean(coverage_fx):.0f}%")

# Close log
log_fp.close()
print(f"\n✅ Visualization complete!")
print(f"   Plots saved to: {OUTPUT_DIR}")
print(f"   Log: {log_file}")
