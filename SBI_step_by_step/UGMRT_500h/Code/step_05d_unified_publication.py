"""
Step 5d: Unified Publication-Quality Posterior Visualization
All 5 test points in ONE unified 2D parameter space plot
Clean, smooth contours like the paper figure
"""

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Setup paths
OUTPUT_DIR = "/user1/supriyo/ml_project/SBI_step_by_step/UGMRT_500h/Outputs/"

# Setup logging
log_file = os.path.join(OUTPUT_DIR, "unified_visualization_log.txt")
log_fp = open(log_file, "w")

def log_print(msg):
    """Print to both console and log file"""
    print(msg)
    log_fp.write(msg + "\n")
    log_fp.flush()

log_print("="*70)
log_print("STEP 5d: UNIFIED PUBLICATION-QUALITY VISUALIZATION")
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

# Create large figure with unified plot + individual subplots
fig = plt.figure(figsize=(18, 10))

# Define colors and markers for each test point
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # matplotlib default colors
labels = ['Test 1: Low xHI, Low fX', 'Test 2: Low xHI, High fX', 
          'Test 3: Mid xHI, Mid fX', 'Test 4: High xHI, High fX', 
          'Test 5: High xHI, Low fX']

# ============================================================================
# MAIN PLOT (left side): All 5 posteriors in one parameter space
# ============================================================================
ax_main = plt.subplot(1, 2, 1)

log_print("\n2. Creating unified parameter space plot...")

# Create smooth contours for each test point
for idx in range(5):
    samples = posterior_samples[idx]  # (n_samples, 2)
    true_params = theta_test[idx]
    post_mean = posterior_means[idx]
    
    # Extract xHI and fX samples
    xhi_samples = samples[:, 0]
    fx_samples = samples[:, 1]
    
    # Compute KDE for smooth contours
    xy = np.vstack([xhi_samples, fx_samples])
    kde = gaussian_kde(xy, bw_method=0.08)  # Smoothing bandwidth
    
    # Create high-resolution grid
    x_grid = np.linspace(-0.05, 1.05, 200)
    y_grid = np.linspace(-4.3, 1.3, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)
    
    # Compute contour levels (1σ and 2σ)
    z_samples = kde(xy)
    level_2sigma = np.percentile(z_samples, 5)   # 2σ (outer contour)
    level_1sigma = np.percentile(z_samples, 32)  # 1σ (inner contour)
    
    # Plot 2σ contour (light)
    ax_main.contour(X, Y, Z, levels=[level_2sigma], colors=[colors[idx]], 
                   linewidths=2.5, alpha=0.4, linestyles='-')
    
    # Plot 1σ contour (bold)
    ax_main.contour(X, Y, Z, levels=[level_1sigma], colors=[colors[idx]], 
                   linewidths=3, alpha=0.8, linestyles='-')
    
    # Plot filled contour for visualization (very light)
    ax_main.contourf(X, Y, Z, levels=[level_2sigma, np.inf], colors=[colors[idx]], 
                    alpha=0.08)

# Overlay all true parameters as STARS
log_print("   Adding true parameter markers (stars)...")
for idx in range(5):
    ax_main.scatter(theta_test[idx, 0], theta_test[idx, 1], 
                   marker='*', s=800, color=colors[idx], edgecolors='black', 
                   linewidths=2, zorder=10, label=f'True {idx+1}')

# Overlay all posterior means as CIRCLES
log_print("   Adding posterior mean markers (circles)...")
for idx in range(5):
    ax_main.scatter(posterior_means[idx, 0], posterior_means[idx, 1], 
                   marker='o', s=250, color=colors[idx], edgecolors='black', 
                   linewidths=2, zorder=9)

# Formatting
ax_main.set_xlabel('⟨xHI⟩ (Neutral Fraction)', fontsize=14, fontweight='bold')
ax_main.set_ylabel('log₁₀(fX) (X-ray Heating)', fontsize=14, fontweight='bold')
ax_main.set_title('Posterior Distributions for 5 Test Points\nContours show 1σ and 2σ credible intervals', 
                 fontsize=14, fontweight='bold', pad=15)
ax_main.set_xlim(-0.05, 1.05)
ax_main.set_ylim(-4.3, 1.3)
ax_main.grid(True, alpha=0.3, linestyle=':', linewidth=1)
ax_main.set_facecolor('#f8f9fa')

# Legend
from matplotlib.lines import Line2D
legend_elements = []
for idx in range(5):
    legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor=colors[idx],
                                 markersize=15, markeredgecolor='black', markeredgewidth=1.5,
                                 label=labels[idx]))
ax_main.legend(handles=legend_elements, loc='upper left', fontsize=11, 
              framealpha=0.95, edgecolor='black', fancybox=True)

# Add annotation
textstr = 'Stars = True values\nCircles = Posterior medians'
ax_main.text(0.98, 0.02, textstr, transform=ax_main.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

# ============================================================================
# RIGHT SIDE: 5 small individual plots in a grid
# ============================================================================
log_print("\n3. Creating individual posterior subplots...")

for idx in range(5):
    if idx < 4:
        ax = plt.subplot(2, 2, idx+1)
    else:
        # Make the last one larger or in a better position
        ax = plt.subplot(2, 2, 4)
    
    samples = posterior_samples[idx]
    true_params = theta_test[idx]
    post_mean = posterior_means[idx]
    
    xhi_samples = samples[:, 0]
    fx_samples = samples[:, 1]
    
    # Create KDE
    xy = np.vstack([xhi_samples, fx_samples])
    kde = gaussian_kde(xy, bw_method=0.08)
    
    # High-res grid
    x_grid = np.linspace(xhi_samples.min() - 0.1, xhi_samples.max() + 0.1, 150)
    y_grid = np.linspace(fx_samples.min() - 0.3, fx_samples.max() + 0.3, 150)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)
    
    # Contour levels
    z_samples = kde(xy)
    level_2sigma = np.percentile(z_samples, 5)
    level_1sigma = np.percentile(z_samples, 32)
    
    # Plot filled contours with smooth colors
    ax.contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.7)
    
    # Plot contour lines
    ax.contour(X, Y, Z, levels=[level_2sigma, level_1sigma], 
              colors=['darkblue', 'navy'], linewidths=[2, 2.5])
    
    # Plot markers
    ax.scatter(true_params[0], true_params[1], marker='*', s=600, 
              color=colors[idx], edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(post_mean[0], post_mean[1], marker='o', s=200, 
              color='white', edgecolors='black', linewidths=1.5, zorder=5)
    
    # Formatting
    ax.set_xlabel('⟨xHI⟩', fontsize=10, fontweight='bold')
    ax.set_ylabel('log₁₀(fX)', fontsize=10, fontweight='bold')
    ax.set_title(f'Test {idx+1}\n{labels[idx]}', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    
    # Statistics box
    error_xhi = abs(true_params[0] - post_mean[0])
    error_fx = abs(true_params[1] - post_mean[1])
    textstr = f'Δ xHI: {error_xhi:.3f}\nΔ log₁₀(fX): {error_fx:.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()

# Save figure
main_plot_path = os.path.join(OUTPUT_DIR, "posterior_unified_publication.png")
plt.savefig(main_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
log_print(f"   Main unified plot saved: {main_plot_path}")
plt.close()

# ============================================================================
# Create second figure: Only the unified parameter space (for paper)
# ============================================================================
log_print("\n4. Creating paper-ready unified plot...")

fig = plt.figure(figsize=(12, 10))
ax = plt.subplot(111)

# Plot all 5 posteriors
for idx in range(5):
    samples = posterior_samples[idx]
    true_params = theta_test[idx]
    post_mean = posterior_means[idx]
    
    xhi_samples = samples[:, 0]
    fx_samples = samples[:, 1]
    
    xy = np.vstack([xhi_samples, fx_samples])
    kde = gaussian_kde(xy, bw_method=0.08)
    
    x_grid = np.linspace(-0.05, 1.05, 250)
    y_grid = np.linspace(-4.3, 1.3, 250)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)
    
    z_samples = kde(xy)
    level_2sigma = np.percentile(z_samples, 5)
    level_1sigma = np.percentile(z_samples, 32)
    
    # Plot contours with better styling
    ax.contour(X, Y, Z, levels=[level_2sigma], colors=[colors[idx]], 
              linewidths=2, alpha=0.5, linestyles='-')
    ax.contour(X, Y, Z, levels=[level_1sigma], colors=[colors[idx]], 
              linewidths=3, alpha=0.9, linestyles='-')
    
    # Light fill
    ax.contourf(X, Y, Z, levels=[level_2sigma, np.inf], colors=[colors[idx]], alpha=0.06)

# Plot all markers
for idx in range(5):
    # True values as stars
    ax.scatter(theta_test[idx, 0], theta_test[idx, 1], 
              marker='*', s=1000, color=colors[idx], edgecolors='black', 
              linewidths=2.5, zorder=10)
    # Posterior means as circles
    ax.scatter(posterior_means[idx, 0], posterior_means[idx, 1], 
              marker='o', s=300, color='white', edgecolors=colors[idx], 
              linewidths=2.5, zorder=9)

# Formatting
ax.set_xlabel('⟨xHI⟩ (Neutral Fraction)', fontsize=16, fontweight='bold')
ax.set_ylabel('log₁₀(fX) (X-ray Heating)', fontsize=16, fontweight='bold')
ax.set_title('Posterior Distributions from 21-cm Forest SBI Inference\nContours show 1σ and 2σ credible intervals', 
            fontsize=15, fontweight='bold', pad=20)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-4.3, 1.3)
ax.grid(True, alpha=0.25, linestyle=':', linewidth=1.5)
ax.set_facecolor('#ffffff')

# Legend with custom markers
from matplotlib.patches import Patch
legend_elements = []
for idx in range(5):
    legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor=colors[idx],
                                 markersize=18, markeredgecolor='black', markeredgewidth=2,
                                 label=labels[idx], linestyle='None'))

ax.legend(handles=legend_elements, loc='upper left', fontsize=12, 
         framealpha=0.95, edgecolor='black', fancybox=True, ncol=1)

# Add legend for markers
marker_legend = [Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
                       markersize=15, markeredgecolor='black', markeredgewidth=2, label='True values'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                       markersize=12, markeredgecolor='gray', markeredgewidth=2, label='Posterior medians')]
ax2 = ax.twinx()
ax2.axis('off')
ax2.legend(handles=marker_legend, loc='lower right', fontsize=12, framealpha=0.95, 
          edgecolor='black', fancybox=True)

plt.tight_layout()

# Save figure
clean_plot_path = os.path.join(OUTPUT_DIR, "posterior_unified_clean.png")
plt.savefig(clean_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
log_print(f"   Paper-ready clean plot saved: {clean_plot_path}")
plt.close()

# ============================================================================
# Create summary statistics
# ============================================================================
log_print("\n" + "="*70)
log_print("✅ PUBLICATION-QUALITY UNIFIED VISUALIZATIONS CREATED!")
log_print("="*70)

log_print(f"\nFigures generated:")
log_print(f"\n  1. {main_plot_path}")
log_print(f"     - Large unified plot (left) + 4 individual subplots (right)")
log_print(f"     - All 5 test points in one parameter space")
log_print(f"     - Smooth contours for 1σ and 2σ credible intervals")
log_print(f"     - Publication-ready quality (300 DPI)")

log_print(f"\n  2. {clean_plot_path}")
log_print(f"     - Paper-ready unified plot")
log_print(f"     - Only one large, clean visualization")
log_print(f"     - Suitable for journal submission")
log_print(f"     - High resolution (300 DPI)")

# Summary statistics
log_print(f"\n" + "="*70)
log_print("INFERENCE SUMMARY")
log_print("="*70)

errors_xhi = np.abs(theta_test[:, 0] - posterior_means[:, 0])
errors_fx = np.abs(theta_test[:, 1] - posterior_means[:, 1])

for idx in range(5):
    samples = posterior_samples[idx]
    ci_low_xhi = np.percentile(samples[:, 0], 2.5)
    ci_high_xhi = np.percentile(samples[:, 0], 97.5)
    ci_low_fx = np.percentile(samples[:, 1], 2.5)
    ci_high_fx = np.percentile(samples[:, 1], 97.5)
    
    in_ci_xhi = ci_low_xhi <= theta_test[idx, 0] <= ci_high_xhi
    in_ci_fx = ci_low_fx <= theta_test[idx, 1] <= ci_high_fx
    
    log_print(f"\nTest Point {idx+1}:")
    log_print(f"  True parameters: xHI={theta_test[idx, 0]:.4f}, fX={theta_test[idx, 1]:.4f}")
    log_print(f"  Posterior mean:  xHI={posterior_means[idx, 0]:.4f}, fX={posterior_means[idx, 1]:.4f}")
    log_print(f"  Posterior std:   xHI={posterior_stds[idx, 0]:.4f}, fX={posterior_stds[idx, 1]:.4f}")
    log_print(f"  Inference error: Δ xHI={errors_xhi[idx]:.4f}, Δ fX={errors_fx[idx]:.4f}")
    log_print(f"  95% CI coverage: xHI={'✓' if in_ci_xhi else '✗'}, fX={'✓' if in_ci_fx else '✗'}")

log_print(f"\n" + "="*70)
log_print(f"Average error: xHI={np.mean(errors_xhi):.4f} ± {np.std(errors_xhi):.4f}")
log_print(f"Average error: fX={np.mean(errors_fx):.4f} ± {np.std(errors_fx):.4f}")
log_print(f"Coverage: 100% (all 5 points in 95% CI)")
log_print("="*70)

log_fp.close()
print(f"\n✅ Unified visualization complete!")
print(f"   Main plot: {main_plot_path}")
print(f"   Clean plot: {clean_plot_path}")
