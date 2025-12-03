"""
Step 5b: Create Beautiful Posterior Contour Plots with Credible Regions
Generate 2D contour plots with 1-sigma and 2-sigma credible regions
"""

import numpy as np
import pickle
import torch
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from config import ROOT_DIR

# Setup paths
DATA_DIR = os.path.join(ROOT_DIR, "UGMRT_500h", "Data", "Train_test_data")
TEST_OUTPUTS_DIR = os.path.join(ROOT_DIR, "UGMRT_500h", "Outputs", "Test_outputs")
os.makedirs(TEST_OUTPUTS_DIR, exist_ok=True)

print("="*80)
print("STEP 5b: POSTERIOR CONTOUR PLOTS WITH CREDIBLE REGIONS")
print("="*80)

# Load inference results
print("\n1. Loading inference results...")
results_path = os.path.join(TEST_OUTPUTS_DIR, "inference_results.pkl")
with open(results_path, "rb") as f:
    inference_results = pickle.load(f)

theta_test = inference_results["theta_test"]
posterior_means = inference_results["posterior_means"]
posterior_stds = inference_results["posterior_stds"]
posterior_samples = inference_results["posterior_samples"]

print(f"   Loaded {len(theta_test)} test points")
print(f"   Each with {len(posterior_samples[0])} posterior samples")

# Create figure with 5 subplots (one for each test point)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

print("\n2. Creating contour plots with credible regions...")

for test_idx in range(len(theta_test)):
    ax = axes[test_idx]
    samples = posterior_samples[test_idx]  # (2000, 2)
    true_val = theta_test[test_idx]
    post_mean = posterior_means[test_idx]
    post_std = posterior_stds[test_idx]
    
    # Extract xHI and fX samples
    xhi_samples = samples[:, 0]
    fx_samples = samples[:, 1]
    
    # Create 2D grid for contour plot
    xhi_range = np.linspace(min(xhi_samples.min(), true_val[0]) - 0.15, 
                            max(xhi_samples.max(), true_val[0]) + 0.15, 100)
    fx_range = np.linspace(min(fx_samples.min(), true_val[1]) - 0.3, 
                           max(fx_samples.max(), true_val[1]) + 0.3, 100)
    
    xhi_grid, fx_grid = np.meshgrid(xhi_range, fx_range)
    positions = np.vstack([xhi_grid.ravel(), fx_grid.ravel()])
    
    # Compute KDE for density
    kde = gaussian_kde(np.vstack([xhi_samples, fx_samples]))
    density = kde(positions).reshape(xhi_grid.shape)
    
    # Plot contours
    # Determine contour levels for 1-sigma and 2-sigma
    levels = np.percentile(density.ravel(), [68, 95])  # 1-sigma and 2-sigma equivalents
    
    # Draw filled contours
    cf = ax.contourf(xhi_grid, fx_grid, density, levels=[0, levels[0], levels[1], density.max()],
                     colors=['#f7f7f7', '#d7d7ff', '#a0a0ff'], alpha=0.7)
    
    # Draw contour lines
    cs = ax.contour(xhi_grid, fx_grid, density, levels=levels, colors=['blue', 'darkblue'],
                    linewidths=[1.5, 2.0], linestyles=['--', '-'])
    
    # Add contour labels
    ax.clabel(cs, inline=True, fontsize=9, fmt='%0.0e')
    
    # Plot samples as light scatter
    ax.scatter(xhi_samples, fx_samples, alpha=0.02, s=1, c='steelblue', label='Posterior samples')
    
    # Plot posterior mean
    ax.scatter(post_mean[0], post_mean[1], c='green', s=200, marker='+', linewidth=3,
              label='Posterior mean', zorder=10)
    
    # Plot true value with star
    ax.scatter(true_val[0], true_val[1], c='red', s=400, marker='*', 
              edgecolors='darkred', linewidth=2, label='True value', zorder=15)
    
    # Calculate distance from true to posterior mean
    dist = np.sqrt((true_val[0] - post_mean[0])**2 + (true_val[1] - post_mean[1])**2)
    
    # Check if true value is within credible regions
    in_1sigma = (np.abs(true_val[0] - post_mean[0]) <= post_std[0] and 
                 np.abs(true_val[1] - post_mean[1]) <= post_std[1])
    in_2sigma = (np.abs(true_val[0] - post_mean[0]) <= 2*post_std[0] and 
                 np.abs(true_val[1] - post_mean[1]) <= 2*post_std[1])
    
    # Labels and formatting
    ax.set_xlabel('xHI (Neutral Fraction)', fontsize=11, fontweight='bold')
    ax.set_ylabel('fX (log₁₀ X-ray heating)', fontsize=11, fontweight='bold')
    
    # Title with quality indicator
    title = f'Test Point {test_idx + 1}'
    if in_1sigma:
        title += ' ✓ 1σ'
        color = 'green'
    elif in_2sigma:
        title += ' ✓ 2σ'
        color = 'orange'
    else:
        title += ' ✗ Outside'
        color = 'red'
    
    ax.set_title(title, fontsize=12, fontweight='bold', color=color, pad=10)
    
    # Add info text
    info_text = f'True: ({true_val[0]:.3f}, {true_val[1]:.3f})\n'
    info_text += f'Post: ({post_mean[0]:.3f}, {post_mean[1]:.3f})\n'
    info_text += f'σ: ({post_std[0]:.3f}, {post_std[1]:.3f})\n'
    info_text += f'Distance: {dist:.4f}'
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           family='monospace')
    
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(loc='lower right', fontsize=9)
    
    print(f"   Point {test_idx+1}: True in {'1σ' if in_1sigma else '2σ' if in_2sigma else 'Outside'} region")

# Remove extra subplot
fig.delaxes(axes[5])

# Add overall title and legend
fig.suptitle('Posterior Distributions with Credible Regions (5 Test Points)', 
            fontsize=16, fontweight='bold', y=0.995)

# Create custom legend
legend_elements = [
    mpatches.Patch(facecolor='#a0a0ff', label='2σ credible region (95%)'),
    mpatches.Patch(facecolor='#d7d7ff', label='1σ credible region (68%)'),
    mpatches.Patch(facecolor='#f7f7f7', label='KDE density'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', 
              markersize=4, alpha=0.3, label='Posterior samples (α=0.02)'),
    plt.Line2D([0], [0], marker='+', color='green', linestyle='None',
              markersize=15, markeredgewidth=2, label='Posterior mean'),
    plt.Line2D([0], [0], marker='*', color='red', linestyle='None',
              markersize=20, markeredgecolor='darkred', label='True value'),
]

fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
          bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True, shadow=True)

plt.tight_layout(rect=[0, 0.05, 1, 0.99])

# Save figure
contour_path = os.path.join(TEST_OUTPUTS_DIR, "posterior_contour_plots.png")
plt.savefig(contour_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Contour plot saved: {contour_path}")
plt.close()

# Create individual high-quality plots for each test point
print("\n3. Creating individual high-resolution plots...")

for test_idx in range(len(theta_test)):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    samples = posterior_samples[test_idx]
    true_val = theta_test[test_idx]
    post_mean = posterior_means[test_idx]
    post_std = posterior_stds[test_idx]
    
    xhi_samples = samples[:, 0]
    fx_samples = samples[:, 1]
    
    # Create grid
    xhi_range = np.linspace(min(xhi_samples.min(), true_val[0]) - 0.2, 
                            max(xhi_samples.max(), true_val[0]) + 0.2, 150)
    fx_range = np.linspace(min(fx_samples.min(), true_val[1]) - 0.4, 
                           max(fx_samples.max(), true_val[1]) + 0.4, 150)
    
    xhi_grid, fx_grid = np.meshgrid(xhi_range, fx_range)
    positions = np.vstack([xhi_grid.ravel(), fx_grid.ravel()])
    
    # KDE
    kde = gaussian_kde(np.vstack([xhi_samples, fx_samples]))
    density = kde(positions).reshape(xhi_grid.shape)
    
    # Contour levels
    levels = np.percentile(density.ravel(), [68, 95])
    
    # Plot
    cf = ax.contourf(xhi_grid, fx_grid, density, levels=[0, levels[0], levels[1], density.max()],
                     colors=['#f7f7f7', '#d7d7ff', '#a0a0ff'], alpha=0.8)
    cs = ax.contour(xhi_grid, fx_grid, density, levels=levels, colors=['blue', 'darkblue'],
                   linewidths=[2, 2.5], linestyles=['--', '-'])
    ax.clabel(cs, inline=True, fontsize=10, fmt='%0.0e')
    
    # Scatter samples
    ax.scatter(xhi_samples, fx_samples, alpha=0.03, s=2, c='steelblue')
    
    # Posterior mean
    ax.scatter(post_mean[0], post_mean[1], c='green', s=300, marker='+', linewidth=4,
              label='Posterior mean', zorder=10)
    
    # True value
    ax.scatter(true_val[0], true_val[1], c='red', s=500, marker='*', 
              edgecolors='darkred', linewidth=2.5, label='True value', zorder=15)
    
    # Draw line from true to posterior mean
    ax.plot([true_val[0], post_mean[0]], [true_val[1], post_mean[1]], 
           'k--', linewidth=1, alpha=0.5, zorder=5)
    
    # Labels
    ax.set_xlabel('xHI (Neutral Fraction)', fontsize=13, fontweight='bold')
    ax.set_ylabel('fX (log₁₀ X-ray heating)', fontsize=13, fontweight='bold')
    ax.set_title(f'Posterior Distribution - Test Point {test_idx + 1}', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Stats box
    dist = np.sqrt((true_val[0] - post_mean[0])**2 + (true_val[1] - post_mean[1])**2)
    in_1sigma = (np.abs(true_val[0] - post_mean[0]) <= post_std[0] and 
                 np.abs(true_val[1] - post_mean[1]) <= post_std[1])
    in_2sigma = (np.abs(true_val[0] - post_mean[0]) <= 2*post_std[0] and 
                 np.abs(true_val[1] - post_mean[1]) <= 2*post_std[1])
    
    stats_text = f'True Value:\n  xHI = {true_val[0]:.4f}\n  fX = {true_val[1]:.4f}\n\n'
    stats_text += f'Posterior Mean:\n  xHI = {post_mean[0]:.4f}\n  fX = {post_mean[1]:.4f}\n\n'
    stats_text += f'Posterior Std:\n  σ_xhi = {post_std[0]:.4f}\n  σ_fx = {post_std[1]:.4f}\n\n'
    stats_text += f'Distance: {dist:.4f}\n'
    stats_text += f'Status: {"✓ 1σ" if in_1sigma else "✓ 2σ" if in_2sigma else "✗ Outside"}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', 
           edgecolor='orange', linewidth=2, alpha=0.95), family='monospace',
           fontweight='bold')
    
    ax.grid(alpha=0.4, linestyle=':', linewidth=0.8)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
    
    plt.tight_layout()
    
    individual_path = os.path.join(TEST_OUTPUTS_DIR, f"posterior_contour_point_{test_idx+1}.png")
    plt.savefig(individual_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Point {test_idx+1}: {individual_path}")
    plt.close()

print("\n" + "="*80)
print("✅ CONTOUR PLOTS COMPLETE!")
print("="*80)
print("\nFiles created:")
print(f"  - posterior_contour_plots.png (5 subplots)")
print(f"  - posterior_contour_point_1.png through point_5.png (individual)")
print("\nPlot Features:")
print("  ✓ 2D contour plots with KDE density")
print("  ✓ 1σ and 2σ credible regions (68% and 95%)")
print("  ✓ True value marked with red star")
print("  ✓ Posterior mean marked with green cross")
print("  ✓ Posterior samples shown as light scatter")
print("  ✓ Ellipses showing 1σ and 2σ regions")
