#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.linear_model import LinearRegression

# Set larger font sizes globally for all figures
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 25,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 20,
    'figure.titlesize': 25
})

# Set seaborn style with larger fonts
sns.set_context("poster", font_scale=3)
sns.set_style("white")

# Load non reasoning 1B queries energy consumption
df = pd.read_csv('1B_queries_energy_consumption_short.csv')

# Load reasoning 1B queries energy consumption
df_reasoning = pd.read_csv('1B_queries_energy_consumption_reasoning.csv')

# Calculate statistics for non-reasoning data
non_reasoning_baseline_stats = {
    'median': df['Baseline (GWh)'].median(),
    'p5': df['Baseline (GWh)'].quantile(0.05),
    'p95': df['Baseline (GWh)'].quantile(0.95)
}

non_reasoning_improved_stats = {
    'median': df['Improved (GWh)'].median(),
    'p5': df['Improved (GWh)'].quantile(0.05),
    'p95': df['Improved (GWh)'].quantile(0.95)
}

# Calculate statistics for reasoning data
reasoning_baseline_stats = {
    'median': df_reasoning['Baseline (GWh)'].median(),
    'p5': df_reasoning['Baseline (GWh)'].quantile(0.05),
    'p95': df_reasoning['Baseline (GWh)'].quantile(0.95)
}

reasoning_improved_stats = {
    'median': df_reasoning['Improved (GWh)'].median(),
    'p5': df_reasoning['Improved (GWh)'].quantile(0.05),
    'p95': df_reasoning['Improved (GWh)'].quantile(0.95)
}

print("Non-reasoning Baseline Stats:", non_reasoning_baseline_stats)
print("Non-reasoning Improved Stats:", non_reasoning_improved_stats)
print("Reasoning Baseline Stats:", reasoning_baseline_stats)
print("Reasoning Improved Stats:", reasoning_improved_stats)

# Note 30W H100 cluster at 80% utilizatio for 24 hours is 0.576 GWh
# 40 W H100 cluster at 80% utilization for 24 hours is 0.768 GWh


# Figure 1: Simple comparison of non-reasoning models
# plt.style.use('default')
fig1, ax1 = plt.subplots(figsize=(12, 8))

categories = ['Baseline', 'Line-of-sight \n Improvement']
medians = [non_reasoning_baseline_stats['median'], non_reasoning_improved_stats['median']]

# Remove error bars - just plot medians
bars1 = ax1.bar(categories, medians, color=['#45B7D1', '#4ECDC4'], alpha=0.8, width=0.6)

ax1.set_ylabel('Energy Consumption (GWh)', fontsize=14)
ax1.set_title('Traditional Regime\n1 Billion Queries per Day', fontsize=14, pad=20)
ax1.grid(True, alpha=0.3, axis='y')

# Create custom legend for the first figure and place it outside with increased font size and line breaks
legend_elements_fig1 = []
legend_elements_fig1.append(plt.Line2D([0], [0], color='#45B7D1', 
                     label=f'Baseline:\n{medians[0]:.2f} GWh', 
                     linewidth=3))
legend_elements_fig1.append(plt.Line2D([0], [0], color='#4ECDC4', 
                     label=f'Line-of-sight \n Improvement:\n{medians[1]:.2f} GWh', 
                     linewidth=3))

# Place legend outside the plot area on the right with increased font size
ax1.legend(handles=legend_elements_fig1, frameon=True, facecolor='white', 
          edgecolor='none', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)

# Remove the floating value labels above bars
# for i, (bar, median) in enumerate(zip(bars1, medians)):
#     height = bar.get_height()
#     ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
#              f'{median:.3f} GWh', ha='center', va='bottom', fontweight='bold')

# Add horizontal reference lines
ax1.axhline(y=0.576, color='gray', linestyle='--', alpha=0.7, linewidth=2)
ax1.text(0.02, 0.576 + 0.01, '30 MW H100 Datacenter', fontsize=14, 
         verticalalignment='bottom', color='black')

ax1.axhline(y=0.3, color='gray', linestyle='--', alpha=0.7, linewidth=2)
ax1.text(0.02, 0.3 + 0.01, '1 billion traditional web searches', fontsize=14, 
         verticalalignment='bottom', color='black')
from matplotlib.ticker import MaxNLocator
# Increase font size of tick labels
# ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
# Increase font size of tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Create manuscript_figures directory if it doesn't exist
import os
os.makedirs('manuscript_figures/updated_figures', exist_ok=True)

plt.tight_layout()

# Save the first figure
plt.savefig('manuscript_figures/updated_figures/FIG_1B_a.svg', format='svg', dpi=300, bbox_inches='tight')
plt.savefig('manuscript_figures/updated_figures/FIG_1B_a.png', format='png', dpi=300, bbox_inches='tight')

plt.show()

# Figure 2: Mixed comparison with weighted averages
fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))


# Calculate weighted averages (90% non-reasoning + 10% reasoning)
mixed_baseline_stats = {
    'median': 0.9 * non_reasoning_baseline_stats['median'] + 0.1 * reasoning_baseline_stats['median'],
    'p5': 0.9 * non_reasoning_baseline_stats['p5'] + 0.1 * reasoning_baseline_stats['p5'],
    'p95': 0.9 * non_reasoning_baseline_stats['p95'] + 0.1 * reasoning_baseline_stats['p95']
}

mixed_improved_stats = {
    'median': 0.9 * non_reasoning_improved_stats['median'] + 0.1 * reasoning_improved_stats['median'],
    'p5': 0.9 * non_reasoning_improved_stats['p5'] + 0.1 * reasoning_improved_stats['p5'],
    'p95': 0.9 * non_reasoning_improved_stats['p95'] + 0.1 * reasoning_improved_stats['p95']
}

print("\nMixed Baseline Stats (90% Traditional + 10% Test-Time Scaling):", mixed_baseline_stats)
print("Mixed Improved Stats (90% Traditional + 10% Test-Time Scaling):", mixed_improved_stats)

# Prepare data for stacked bars
categories_mixed = ['Baseline\nno Test-Time Scaling', 'Mixed\n10% Test-Time Scaling', 'Mixed Optimized\n10% Test-Time Scaling']

# For the mixed bars, we'll show the composition with stacked bars
# Bar 1: Pure non-reasoning baseline
bar1_medians = [non_reasoning_baseline_stats['median']]

# Bar 2: Mixed baseline (90% non-reasoning + 10% reasoning) - shown as stacked
bar2_non_reasoning_part = [0.9 * non_reasoning_baseline_stats['median']]
bar2_reasoning_part = [0.1 * reasoning_baseline_stats['median']]

# Bar 3: Mixed improved (90% non-reasoning + 10% reasoning) - shown as stacked  
bar3_non_reasoning_part = [0.9 * non_reasoning_improved_stats['median']]
bar3_reasoning_part = [0.1 * reasoning_improved_stats['median']]

# Create the bars
x_pos = np.arange(3)
width = 0.6

# Apply plot styling consistent with the reference file
# plt.style.use('default')

# Bar 1: Simple bar for non-reasoning baseline (Traditional)
bars_1 = ax2.bar([0], bar1_medians, width, color='#B19CD9', alpha=0.8, label='Traditional')

# Bar 2: Stacked bar for mixed baseline
bars_2a = ax2.bar([1], bar2_non_reasoning_part, width, color='#45B7D1', alpha=0.8)
bars_2b = ax2.bar([1], bar2_reasoning_part, width, bottom=bar2_non_reasoning_part, 
                  color='#FF6B6B', alpha=0.8, label='Test-Time Scaling')

# Bar 3: Stacked bar for mixed improved
bars_3a = ax2.bar([2], bar3_non_reasoning_part, width, color='#4ECDC4', alpha=0.8, label='Traditional (Improved)')
bars_3b = ax2.bar([2], bar3_reasoning_part, width, bottom=bar3_non_reasoning_part, 
                  color='#96CEB4', alpha=0.8, label='Test-Time Scaling (Improved)')

# Get medians for labeling (no error bars)
all_medians = [non_reasoning_baseline_stats['median'], mixed_baseline_stats['median'], mixed_improved_stats['median']]

ax2.set_ylabel('Energy Consumption (GWh)', fontsize=14)
ax2.set_title('Traditional vs Mixed (10% Test-Time Scaling) Regime\n1 Billion Queries per Day', fontsize=14, pad=20)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(categories_mixed)
ax2.grid(True, alpha=0.3, axis='y')

# Create custom legend entries with proportional values that match what's shown in bars
legend_elements = []
legend_elements.append(plt.Line2D([0], [0], color='#B19CD9', 
                     label=f'Traditional (Baseline):\n{non_reasoning_baseline_stats["median"]:.2f} GWh', 
                     linewidth=3))
legend_elements.append(plt.Line2D([0], [0], color='#45B7D1', 
                     label=f'Traditional (Baseline):\n{0.9 * non_reasoning_baseline_stats["median"]:.2f} GWh', 
                     linewidth=3))
legend_elements.append(plt.Line2D([0], [0], color='#FF6B6B', 
                     label=f'Test-Time Scaling (Baseline):\n{0.1 * reasoning_baseline_stats["median"]:.2f} GWh', 
                     linewidth=3))
legend_elements.append(plt.Line2D([0], [0], color='#4ECDC4', 
                     label=f'Traditional (Line-of-sight \nImprovement):\n{0.9 * non_reasoning_improved_stats["median"]:.2f} GWh', 
                     linewidth=3))
legend_elements.append(plt.Line2D([0], [0], color='#96CEB4', 
                     label=f'Test-Time Scaling (Line-of-sight \nImprovement):\n{0.1 * reasoning_improved_stats["median"]:.2f} GWh', 
                     linewidth=3))

# Place legend outside the plot area on the right with increased font size
ax2.legend(handles=legend_elements, frameon=True, facecolor='white', 
          edgecolor='none', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)

# Remove the floating value labels above bars
# for i, median in enumerate(all_medians):
#     ax2.text(i, median + 0.2,
#              f'{median:.3f} GWh', ha='center', va='bottom', fontweight='bold')

# Add horizontal reference lines
ax2.axhline(y=0.576, color='gray', linestyle='--', alpha=0.7, linewidth=2)
ax2.text(0.02, 0.576 + 0.01, '30 MW H100 Datacenter', fontsize=14, 
         verticalalignment='bottom', color='black')

ax2.axhline(y=0.3, color='gray', linestyle='--', alpha=0.7, linewidth=2)
ax2.text(0.02, 0.3 + 0.01, '1 billion traditional web searches', fontsize=14, 
         verticalalignment='bottom', color='black')

from matplotlib.ticker import MaxNLocator
# Increase font size of tick labels
ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
# Increase font size of tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
# Save the second figure
plt.savefig('manuscript_figures/updated_figures/FIG_1B_b.svg', format='svg', dpi=300, bbox_inches='tight')
plt.savefig('manuscript_figures/updated_figures/FIG_1B_b.png', format='png', dpi=300, bbox_inches='tight')

plt.show()

# Print summary statistics
print(f"\n=== SUMMARY STATISTICS ===")
print(f"Traditional Baseline: {non_reasoning_baseline_stats['median']:.2f} GWh (median)")
print(f"Traditional Improved: {non_reasoning_improved_stats['median']:.2f} GWh (median)")
print(f"Mixed Baseline (90%+10%): {mixed_baseline_stats['median']:.2f} GWh (median)")
print(f"Mixed Improved (90%+10%): {mixed_improved_stats['median']:.2f} GWh (median)")
print(f"\nImprovement from Baseline to Improved:")
print(f"  Traditional: {((non_reasoning_baseline_stats['median'] - non_reasoning_improved_stats['median']) / non_reasoning_baseline_stats['median'] * 100):.1f}% reduction")
print(f"  Mixed Workload: {((mixed_baseline_stats['median'] - mixed_improved_stats['median']) / mixed_baseline_stats['median'] * 100):.1f}% reduction")





# %%
