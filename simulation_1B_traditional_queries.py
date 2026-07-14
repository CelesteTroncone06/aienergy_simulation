"""
PARAMETER ALIGNMENT WITH estimate_per_mode_regression.py:

1. POWER PARAMETERS:
   - DeepSeek-R1: 14.1 kW (was 14.3 kW)
   - Other models: 11.3 kW (was 11.9 kW)

2. EFFICIENCY RANGES:
   - PU range: (0.4, 0.9) (was (0.3, 0.7))
   - PUE range: (1.05, 1.6) (was (1.05, 1.4))

3. IMPROVEMENT MULTIPLIERS:
   - Hardware: lognorm(1.5, 2.5) (was lognorm(2.5, 3))
   - Algorithm/Model: lognorm(1.5, 10) (was lognorm(1.5, 9))
   - Improved serving: lognorm(1.5, 3) (was lognorm(1.5, 2.5))

4. STYLING:
   - Font sizes aligned with reference file
   - Figure size: (10, 8) (was (15, 8) for fig1, (12, 10) for fig2)
   - Seaborn font scale: 2.5 (was 1.8)
   - All legend and tick font sizes updated to match

---------------------------------------------------------------------

This script estimates energy consumption per query for open-weight
language models running on H100 GPUs in the traditional-query regime.

The simulation uses Monte Carlo sampling to estimate the distribution
of energy per query, accounting for variability in model throughput,
query characteristics and system efficiency.

Inputs:
- model_throughput_DB.csv: throughput data for different models

Outputs:
- Energy per query estimates
- Figures corresponding to the traditional-query analysis
"""

import numpy as np # numerical operations and random sampling
import matplotlib.pyplot as plt
import pandas as pd # loads and processes model throughput data
import seaborn as sns # statistical data visualisation
import matplotlib
# linear regression model for TPS/throughput
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
sns.set_context("poster", font_scale=2.5)
sns.set_style("white")

# Seed for reproducibility
np.random.seed(42)

# Load model_throughput_consolidated_DB.csv
model_throughput = pd.read_csv('model_throughput_DB.csv')
print(model_throughput.columns)

# Drop those with Quantization != "FP8"
model_throughput = model_throughput[model_throughput['Quantization'] == 'FP8']
# dropped models that do not use FP8 as the authors assume FP8, balances
# memory and speed

#%%
# Helper function to get log-normal parameters from (5th, 95th percentile)
def lognorm_params(min_val, max_val):
    # Convert minimum and maximum values into parameters of a log-normal
    # distribution. The method assumes that min_val and max_val represent
    # approximately the 5th and 95th percentiles of the distribution.
    # 1.645 is the 95th percentile of a standard normal distribution.
    # For a normal distribution, ~90% of values lie between
    # mean - 1.645*sigma and mean + 1.645*sigma
    sigma = (np.log(max_val) - np.log(min_val)) / (2 * 1.645)
    mu = np.log(min_val) + 1.645 * sigma
    return mu, sigma

def create_tps_regression_models(model_data):
    """Create regression models for each model to predict TPS from input/output lengths"""
    regression_models = {}
    interpolation_models = {}
    max_tps_values = {}  # Store maximum TPS for each model
    
    # Get numeric TPS values
    model_data['TPS_numeric'] = pd.to_numeric(model_data['Tokens per Second (TPS)'], errors='coerce')
    
    # Convert Input Length and Output Length to numeric (handle non-numeric values)
    model_data['Input_Length_numeric'] = pd.to_numeric(model_data['Input Length'], errors='coerce')
    model_data['Output_Length_numeric'] = pd.to_numeric(model_data['Output Length'], errors='coerce')
    
    print("Building TPS Models:")
    print("=" * 50)
    
    for model_name in model_data['Model'].unique():
        model_subset = model_data[model_data['Model'] == model_name].copy()
        
        # Remove any rows with NaN values
        model_subset = model_subset.dropna(subset=['Input_Length_numeric', 'Output_Length_numeric', 'TPS_numeric'])
        
        if len(model_subset) < 2:  # Need at least 2 points for interpolation
            print(f"{model_name}: Insufficient data points ({len(model_subset)}), skipping")
            continue
        
        # Store the maximum TPS value for this model (for capping predictions)
        max_tps_values[model_name] = model_subset['TPS_numeric'].max()
        
        if len(model_subset) < 3:  # Use highest TPS for 2 points
            print(f"{model_name}: Using highest TPS (only {len(model_subset)} data points)")
            
            # Use the highest TPS value from available data
            max_tps = model_subset['TPS_numeric'].max()  # Changed from mean to max
            
            interpolation_models[model_name] = {
                'type': 'max_tps',
                'max_tps': max_tps,
                'n_points': len(model_subset),
                'tps_range': (model_subset['TPS_numeric'].min(), model_subset['TPS_numeric'].max())
            }
            
            print(f"  Data points: {len(model_subset)}")
            print(f"  TPS range: {model_subset['TPS_numeric'].min():.2f} - {model_subset['TPS_numeric'].max():.2f}")
            print(f"  Using max TPS: {max_tps:.2f}")
            print()
            
        else:  # Use regression for 3+ points
            print(f"{model_name}: Using regression ({len(model_subset)} data points)")
            
            # Features: [Input_Length, Output_Length]
            X = model_subset[['Input_Length_numeric', 'Output_Length_numeric']].values
            y = model_subset['TPS_numeric'].values
            
            # Log-linear regression (handle zeros by adding small epsilon)
            X_log = np.log(np.maximum(X, 1e-6))  # Avoid log(0)
            y_log = np.log(np.maximum(y, 1e-6))
            
            # Fit regression model
            reg = LinearRegression()
            reg.fit(X_log, y_log)
            
            regression_models[model_name] = {
                'type': 'regression',
                'model': reg,
                'n_points': len(model_subset),
                'input_range': (X[:, 0].min(), X[:, 0].max()),
                'output_range': (X[:, 1].min(), X[:, 1].max()),
                'max_tps': max_tps_values[model_name]  # Store max TPS for capping
            }
            
            print(f"  Data points: {len(model_subset)}")
            print(f"  Input length range: {X[:, 0].min():.0f} - {X[:, 0].max():.0f}")
            print(f"  Output length range: {X[:, 1].min():.0f} - {X[:, 1].max():.0f}")
            print(f"  Max TPS (cap): {max_tps_values[model_name]:.2f}")
            print()
    
    return regression_models, interpolation_models, max_tps_values

def predict_tps_for_lengths(model_name, input_length, output_length, regression_models, interpolation_models, max_tps_values):
    """Predict TPS for given input and output lengths using regression or interpolation, capped at max observed TPS"""
    
    # Try regression first
    if model_name in regression_models:
        # Log transform the features
        log_features = np.array([[np.log(max(input_length, 1e-6)), np.log(max(output_length, 1e-6))]])
        
        # Predict log(TPS)
        log_tps_pred = regression_models[model_name]['model'].predict(log_features)[0]
        
        # Transform back to original scale
        predicted_tps = np.exp(log_tps_pred)
        
        # Cap at maximum observed TPS for this model
        max_tps = regression_models[model_name]['max_tps']
        return min(predicted_tps, max_tps)
    
    # Try interpolation fallback
    elif model_name in interpolation_models:
        model_info = interpolation_models[model_name]
        
        if model_info['type'] == 'max_tps':
            # Use the maximum TPS value (already capped by design)
            return model_info['max_tps']
    
    else:
        return None

# Build regression models
tps_regression_models, interpolation_models, max_tps_values = create_tps_regression_models(model_throughput)

# Simulation settings --- CHANGE THESE ---
n_runs = 10000 # n_runs is large to simulate stochastic query lengths for Monte Carlo
median_tokens = 300   # median tokens query length
fixed_input_length = 500  # Fixed input length for predictions
# Calculate lambda parameter for exponential distribution to achieve desired median
lambda_param = np.log(2) / median_tokens  # For exponential, median = ln(2)/λ

# Define ranges and values
def get_node_power(model_name):
    return 12.8 if model_name == 'DeepSeek-R1' else 10.2

pu_range = (0.4, 0.9)        # for lognormal, as 0.7Pmax is where it is centred
PUE_range = (1.05, 1.6)       # for lognormal, as PUE ranges between those values

# Compute log-normal parameters where needed
mu_pu, sigma_pu = lognorm_params(*pu_range)
mu_pue, sigma_pue = lognorm_params(*PUE_range)

# Create separate distributions for each model using regression
all_model_energies = {}
all_model_tps = {}  # Add this to store TPS predictions
print("\nGenerating Energy Distributions:")
print("=" * 50)

# Combine both model types for processing
all_tps_models = {**tps_regression_models, **interpolation_models}

for model_name in all_tps_models.keys():
    print(f"Processing {model_name}...")
    
    # Get model-specific node power
    node_power = get_node_power(model_name)
    
    # Generate random output token lengths (exponential distribution)
    model_token_lengths = np.round(np.random.exponential(1/lambda_param, n_runs)).astype(int)
    
    # Predict TPS for each token length using regression or interpolation
    model_tokens_per_sec = np.array([
        predict_tps_for_lengths(model_name, fixed_input_length, token_length, tps_regression_models, interpolation_models, max_tps_values)
        for token_length in model_token_lengths
    ])
    
    # Handle any None values (fallback to mean if needed)
    valid_tps = model_tokens_per_sec[model_tokens_per_sec != None]
    if len(valid_tps) == 0:
        print(f"  Warning: No valid TPS predictions for {model_name}, skipping")
        continue
    
    model_tokens_per_sec = model_tokens_per_sec.astype(float)
    
    # Store TPS predictions for this model
    all_model_tps[model_name] = model_tokens_per_sec
    
    # Calculate energies for this model
    model_node_power_array = np.full(n_runs, node_power)  # Use model-specific power value
    model_pu = np.random.lognormal(mu_pu, sigma_pu, n_runs)
    model_pue = np.random.lognormal(mu_pue, sigma_pue, n_runs)
    
    # Calculate base energies for this model
    model_energies = np.empty(n_runs)
    for i in range(n_runs):
        energy_kj = model_pue[i] * (model_node_power_array[i] * model_pu[i] * model_token_lengths[i]) / model_tokens_per_sec[i]
        model_energies[i] = (energy_kj / 3600) * 1000
    
    all_model_energies[model_name] = model_energies
    print(f"  Generated {n_runs} energy samples")

# Create violin plots for each model
plot_data_list = []
for model_name, energies in all_model_energies.items():
    if not np.isnan(energies).all():  # Skip models with all NaN values
        # Filter outliers (5-95 percentile) - this filtered data will be used for both KDE and boxplot
        p5, p95 = np.percentile(energies, [5, 95])
        filtered_energies = energies[(energies >= p5) & (energies <= p95)]  # Changed to inclusive bounds
        
        df_model = pd.DataFrame({
            'Energy (Wh)': filtered_energies,
            'Model': model_name
        })
        plot_data_list.append(df_model)

# Combine all model data
plot_data_combined = pd.concat(plot_data_list)

# Define the desired order of models
model_order = [
    'DeepSeek-R1',
    'Llama 3.1 405B',
    'Llama-3.1 Nemotron Ultra 253B',
    'Mixtral 8x22B',
    'Llama 3.1 70B'
]

# Create a custom color palette
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
color_dict = dict(zip(model_order, colors))

# Set figure style
plt.style.use('default')
plt.figure(figsize=(10, 8))

# Create violin plot with custom parameters
ax = sns.violinplot(data=plot_data_combined, 
                    x='Energy (Wh)', 
                    y='Model',
                    hue='Model',  # Use hue instead of direct palette
                    order=model_order,
                    orient='h',
                    inner=None,  # Don't show internal lines (we'll add them manually)
                    cut=0,  # Don't extend the KDE below 0
                    width=0.9,  # Make the violins wider
                    density_norm='width',  # Scale all violins to the same width
                    palette=color_dict,
                    legend=False,  # Don't show the legend since it's redundant
                    bw_adjust=1)  # Adjust bandwidth for smoother KDE

# Add quartile lines manually for each violin
for i, model_name in enumerate(model_order):
    if model_name in all_model_energies:
        energies = all_model_energies[model_name]
        if not np.isnan(energies).all():
            # Get quartiles
            p25, p50, p75 = np.percentile(energies, [25, 50, 75])
            # Add lines at each quartile
            ax.hlines(y=i, xmin=p25, xmax=p75, color='black', linewidth=2, alpha=0.7)  # IQR line
            ax.vlines(x=p50, ymin=i-0.1, ymax=i+0.1, color='white', linewidth=3)  # Median line
            ax.vlines(x=p50, ymin=i-0.1, ymax=i+0.1, color='black', linewidth=2)  # Median line border

# Remove any whisker lines that might remain
for artist in ax.get_children():
    if isinstance(artist, matplotlib.lines.Line2D):
        if artist.get_linestyle() == '--':  # This catches the whisker lines
            artist.set_visible(False)

# Create custom legend entries with median and Q1/Q3 values
legend_elements = []
for model_name in model_order:
    if model_name in all_model_energies:
        energies = all_model_energies[model_name]
        if not np.isnan(energies).all():
            p25, p50, p75 = np.percentile(energies, [25, 50, 75])
            legend_elements.append(plt.Line2D([0], [0], color=color_dict[model_name], 
                                 label=f'{model_name}: {p50:.3f} Wh (Q1:{p25:.3f}, Q3:{p75:.3f})', 
                                 linewidth=3))

# Add legend
ax.legend(handles=legend_elements, frameon=True, facecolor='white', 
         edgecolor='none', loc='lower right', bbox_to_anchor=(1, 0),
         fontsize=12)

# Customize the plot
plt.title('Per Query Energy Consumption (P5-P95)', fontsize=14, pad=20)
plt.xlabel('Energy per Query (Wh)', fontsize=12)
plt.grid(True, alpha=0.3)

# Remove y-axis label since it's redundant
ax.set_ylabel('')

# Increase font size of tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add a light grid for better readability
ax.yaxis.grid(True, linestyle='--', alpha=0.3)

# Adjust layout to prevent text cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print statistics for each model in the specified order
print("\nModel-specific Statistics (based on 5-95 percentile filtered data):")
for model_name in model_order:
    if model_name in all_model_energies:
        energies = all_model_energies[model_name]
        if not np.isnan(energies).all():
            p5, p25, p50, p75, p95 = np.percentile(energies, [5, 25, 50, 75, 95])
            mean = np.mean(energies)
            print(f"\n{model_name}:")
            print(f"  5th percentile: {p5:.3f} Wh")
            print(f"  Q1 (25th): {p25:.3f} Wh")
            print(f"  Median: {p50:.3f} Wh")
            print(f"  Q3 (75th): {p75:.3f} Wh")
            print(f"  95th percentile: {p95:.3f} Wh")
            print(f"  Mean: {mean:.3f} Wh")

# Print TPS statistics for each model
print("\nModel-specific TPS Statistics:")
print("=" * 50)
for model_name in model_order:
    if model_name in all_model_tps:
        tps_values = all_model_tps[model_name]
        if not np.isnan(tps_values).all():
            p5, p25, p50, p75, p95 = np.percentile(tps_values, [5, 25, 50, 75, 95])
            mean = np.mean(tps_values)
            std = np.std(tps_values)
            print(f"\n{model_name}:")
            print(f"  5th percentile: {p5:.1f} TPS")
            print(f"  Q1 (25th): {p25:.1f} TPS")
            print(f"  Median: {p50:.1f} TPS")
            print(f"  Q3 (75th): {p75:.1f} TPS")
            print(f"  95th percentile: {p95:.1f} TPS")
            print(f"  Mean: {mean:.1f} TPS")
            print(f"  Std Dev: {std:.1f} TPS")

# Print regression model summary
print("\n" + "="*60)
print("TPS MODEL SUMMARY")
print("="*60)
print(f"Fixed Input Length Used: {fixed_input_length} tokens")
print(f"Output Length Distribution: Exponential (median = {median_tokens} tokens)")
print()

# Print regression models first
for model_name in sorted(tps_regression_models.keys()):
    model_info = tps_regression_models[model_name]
    n_points = model_info['n_points']
    
    print(f"{model_name} (Regression):")
    print(f"  Training Points: {n_points}")
    print(f"  Input Range: {model_info['input_range'][0]:.0f} - {model_info['input_range'][1]:.0f} tokens")
    print(f"  Output Range: {model_info['output_range'][0]:.0f} - {model_info['output_range'][1]:.0f} tokens")
    
    # Show regression coefficients (log-linear model interpretation)
    coef = model_info['model'].coef_
    intercept = model_info['model'].intercept_
    print(f"  Model: log(TPS) = {intercept:.3f} + {coef[0]:.3f}*log(input) + {coef[1]:.3f}*log(output)")
    print()

# Print interpolation models
max_tps_models_found = False
for model_name in sorted(interpolation_models.keys()):
    max_tps_models_found = True
    model_info = interpolation_models[model_name]
    n_points = model_info['n_points']
    
    print(f"{model_name} (Max TPS):")
    print(f"  Method: Using highest observed TPS (optimistic)")
    print(f"  Training Points: {n_points}")
    print(f"  TPS Range: {model_info['tps_range'][0]:.0f} - {model_info['tps_range'][1]:.0f}")
    print(f"  Using TPS: {model_info['max_tps']:.0f} (max value)")
    print()

if max_tps_models_found:
    print(f"Max TPS models: {len(interpolation_models)}")

print(f"Total models processed: {len(all_tps_models)}")
# Print mean of the token output length
print(f"Mean of the token output length: {np.mean(model_token_lengths)}")
print(f"Std of the token output length: {np.std(model_token_lengths)}")
# Print Q1 and Q3 of the token output length
print(f"Q1 of the token output length: {np.percentile(model_token_lengths, 25)}")
print(f"Q3 of the token output length: {np.percentile(model_token_lengths, 75)}")
# Print P5 and P95 of the token output length
print(f"P5 of the token output length: {np.percentile(model_token_lengths, 5)}")
print(f"P95 of the token output length: {np.percentile(model_token_lengths, 95)}")
# Print mean of the token output length
print(f"Mean of the token output length: {np.mean(model_token_lengths)}")
# Print std of the token output length
print(f"Std of the token output length: {np.std(model_token_lengths)}")
print(f"Max of the token output length: {np.max(model_token_lengths)}")
print(f"Min of the token output length: {np.min(model_token_lengths)}")
# %%

#%%
# Figure 2: Mixed distribution of top 3 models (by highest median energy consumption)
print("\n" + "="*60)
print("FIGURE 2: TOP 3 MOST ENERGY INTENSIVE MODELS MIXED DISTRIBUTION")
print("="*60)

# Calculate median energy for each model to determine top 3
model_medians = {}
for model_name in model_order:
    if model_name in all_model_energies:
        energies = all_model_energies[model_name]
        if not np.isnan(energies).all():
            median_energy = np.median(energies)
            model_medians[model_name] = median_energy

# Sort models by median energy (descending - most energy intensive first)
sorted_models = sorted(model_medians.items(), key=lambda x: x[1], reverse=True)
top_3_models = [model[0] for model in sorted_models[:3]]

print(f"Top 3 most energy intensive models (by median energy consumption):")
for i, model_name in enumerate(top_3_models, 1):
    median_energy = model_medians[model_name]
    print(f"{i}. {model_name}: {median_energy:.3f} Wh")

# Combine energy distributions from top 3 models
# Two-stage filtering: First apply standard 5-95 (consistent with Figure 1), 
# then optionally apply additional filtering for variance reduction
additional_filtering = True  # Set to False to use only standard 5-95 filtering

mixed_energies = []
print(f"\nUsing two-stage filtering approach:")
print(f"  Stage 1: Standard 5-95 percentile filtering (consistent with Figure 1)")
print(f"  Stage 2: Additional filtering {'ENABLED' if additional_filtering else 'DISABLED'}")

for model_name in top_3_models:
    if model_name in all_model_energies:
        energies = all_model_energies[model_name]
        if not np.isnan(energies).all():
            
            # Stage 1: Standard 5-95 percentile filtering (consistent with Figure 1)
            p5, p95 = np.percentile(energies, [5, 95])
            stage1_filtered = energies[(energies >= p5) & (energies <= p95)]
            
            # Stage 2: Additional filtering on the already filtered data
            if additional_filtering:
                # Apply 10-90 percentile filtering on the Stage 1 filtered data
                p10_stage2, p90_stage2 = np.percentile(stage1_filtered, [10, 90])
                final_filtered = stage1_filtered[(stage1_filtered >= p10_stage2) & (stage1_filtered <= p90_stage2)]
            else:
                final_filtered = stage1_filtered
            
            # Show the effect of both filtering stages
            original_std = np.std(energies)
            stage1_std = np.std(stage1_filtered)
            final_std = np.std(final_filtered)
            
            stage1_reduction = (1 - stage1_std/original_std) * 100
            total_reduction = (1 - final_std/original_std) * 100
            
            print(f"  {model_name}:")
            print(f"    Original: {len(energies):,} samples, std: {original_std:.3f} Wh")
            print(f"    Stage 1:  {len(stage1_filtered):,} samples, std: {stage1_std:.3f} Wh (reduction: {stage1_reduction:.1f}%)")
            print(f"    Final:    {len(final_filtered):,} samples, std: {final_std:.3f} Wh (total reduction: {total_reduction:.1f}%)")
            
            mixed_energies.extend(final_filtered)

mixed_energies = np.array(mixed_energies)

# Add improvement pathways using the same logic as fig1.py
print("\n" + "="*40)
print("APPLYING IMPROVEMENT PATHWAYS")
print("="*40)

# Define improvement multipliers (same as fig1.py)
mu_hardware, sigma_hardware = lognorm_params(1.5, 2.5)  # Hardware multiplier
mu_algorithm, sigma_algorithm = lognorm_params(1.5, 10)  # Algorithm/Model multiplier
mu_improved, sigma_improved = lognorm_params(1.5, 3)  # Improved serving multiplier

# Apply improvements by dividing energy by multipliers (since multipliers increase efficiency)
# Note: Apply multipliers to the already-filtered baseline data
n_mixed_samples = len(mixed_energies)
hardware_multiplier = np.random.lognormal(mu_hardware, sigma_hardware, n_mixed_samples)
algorithm_multiplier = np.random.lognormal(mu_algorithm, sigma_algorithm, n_mixed_samples)
improved_multiplier = np.random.lognormal(mu_improved, sigma_improved, n_mixed_samples)

hardware_energies = mixed_energies / hardware_multiplier
algorithm_energies = mixed_energies / algorithm_multiplier
improved_energies = mixed_energies / improved_multiplier

print(f"Applied improvement multipliers to {n_mixed_samples:,} baseline samples (after all filtering)")
print(f"Hardware multiplier range: {hardware_multiplier.min():.2f} - {hardware_multiplier.max():.2f}")
print(f"Algorithm/Model multiplier range: {algorithm_multiplier.min():.2f} - {algorithm_multiplier.max():.2f}")
print(f"Improved serving multiplier range: {improved_multiplier.min():.2f} - {improved_multiplier.max():.2f}")

print(f"\nApplying 5-95 percentile filtering to improvement categories to remove outliers (consistent with Figure 1):")

# Create DataFrame for violin plot with all distributions
def prepare_improvement_data(energies, category):
    # Apply 5-95 percentile filtering to improvement categories to remove outliers created by multipliers
    # This is consistent with how Figure 1 handles each model's data
    if category != 'Baseline':
        # Apply same 5-95 percentile filtering as used in Figure 1
        p5, p95 = np.percentile(energies, [5, 95])
        filtered_energies = energies[(energies >= p5) & (energies <= p95)]
        print(f"    {category}: Applied 5-95 percentile filtering {len(energies):,} → {len(filtered_energies):,} samples")
        return pd.DataFrame({
            'Energy (Wh)': filtered_energies,
            'Distribution': category
        })
    else:
        # Baseline already has all filtering applied
        return pd.DataFrame({
            'Energy (Wh)': energies,
            'Distribution': category
        })

# Combine all distributions
plot_data_all = pd.concat([
    prepare_improvement_data(mixed_energies, 'Baseline'),
    prepare_improvement_data(hardware_energies, 'Hardware'),
    prepare_improvement_data(algorithm_energies, 'Algorithm/Model'),
    prepare_improvement_data(improved_energies, 'Improved\nServing')
])

# Define colors for each category (same as fig1.py)
colors = {
    'Baseline': '#2ecc71',
    'Hardware': '#3498db',
    'Algorithm/Model': '#9b59b6',
    'Improved\nServing': '#e67e22'
}

# Create Figure 2 - Multiple violin plots with improvements (matching fig1.py style)
plt.style.use('default')  
plt.figure(figsize=(10, 8))

# Create violin plot with same parameters as Figure 1
ax = sns.violinplot(data=plot_data_all, 
                    x='Energy (Wh)', 
                    y='Distribution',
                    orient='h',
                    inner=None,  # Don't show internal lines (we'll add them manually) - SAME AS FIGURE 1
                    cut=0,  # Don't extend the KDE below 0
                    width=0.9,  # Same as Figure 1
                    density_norm='width',  # Scale all violins to the same width
                    palette=colors,
                    bw_adjust=1)  # Same as Figure 1

# Add quartile lines manually for each violin (EXACT SAME AS FIGURE 1)
distribution_names = ['Baseline', 'Hardware', 'Algorithm/Model', 'Improved\nServing']
energies_list = [mixed_energies, hardware_energies, algorithm_energies, improved_energies]

for i, (dist_name, energies) in enumerate(zip(distribution_names, energies_list)):
    # Get quartiles from the actual data being plotted
    p25, p50, p75 = np.percentile(energies, [25, 50, 75])
    # Add lines at each quartile (EXACT SAME AS FIGURE 1)
    ax.hlines(y=i, xmin=p25, xmax=p75, color='black', linewidth=2, alpha=0.7)  # IQR line
    ax.vlines(x=p50, ymin=i-0.1, ymax=i+0.1, color='white', linewidth=3)  # Median line
    ax.vlines(x=p50, ymin=i-0.1, ymax=i+0.1, color='black', linewidth=2)  # Median line border

# Remove any whisker lines that might remain (SAME AS FIGURE 1)
for artist in ax.get_children():
    if isinstance(artist, matplotlib.lines.Line2D):
        if artist.get_linestyle() == '--':  # This catches the whisker lines
            artist.set_visible(False)

# Create custom legend entries with median values (same logic as fig1.py)
legend_elements = []
for category, energies in [
    ('Baseline', mixed_energies),
    ('Hardware', hardware_energies),
    ('Algorithm/Model', algorithm_energies),
    ('Improved\nServing', improved_energies)
]:
    # Use the same energies that are actually plotted (no additional filtering)
    p25, p50, p75 = np.percentile(energies, [25, 50, 75])
    legend_elements.append(plt.Line2D([0], [0], color=colors[category], 
                         label=f'{category.replace(chr(10), " ")}: {p50:.3f} Wh (Q1:{p25:.3f}, Q3:{p75:.3f})', 
                         linewidth=2))

# Add legend with same styling as Figure 1
ax.legend(handles=legend_elements, frameon=True, facecolor='white', 
         edgecolor='none', loc='lower right', bbox_to_anchor=(1, 0),
         fontsize=14)  # Same as Figure 1

# Customize the plot (matching Figure 1 styling)
plt.title(f'Energy Distribution with Improvement Pathways (P5-P95) \nBaseline: Blend of Models >200B parameters)', 
          fontsize=14, pad=20)  # Same as Figure 1
plt.xlabel('Energy per Query (Wh)', fontsize=12)  # Same as Figure 1
plt.grid(True, alpha=0.3)

# Remove y-axis label since it's redundant (same as Figure 1)
ax.set_ylabel('')

# Increase font size of tick labels (same as Figure 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add a light grid for better readability (same as Figure 1)
ax.yaxis.grid(True, linestyle='--', alpha=0.3)

# Adjust layout (same as Figure 1)
plt.tight_layout()

# Show the plot
plt.show()

# Print statistics for all distributions (using same data as plotted)
print(f"\nDistribution Statistics (same data as plotted):")
print("="*60)

for category, energies in [
    ('Baseline (Mixed Top 3)', mixed_energies),
    ('Hardware', hardware_energies),
    ('Algorithm/Model', algorithm_energies),
    ('Improved Serving', improved_energies)
]:
    # Use same data that's actually plotted (no additional filtering)
    p5, p25, p50, p75, p95 = np.percentile(energies, [5, 25, 50, 75, 95])
    mean = np.mean(energies)
    std = np.std(energies)
    
    print(f"\n{category} (n = {len(energies):,} samples):")
    print(f"  5th percentile: {p5:.3f} Wh")
    print(f"  Q1 (25th): {p25:.3f} Wh")
    print(f"  Median: {p50:.3f} Wh")
    print(f"  Q3 (75th): {p75:.3f} Wh")
    print(f"  95th percentile: {p95:.3f} Wh")
    print(f"  Mean: {mean:.3f} Wh")
    print(f"  Std Dev: {std:.3f} Wh")

# Print improvement percentages relative to baseline (using same data as plotted)
baseline_median = np.median(mixed_energies)

print(f"\nImprovement vs Baseline (using same data as plotted):")
print("="*30)
for category, energies in [
    ('Hardware', hardware_energies),
    ('Algorithm/Model', algorithm_energies),
    ('Improved Serving', improved_energies)
]:
    improved_median = np.median(energies)
    improvement_pct = (1 - improved_median/baseline_median) * 100
    print(f"{category}: {improvement_pct:.1f}% reduction ({baseline_median:.3f} → {improved_median:.3f} Wh)")

# %%

# Figure 3: Token Length Distribution
print("\n" + "="*60)
print("FIGURE 3: TOKEN LENGTH DISTRIBUTION")
print("="*60)

print(f"Distribution parameters:")
print(f"  Type: Exponential")
print(f"  Median: {median_tokens} tokens")
print(f"  Lambda parameter: {lambda_param:.6f}")
print(f"  Number of samples: {n_runs:,}")

# Generate a fresh sample for visualization (same parameters)
np.random.seed(42)  # For reproducibility
token_lengths_viz = np.round(np.random.exponential(1/lambda_param, n_runs)).astype(int)

# Create Figure 3 - Token Length Distribution
plt.style.use('default')
plt.figure(figsize=(12, 8), facecolor='white')

# Create histogram with bins
bins = np.arange(0, np.percentile(token_lengths_viz, 99) + 50, 50)  # 50-token bins up to 99th percentile
plt.hist(token_lengths_viz, bins=bins, density=True, alpha=0.7, color='#3498db', 
         edgecolor='black', linewidth=0.5, label=f'Histogram (n={n_runs:,})')

# Add vertical lines for key statistics
median_actual = np.median(token_lengths_viz)
mean_actual = np.mean(token_lengths_viz)
max_actual = np.max(token_lengths_viz)
p25, p75 = np.percentile(token_lengths_viz, [25, 75])

plt.axvline(median_actual, color='orange', linestyle='--', linewidth=2, 
           label=f'Median: {median_actual:.0f} tokens (Q1:{p25:.0f}, Q3:{p75:.0f})')
plt.axvline(mean_actual, color='red', linestyle=':', linewidth=2, 
           label=f'Mean: {mean_actual:.0f} tokens')
plt.axvline(max_actual, color='purple', linestyle='-.', linewidth=2, 
           label=f'Max: {max_actual:,} tokens')

# Customize the plot
plt.title('Query Output Tokens \n(Exponential Distribution, P5-P95)', fontsize=14, pad=20)
plt.xlabel('Output Tokens per Query', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=12)

# Set reasonable x-axis limit to focus on the main distribution
plt.xlim(25, np.percentile(token_lengths_viz, 95))

# Increase font size of tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Print statistics for token length distribution
print(f"\nToken Length Distribution Statistics:")
print("="*40)
p5, p25, p50, p75, p95 = np.percentile(token_lengths_viz, [5, 25, 50, 75, 95])
mean_tokens = np.mean(token_lengths_viz)
std_tokens = np.std(token_lengths_viz)
total_tokens = np.sum(token_lengths_viz)  # Add total sum calculation

print(f"Sample size: {len(token_lengths_viz):,}")
print(f"Total tokens: {total_tokens:,}")  # Add total tokens info
print(f"5th percentile: {p5:.0f} tokens")
print(f"Q1 (25th): {p25:.0f} tokens")
print(f"Median: {p50:.0f} tokens")
print(f"Q3 (75th): {p75:.0f} tokens")
print(f"95th percentile: {p95:.0f} tokens")
print(f"Mean: {mean_tokens:.1f} tokens")
print(f"Std Dev: {std_tokens:.1f} tokens")
print(f"Max: {np.max(token_lengths_viz):,} tokens")
print(f"Min: {np.min(token_lengths_viz)} tokens")

# Theoretical vs Actual comparison
theoretical_mean = 1/lambda_param
theoretical_median = np.log(2)/lambda_param
print(f"\nTheoretical vs Actual:")
print(f"Mean - Theoretical: {theoretical_mean:.1f}, Actual: {mean_tokens:.1f}")
print(f"Median - Theoretical: {theoretical_median:.1f}, Actual: {median_actual:.1f}")

# %%

# Figure 4: Estimated Total Energy for 1B Queries
print("\n" + "="*60)
print("FIGURE 4: ESTIMATED TOTAL ENERGY FOR 1B QUERIES")
print("="*60)

# Get the complete baseline distribution (before P5-P95 filtering)
complete_baseline_energies = []
for model_name in top_3_models:
    if model_name in all_model_energies:
        energies = all_model_energies[model_name]
        if not np.isnan(energies).all():
            complete_baseline_energies.extend(energies)

complete_baseline_energies = np.array(complete_baseline_energies)

print(f"Complete baseline distribution (n={len(complete_baseline_energies):,})")

# Calculate sum for complete distribution
complete_sum_kwh = np.sum(complete_baseline_energies) / 1000
print(f"Original baseline sum: {complete_sum_kwh:.2f} kWh")

# Generate improved serving energies (same as in Figure 2)
n_baseline_samples = len(complete_baseline_energies)
mu_improved, sigma_improved = lognorm_params(1.5, 3.5)  # Improved at scale multiplier
improved_multiplier = np.random.lognormal(mu_improved, sigma_improved, n_baseline_samples)
improved_serving_energies = complete_baseline_energies / improved_multiplier

print(f"Improved serving distribution (n={len(improved_serving_energies):,})")
improved_sum_kwh = np.sum(improved_serving_energies) / 1000
print(f"Original improved serving sum: {improved_sum_kwh:.2f} kWh")

# Bootstrap function
def bootstrap_energy_estimate(energies, label, n_bootstrap=1000):
    """
    Estimate uncertainty in total energy consumption using bootstrap sampling.

    The function repeatedly resamples simulated energy-per-query values
    to estimate the distribution of total energy required for 1 billion
    queries. Percentile-based confidence intervals are then calculated.

    Parameters:
    - energies: simulated energy consumption values (Wh/query)
    - label: description used in printed output
    - n_bootstrap: number of bootstrap samples

    Returns:
    - Median energy estimate (GWh)
    - 5th, 25th, 75th and 95th percentile estimates
    """
    
    print(f"\nBootstrap confidence intervals for {label}:")
    np.random.seed(42)
    
    bootstrap_totals_gwh = []
    scaling_factor = 1_000_000_000 / len(energies)
    
    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(energies, size=len(energies), replace=True)
        bootstrap_sum_kwh = np.sum(bootstrap_sample) / 1000
        bootstrap_scaled_kwh = bootstrap_sum_kwh * scaling_factor
        bootstrap_total_gwh = bootstrap_scaled_kwh / 1_000_000
        bootstrap_totals_gwh.append(bootstrap_total_gwh)
    
    bootstrap_totals_gwh = np.array(bootstrap_totals_gwh)
    
    # Calculate confidence intervals
    ci_5 = np.percentile(bootstrap_totals_gwh, 5)
    ci_25 = np.percentile(bootstrap_totals_gwh, 25)
    ci_75 = np.percentile(bootstrap_totals_gwh, 75)
    ci_95 = np.percentile(bootstrap_totals_gwh, 95)
    bootstrap_median = np.median(bootstrap_totals_gwh)
    
    print(f"Bootstrap median: {bootstrap_median:.2f} GWh")
    print(f"90% CI (P5-P95): {ci_5:.2f} - {ci_95:.2f} GWh")
    print(f"50% CI (Q1-Q3): {ci_25:.2f} - {ci_75:.2f} GWh")
    print(f"Scaling factor: {scaling_factor:.2f}")
    
    return bootstrap_median, ci_5, ci_25, ci_75, ci_95

# Bootstrap both distributions
baseline_median, baseline_ci5, baseline_ci25, baseline_ci75, baseline_ci95 = bootstrap_energy_estimate(
    complete_baseline_energies, "Baseline")

improved_median, improved_ci5, improved_ci25, improved_ci75, improved_ci95 = bootstrap_energy_estimate(
    improved_serving_energies, "Improved Serving")

# Create Figure 4 - Comparison bar plot
plt.style.use('default')
plt.figure(figsize=(12, 6), facecolor='white')

# Create comparison bar plot
categories = ['Baseline Scenario', 'Improved\nScenario']
x_pos = np.arange(len(categories))
width = 0.6

medians = [baseline_median, improved_median]
ci5_errors = [baseline_median - baseline_ci5, improved_median - improved_ci5]
ci95_errors = [baseline_ci95 - baseline_median, improved_ci95 - improved_median]

# Multiply by 2x to account for orchestration, backup, etc.
fuzz_factor = 1.10*1.076*1.12
medians = [median * fuzz_factor for median in medians]
ci5_errors = [ci5 * fuzz_factor for ci5 in ci5_errors]
ci95_errors = [ci95 * fuzz_factor for ci95 in ci95_errors]

# Plot bars with error bars (using P5-P95 as error bars)
plt.bar(x_pos, medians, width, 
        yerr=[ci5_errors, ci95_errors],
        capsize=10, color=['#e74c3c', '#e67e22'], edgecolor='black', linewidth=1)

# Customize the plot
plt.title('Estimated Total Energy Consumption\nfor 1 Billion Queries', fontsize=14, pad=20)
plt.ylabel('Energy (GWh)', fontsize=12)
plt.xticks(x_pos, categories, fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# Change range of y axis to focus on the data
# plt.ylim(0.15, 0.5)

# Add value labels on top of bars
for i, median in enumerate(medians):
    plt.text(x_pos[i], median, f'{median:.2f} GWh', 
             ha='center', va='bottom', fontsize=12)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Print comparison summary
improvement_pct = (1 - improved_median/baseline_median) * 100
print(f"\nSUMMARY COMPARISON:")
print(f"Baseline: {baseline_median:.2f} GWh")
print(f"Improved scenario: {improved_median:.2f} GWh")
print(f"Improvement: {improvement_pct:.1f}% reduction ({baseline_median:.2f} → {improved_median:.2f} GWh)")

# %%
# Note 30W H100 cluster at 80% utilizatio for 24 hours is 0.576 GWh
# 40 W H100 cluster at 80% utilization for 24 hours is 0.768 GWh

# Save to a csv file medians and CI after fuzzing
df = pd.DataFrame({
    'Baseline (GWh)': [baseline_median * fuzz_factor, baseline_ci5 * fuzz_factor, baseline_ci95 * fuzz_factor],
    'Improved (GWh)': [improved_median * fuzz_factor, improved_ci5 * fuzz_factor, improved_ci95 * fuzz_factor]
})

df['model_type'] = 'Non-Reasoning'

df.to_csv('1B_queries_energy_consumption_short.csv', index=False)

#%%

# Calculate total tokens for 1 billion queries using existing sample
print("\n" + "="*60)
print("TOTAL TOKENS FOR 1 BILLION QUERIES (SCALED FROM SAMPLE)")
print("="*60)

print("Calculating total tokens for 1 billion queries using existing sample...")
print(f"Parameters:")
print(f"  Median output tokens per query: {median_tokens}")
print(f"  Fixed input tokens per query: {fixed_input_length}")
print(f"  Sample size: {len(model_token_lengths):,}")

# Use the existing sample and scale up (same approach as energy calculations)
billion_queries = 1_000_000_000
scaling_factor = billion_queries / len(model_token_lengths)

# Calculate totals from existing sample
sample_output_tokens = np.sum(model_token_lengths)
scaled_output_tokens = sample_output_tokens * scaling_factor
total_input_tokens = fixed_input_length * billion_queries
total_tokens = scaled_output_tokens + total_input_tokens

# Print results
print(f"\nRESULTS FOR 1 BILLION QUERIES:")
print("="*50)
print(f"Sample output tokens ({len(model_token_lengths):,} queries): {sample_output_tokens:,}")
print(f"Scaling factor: {scaling_factor:,.0f}")
print(f"Scaled output tokens: {scaled_output_tokens:,.0f}")
print(f"Total input tokens: {total_input_tokens:,}")
print(f"TOTAL TOKENS: {total_tokens:,.0f}")
print()
print(f"Average output tokens per query: {scaled_output_tokens/billion_queries:.1f}")
print(f"Average total tokens per query: {total_tokens/billion_queries:.1f}")

# Additional statistics from existing sample
print(f"\nOutput token distribution statistics (from existing sample):")
print(f"  Min: {np.min(model_token_lengths):,}")
print(f"  Max: {np.max(model_token_lengths):,}")
print(f"  Median: {np.median(model_token_lengths):,}")
print(f"  Mean: {np.mean(model_token_lengths):.1f}")
print(f"  Std: {np.std(model_token_lengths):.1f}")

#%%
