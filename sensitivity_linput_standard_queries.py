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
sns.set_context("poster", font_scale=2.5)
sns.set_style("white")

# Seed for reproducibility
np.random.seed(42)

# Load model_throughput_consolidated_DB.csv
model_throughput = pd.read_csv('model_throughput_DB.csv')
print(model_throughput.columns)

# Drop those with Quantization != "FP8"
model_throughput = model_throughput[model_throughput['Quantization'] == 'FP8']

#%%
# Helper function to get log-normal parameters from (5th, 95th percentile)
def lognorm_params(min_val, max_val):
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

def run_analysis_for_input_length(fixed_input_length, tps_regression_models, interpolation_models, max_tps_values):
    """Run the energy analysis for a specific input length"""
    
    # Simulation settings
    n_runs = 10000
    median_tokens = 300   # median tokens query length
    # Calculate lambda parameter for exponential distribution to achieve desired median
    lambda_param = np.log(2) / median_tokens  # For exponential, median = ln(2)/λ
    
    # Define ranges and values
    def get_node_power(model_name):
        return 12.8 if model_name == 'DeepSeek-R1' else 10.2
    
    pu_range = (0.4, 0.9)        # for lognormal
    PUE_range = (1.05, 1.6)       # for lognormal
    
    # Compute log-normal parameters where needed
    mu_pu, sigma_pu = lognorm_params(*pu_range)
    mu_pue, sigma_pue = lognorm_params(*PUE_range)
    
    # Create separate distributions for each model using regression
    all_model_energies = {}
    all_model_tps = {}  # Add this to store TPS predictions
    print(f"\nGenerating Energy Distributions for input length {fixed_input_length}:")
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
            energy_kj = model_pue[i] * (model_node_power_array[i] * model_pu[i] * ((model_token_lengths[i]) + fixed_input_length/30)) / model_tokens_per_sec[i]
            model_energies[i] = (energy_kj / 3600) * 1000
        
        all_model_energies[model_name] = model_energies
        print(f"  Generated {n_runs} energy samples")
    
    # Rename the key in all_model_energies and all_model_tps to match the renamed model label
    if 'DeepSeek-R1' in all_model_energies:
        all_model_energies['DeepSeek-R1 671B'] = all_model_energies.pop('DeepSeek-R1')
    if 'DeepSeek-R1' in all_model_tps:
        all_model_tps['DeepSeek-R1 671B'] = all_model_tps.pop('DeepSeek-R1')
    
    # Define the desired order of models
    model_order = [
        'DeepSeek-R1 671B',
        'Llama 3.1 405B',
        'Llama-3.1 Nemotron Ultra 253B',
        'Mixtral 8x22B',
        'Llama 3.1 70B'
    ]
    
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
    mu_improved, sigma_improved = lognorm_params(1.5, 5)  # Improved serving multiplier
    
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
    
    return mixed_energies, algorithm_energies, improved_energies, hardware_energies, fixed_input_length

# Build regression models once
tps_regression_models, interpolation_models, max_tps_values = create_tps_regression_models(model_throughput)

# Run analysis for different input lengths
input_lengths = [500, 1000, 5000]
all_results = {}

for fixed_input_length in input_lengths:
    results = run_analysis_for_input_length(fixed_input_length, tps_regression_models, interpolation_models, max_tps_values)
    all_results[fixed_input_length] = results

# Create combined subplot figure
print("\n" + "="*60)
print("CREATING COMBINED SENSITIVITY ANALYSIS FIGURE")
print("="*60)

# Create 1x3 subplot figure
# plt.style.use('default')
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('Energy Distribution with Line-of-Sight Improvements: Input Length Sensitivity (P5-P95) Baseline: Blend of Models >200B parameters', 
             fontsize=16, y=0.95)

# Define colors for each category
colors = {
    'Baseline': '#2ecc71',
    'Model': '#9b59b6',
    'Serving \nPlatform': '#e67e22',
    'Hardware \n& Datacenter': '#3498db'
}

def prepare_improvement_data(energies, category):
    """Apply 5-95 percentile filtering to improvement categories to remove outliers created by multipliers"""
    if category != 'Baseline':
        # Apply same 5-95 percentile filtering as used in Figure 1
        p5, p95 = np.percentile(energies, [5, 95])
        filtered_energies = energies[(energies >= p5) & (energies <= p95)]
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

for idx, (fixed_input_length, (mixed_energies, algorithm_energies, improved_energies, hardware_energies, _)) in enumerate(all_results.items()):
    ax = axes[idx]
    
    # Combine all distributions with updated names and order
    plot_data_all = pd.concat([
        prepare_improvement_data(mixed_energies, 'Baseline'),
        prepare_improvement_data(algorithm_energies, 'Model'),
        prepare_improvement_data(improved_energies, 'Serving \nPlatform'),
        prepare_improvement_data(hardware_energies, 'Hardware \n& Datacenter')
    ])
    
    # Create violin plot
    sns.violinplot(data=plot_data_all, 
                   x='Energy (Wh)', 
                   y='Distribution',
                   orient='h',
                   inner=None,
                   cut=0,
                   width=0.9,
                   density_norm='width',
                   palette=colors,
                   bw_adjust=1,
                   ax=ax)
    
    # Add quartile lines manually for each violin
    distribution_names = ['Baseline', 'Model', 'Serving \nPlatform', 'Hardware \n& Datacenter']
    energies_list = [mixed_energies, algorithm_energies, improved_energies, hardware_energies]
    
    for i, (dist_name, energies) in enumerate(zip(distribution_names, energies_list)):
        # Get quartiles from the actual data being plotted
        p25, p50, p75 = np.percentile(energies, [25, 50, 75])
        # Add lines at each quartile
        ax.hlines(y=i, xmin=p25, xmax=p75, color='black', linewidth=2, alpha=0.7)  # IQR line
        ax.vlines(x=p50, ymin=i-0.1, ymax=i+0.1, color='white', linewidth=3)  # Median line
        ax.vlines(x=p50, ymin=i-0.1, ymax=i+0.1, color='black', linewidth=2)  # Median line border
    
    # Remove any whisker lines that might remain
    for artist in ax.get_children():
        if isinstance(artist, matplotlib.lines.Line2D):
            if artist.get_linestyle() == '--':
                artist.set_visible(False)
    
    # Customize subplot
    ax.set_title(f'Input Length: {fixed_input_length} tokens', fontsize=14, pad=15)
    ax.set_xlabel('Energy per Query (Wh)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend to each subplot
    legend_elements = []
    for category, energies in [
        ('Baseline', mixed_energies),
        ('Model', algorithm_energies),
        ('Serving \nPlatform', improved_energies),
        ('Hardware \n& Datacenter', hardware_energies)
    ]:
        p25, p50, p75 = np.percentile(energies, [25, 50, 75])
        legend_elements.append(plt.Line2D([0], [0], color=colors[category], 
                             label=f'{category.replace(chr(10), " ")}: {p50:.2f} Wh', 
                             linewidth=2))
    
    ax.legend(handles=legend_elements, frameon=True, facecolor='white', 
             edgecolor='none', loc='lower right', bbox_to_anchor=(1, 0),
             fontsize=10)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.88)

# Create manuscript_figures directory if it doesn't exist
import os
os.makedirs('manuscript_figures', exist_ok=True)

# Save the combined figure
plt.savefig('manuscript_figures/updated_figures/figure2_energy_sens.svg', format='svg', dpi=300, bbox_inches='tight')
plt.savefig('manuscript_figures/updated_figures/figure2_energy_sens.png', format='png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Print statistics for all input lengths
print(f"\nSUMMARY STATISTICS FOR ALL INPUT LENGTHS:")
print("="*80)

for fixed_input_length, (mixed_energies, algorithm_energies, improved_energies, hardware_energies, _) in all_results.items():
    print(f"\nInput Length: {fixed_input_length} tokens")
    print("-" * 40)
    
    for category, energies in [
        ('Baseline (Mixed Top 3)', mixed_energies),
        ('Model', algorithm_energies),
        ('Serving Platform', improved_energies),
        ('Hardware & Datacenter', hardware_energies)
    ]:
        p5, p25, p50, p75, p95 = np.percentile(energies, [5, 25, 50, 75, 95])
        mean = np.mean(energies)
        
        print(f"{category}:")
        print(f"  Median: {p50:.3f} Wh (IQR: {p25:.3f}-{p75:.3f} Wh)")
        print(f"  Mean: {mean:.3f} Wh")
    
    # Print improvement percentages relative to baseline
    baseline_median = np.median(mixed_energies)
    print(f"\nImprovement vs Baseline:")
    for category, energies in [
        ('Model', algorithm_energies),
        ('Serving Platform', improved_energies),
        ('Hardware & Datacenter', hardware_energies)
    ]:
        improved_median = np.median(energies)
        improvement_pct = (1 - improved_median/baseline_median) * 100
        print(f"  {category}: {improvement_pct:.1f}% reduction")

# %%
