import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.optimize import curve_fit
from admm_rpca import admm_rpca
from iaml_rpca import iaml_rpca
import warnings
warnings.filterwarnings('ignore')

def generate_test_data(m, n, r, sparsity_level=0.1, noise_level=5.0, seed=None):
    """
    Generate synthetic RPCA test data
    
    Args:
        m, n: Matrix dimensions
        r: Rank of low-rank component
        sparsity_level: Fraction of entries that are corrupted
        noise_level: Standard deviation of sparse corruption
        seed: Random seed for reproducibility
        
    Returns:
        D: Observed matrix
        L_true: True low-rank component
        S_true: True sparse component
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate low-rank component
    U = np.random.randn(m, r)
    V = np.random.randn(n, r)
    L_true = U @ V.T
    
    # Generate sparse component
    S_true = np.zeros((m, n))
    num_corrupted = int(sparsity_level * m * n)
    corruption_indices = np.random.choice(m*n, size=num_corrupted, replace=False)
    S_true.flat[corruption_indices] = np.random.randn(num_corrupted) * noise_level
    
    D = L_true + S_true
    
    return D, L_true, S_true

def time_algorithm(algorithm_func, D, **kwargs):
    """
    Time an algorithm and return execution time and convergence info
    
    Args:
        algorithm_func: Function to time (admm_rpca or iaml_rpca)
        D: Input data matrix
        **kwargs: Additional arguments for the algorithm
        
    Returns:
        execution_time: Time taken in seconds
        iterations: Number of iterations to convergence
        converged: Whether the algorithm converged
    """
    start_time = time.time()
    try:
        L, S, info = algorithm_func(D, verbose=False, **kwargs)
        execution_time = time.time() - start_time
        return execution_time, info['iterations'], info['converged']
    except Exception as e:
        print(f"Algorithm failed: {e}")
        return np.nan, np.nan, False

def complexity_models(n, a, b, c=0):
    """
    Different complexity models for curve fitting
    """
    models = {
        'linear': lambda n, a, b: a * n + b,
        'quadratic': lambda n, a, b, c: a * n**2 + b * n + c,
        'cubic': lambda n, a, b, c: a * n**3 + b * n + c,
        'n_log_n': lambda n, a, b: a * n * np.log(n) + b,
        'n_squared': lambda n, a, b: a * n**2 + b,
        'n_cubed': lambda n, a, b: a * n**3 + b
    }
    return models

def fit_complexity_curve(sizes, times, model_name='quadratic'):
    """
    Fit a complexity curve to the timing data
    
    Args:
        sizes: Array of problem sizes
        times: Array of execution times
        model_name: Name of the complexity model to fit
        
    Returns:
        fitted_params: Parameters of the fitted curve
        r_squared: R-squared value of the fit
        fitted_curve: Function for the fitted curve
    """
    models = complexity_models(sizes, 0, 0, 0)
    
    if model_name not in models:
        model_name = 'quadratic'
    
    model_func = models[model_name]
    
    try:
        # Filter out NaN values
        valid_mask = ~(np.isnan(sizes) | np.isnan(times))
        sizes_clean = sizes[valid_mask]
        times_clean = times[valid_mask]
        
        if len(sizes_clean) < 3:
            return None, 0, None
        
        # Fit the curve
        if model_name in ['linear', 'n_log_n', 'n_squared', 'n_cubed']:
            popt, _ = curve_fit(model_func, sizes_clean, times_clean)
            fitted_params = popt
        else:
            popt, _ = curve_fit(model_func, sizes_clean, times_clean)
            fitted_params = popt
        
        # Calculate R-squared
        y_pred = model_func(sizes_clean, *fitted_params)
        ss_res = np.sum((times_clean - y_pred) ** 2)
        ss_tot = np.sum((times_clean - np.mean(times_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Return fitted function
        fitted_curve = lambda x: model_func(x, *fitted_params)
        
        return fitted_params, r_squared, fitted_curve
        
    except Exception as e:
        print(f"Curve fitting failed for {model_name}: {e}")
        return None, 0, None

def run_scaling_experiment(size_range, rank_ratio=0.05, sparsity_level=0.1, 
                          noise_level=5.0, num_trials=3):
    """
    Run scaling experiment across different problem sizes
    
    Args:
        size_range: List of matrix sizes to test
        rank_ratio: Ratio of rank to matrix size
        sparsity_level: Fraction of corrupted entries
        noise_level: Standard deviation of corruption
        num_trials: Number of trials per size for averaging
        
    Returns:
        results: DataFrame with timing results
    """
    results = []
    
    print("Running time complexity analysis...")
    print(f"Testing sizes: {size_range}")
    print(f"Rank ratio: {rank_ratio}, Sparsity: {sparsity_level:.1%}")
    print(f"Trials per size: {num_trials}")
    print("-" * 60)
    
    for size in size_range:
        print(f"Testing size {size}x{size}...")
        
        # Calculate rank based on size
        rank = max(1, int(size * rank_ratio))
        
        size_results = {
            'size': size,
            'rank': rank,
            'admm_times': [],
            'iaml_times': [],
            'admm_iterations': [],
            'iaml_iterations': [],
            'admm_converged': [],
            'iaml_converged': []
        }
        
        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}...", end=' ')
            
            # Generate test data
            D, L_true, S_true = generate_test_data(
                size, size, rank, sparsity_level, noise_level, 
                seed=42 + trial
            )
            
            # Test ADMM
            admm_time, admm_iter, admm_conv = time_algorithm(
                admm_rpca, D, max_iter=500, tol_primal=1e-5, tol_dual=1e-6
            )
            size_results['admm_times'].append(admm_time)
            size_results['admm_iterations'].append(admm_iter)
            size_results['admm_converged'].append(admm_conv)
            
            # Test IAML
            iaml_time, iaml_iter, iaml_conv = time_algorithm(
                iaml_rpca, D, max_iter=500, tol_primal=1e-5, tol_dual=1e-6
            )
            size_results['iaml_times'].append(iaml_time)
            size_results['iaml_iterations'].append(iaml_iter)
            size_results['iaml_converged'].append(iaml_conv)
            
            print(f"ADMM: {admm_time:.2f}s, IAML: {iaml_time:.2f}s")
        
        # Calculate averages
        result_row = {
            'size': size,
            'rank': rank,
            'admm_time_mean': np.mean(size_results['admm_times']),
            'admm_time_std': np.std(size_results['admm_times']),
            'iaml_time_mean': np.mean(size_results['iaml_times']),
            'iaml_time_std': np.std(size_results['iaml_times']),
            'admm_iter_mean': np.mean(size_results['admm_iterations']),
            'iaml_iter_mean': np.mean(size_results['iaml_iterations']),
            'admm_conv_rate': np.mean(size_results['admm_converged']),
            'iaml_conv_rate': np.mean(size_results['iaml_converged']),
            'speedup_factor': np.mean(size_results['admm_times']) / np.mean(size_results['iaml_times'])
        }
        
        results.append(result_row)
        
        print(f"  Average - ADMM: {result_row['admm_time_mean']:.2f}±{result_row['admm_time_std']:.2f}s, "
              f"IAML: {result_row['iaml_time_mean']:.2f}±{result_row['iaml_time_std']:.2f}s, "
              f"Speedup: {result_row['speedup_factor']:.2f}x")
    
    return pd.DataFrame(results)

def plot_complexity_analysis(results_df, save_path='/home3/tianyi/RobustPCA/result'):
    """
    Create comprehensive plots of the complexity analysis
    
    Args:
        results_df: DataFrame with timing results
        save_path: Directory to save plots
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Extract data
    sizes = results_df['size'].values
    admm_times = results_df['admm_time_mean'].values
    iaml_times = results_df['iaml_time_mean'].values
    admm_stds = results_df['admm_time_std'].values
    iaml_stds = results_df['iaml_time_std'].values
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Time vs Size with error bars and trend lines
    ax1 = axes[0, 0]
    
    # Plot data points with error bars
    ax1.errorbar(sizes, admm_times, yerr=admm_stds, fmt='bo-', label='ADMM', 
                capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.errorbar(sizes, iaml_times, yerr=iaml_stds, fmt='ro-', label='IAML', 
                capsize=5, capthick=2, linewidth=2, markersize=8)
    
    # Fit and plot trend lines
    models_to_try = ['quadratic', 'cubic', 'n_squared']
    best_admm_model = None
    best_iaml_model = None
    best_admm_r2 = -1
    best_iaml_r2 = -1
    
    for model_name in models_to_try:
        # ADMM trend line
        params, r2, curve_func = fit_complexity_curve(sizes, admm_times, model_name)
        if r2 > best_admm_r2 and params is not None:
            best_admm_r2 = r2
            best_admm_model = (model_name, params, curve_func)
        
        # IAML trend line
        params, r2, curve_func = fit_complexity_curve(sizes, iaml_times, model_name)
        if r2 > best_iaml_r2 and params is not None:
            best_iaml_r2 = r2
            best_iaml_model = (model_name, params, curve_func)
    
    # Plot best trend lines
    x_smooth = np.linspace(sizes.min(), sizes.max(), 100)
    
    if best_admm_model:
        model_name, params, curve_func = best_admm_model
        y_smooth = curve_func(x_smooth)
        ax1.plot(x_smooth, y_smooth, 'b--', alpha=0.7, linewidth=2,
                label=f'ADMM trend ({model_name}, R²={best_admm_r2:.3f})')
    
    if best_iaml_model:
        model_name, params, curve_func = best_iaml_model
        y_smooth = curve_func(x_smooth)
        ax1.plot(x_smooth, y_smooth, 'r--', alpha=0.7, linewidth=2,
                label=f'IAML trend ({model_name}, R²={best_iaml_r2:.3f})')
    
    ax1.set_xlabel('Matrix Size (n×n)', fontsize=16)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=16)
    ax1.set_title('Time Complexity Analysis', fontsize=18, fontweight='bold')
    ax1.legend(fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # Plot 2: Speedup factor
    ax2 = axes[0, 1]
    speedup = results_df['speedup_factor'].values
    ax2.plot(sizes, speedup, 'go-', linewidth=2, markersize=8, label='IAML Speedup')
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No speedup')
    ax2.set_xlabel('Matrix Size (n×n)', fontsize=16)
    ax2.set_ylabel('Speedup Factor (ADMM time / IAML time)', fontsize=16)
    ax2.set_title('IAML Speedup over ADMM', fontsize=18, fontweight='bold')
    ax2.legend(fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Plot 3: Iterations comparison
    ax3 = axes[1, 0]
    admm_iters = results_df['admm_iter_mean'].values
    iaml_iters = results_df['iaml_iter_mean'].values
    
    ax3.plot(sizes, admm_iters, 'bo-', linewidth=2, markersize=8, label='ADMM')
    ax3.plot(sizes, iaml_iters, 'ro-', linewidth=2, markersize=8, label='IAML')
    ax3.set_xlabel('Matrix Size (n×n)', fontsize=16)
    ax3.set_ylabel('Iterations to Convergence', fontsize=16)
    ax3.set_title('Convergence Iterations Comparison', fontsize=18, fontweight='bold')
    ax3.legend(fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # Plot 4: Time per iteration
    ax4 = axes[1, 1]
    admm_time_per_iter = admm_times / admm_iters
    iaml_time_per_iter = iaml_times / iaml_iters
    
    ax4.plot(sizes, admm_time_per_iter, 'bo-', linewidth=2, markersize=8, label='ADMM')
    ax4.plot(sizes, iaml_time_per_iter, 'ro-', linewidth=2, markersize=8, label='IAML')
    ax4.set_xlabel('Matrix Size (n×n)', fontsize=16)
    ax4.set_ylabel('Time per Iteration (seconds)', fontsize=16)
    ax4.set_title('Computational Cost per Iteration', fontsize=18, fontweight='bold')
    ax4.legend(fontsize=16)
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/time_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a separate detailed complexity plot
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot with different scales to show theoretical complexity
    ax.loglog(sizes, admm_times, 'bo-', linewidth=3, markersize=10, label='ADMM (Observed)')
    ax.loglog(sizes, iaml_times, 'ro-', linewidth=3, markersize=10, label='IAML (Observed)')
    
    # Add theoretical complexity lines for reference
    n_min, n_max = sizes.min(), sizes.max()
    n_theory = np.logspace(np.log10(n_min), np.log10(n_max), 50)
    
    # Normalize theoretical curves to match data scale
    admm_scale = admm_times[0] / (n_min**2.5)
    iaml_scale = iaml_times[0] / (n_min**2.5)
    
    ax.loglog(n_theory, admm_scale * n_theory**2.5, 'b--', alpha=0.6, linewidth=2, 
             label='O(n^2.5) reference')
    ax.loglog(n_theory, iaml_scale * n_theory**2.5, 'r--', alpha=0.6, linewidth=2, 
             label='O(n^2.5) reference')
    
    # Add trend line equations as text
    if best_admm_model:
        model_name, params, _ = best_admm_model
        ax.text(0.05, 0.95, f'ADMM best fit: {model_name} (R²={best_admm_r2:.3f})', 
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    if best_iaml_model:
        model_name, params, _ = best_iaml_model
        ax.text(0.05, 0.85, f'IAML best fit: {model_name} (R²={best_iaml_r2:.3f})', 
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax.set_xlabel('Matrix Size (n×n)', fontsize=14)
    ax.set_ylabel('Execution Time (seconds)', fontsize=14)
    ax.set_title('Time Complexity Analysis with Theoretical References', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/detailed_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_complexity_summary(results_df):
    """
    Print a summary of the complexity analysis
    """
    print("\n" + "="*80)
    print("TIME COMPLEXITY ANALYSIS SUMMARY")
    print("="*80)
    
    sizes = results_df['size'].values
    admm_times = results_df['admm_time_mean'].values
    iaml_times = results_df['iaml_time_mean'].values
    
    print(f"{'Size':<8} {'ADMM Time':<12} {'IAML Time':<12} {'Speedup':<10} {'ADMM Iter':<12} {'IAML Iter':<12}")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        print(f"{int(row['size']):<8} {row['admm_time_mean']:<12.3f} {row['iaml_time_mean']:<12.3f} "
              f"{row['speedup_factor']:<10.2f} {row['admm_iter_mean']:<12.1f} {row['iaml_iter_mean']:<12.1f}")
    
    print("-" * 70)
    print(f"Average speedup: {results_df['speedup_factor'].mean():.2f}x")
    print(f"Max speedup: {results_df['speedup_factor'].max():.2f}x")
    print(f"Min speedup: {results_df['speedup_factor'].min():.2f}x")
    
    # Fit complexity curves for summary
    models_to_try = ['linear', 'quadratic', 'cubic', 'n_squared']
    
    print(f"\nComplexity Curve Fitting Results:")
    print(f"{'Algorithm':<10} {'Best Model':<12} {'R-squared':<12} {'Parameters'}")
    print("-" * 60)
    
    for alg_name, times in [('ADMM', admm_times), ('IAML', iaml_times)]:
        best_r2 = -1
        best_model_info = None
        
        for model_name in models_to_try:
            params, r2, _ = fit_complexity_curve(sizes, times, model_name)
            if r2 > best_r2 and params is not None:
                best_r2 = r2
                best_model_info = (model_name, params)
        
        if best_model_info:
            model_name, params = best_model_info
            params_str = ', '.join([f'{p:.2e}' for p in params])
            print(f"{alg_name:<10} {model_name:<12} {best_r2:<12.4f} {params_str}")
    
    print("="*80)

def save_results_to_csv(results_df, save_path='/home3/tianyi/RobustPCA/result'):
    """
    Save results to CSV file
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    csv_path = f'{save_path}/time_complexity_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

def main():
    """
    Main function to run the complete time complexity analysis
    """
    print("RPCA Time Complexity Analysis")
    print("="*50)
    
    # Define test parameters
    size_range = [20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]  # Matrix sizes to test
    rank_ratio = 0.05  # Rank as fraction of matrix size
    sparsity_level = 0.1  # 10% sparse corruption
    noise_level = 5.0  # Standard deviation of corruption
    num_trials = 10  # Number of trials per size for averaging
    
    print(f"Test configuration:")
    print(f"  Matrix sizes: {size_range}")
    print(f"  Rank ratio: {rank_ratio}")
    print(f"  Sparsity level: {sparsity_level:.1%}")
    print(f"  Noise level: {noise_level}")
    print(f"  Trials per size: {num_trials}")
    
    # Run the scaling experiment
    results_df = run_scaling_experiment(
        size_range=size_range,
        rank_ratio=rank_ratio,
        sparsity_level=sparsity_level,
        noise_level=noise_level,
        num_trials=num_trials
    )
    
    # Print summary
    print_complexity_summary(results_df)
    
    # Create plots
    plot_complexity_analysis(results_df)
    
    # Save results
    save_results_to_csv(results_df)
    
    print("\nTime complexity analysis completed!")
    print("Check the generated plots and CSV file for detailed results.")

if __name__ == "__main__":
    main() 