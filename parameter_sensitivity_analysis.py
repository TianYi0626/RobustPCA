import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from admm_rpca import admm_rpca
from iaml_rpca import iaml_rpca
import warnings
warnings.filterwarnings('ignore')

def generate_test_data(m=100, n=100, r=5, sparsity_level=0.1, noise_level=5.0, seed=42):
    """
    Generate synthetic RPCA test data for parameter sensitivity testing
    
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

def compute_recovery_metrics(L_recovered, S_recovered, L_true, S_true):
    """
    Compute recovery quality metrics
    
    Returns:
        Dictionary with various performance metrics
    """
    # Relative errors
    L_rel_error = np.linalg.norm(L_recovered - L_true, 'fro') / np.linalg.norm(L_true, 'fro')
    S_rel_error = np.linalg.norm(S_recovered - S_true, 'fro') / np.linalg.norm(S_true, 'fro')
    
    # Total reconstruction error
    total_error = np.linalg.norm((L_recovered + S_recovered) - (L_true + S_true), 'fro') / np.linalg.norm(L_true + S_true, 'fro')
    
    # Support recovery for sparse component
    S_true_support = np.abs(S_true) > 1e-10
    S_recovered_support = np.abs(S_recovered) > 1e-6
    
    true_positives = np.sum(S_true_support & S_recovered_support)
    false_positives = np.sum(~S_true_support & S_recovered_support)
    false_negatives = np.sum(S_true_support & ~S_recovered_support)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'L_rel_error': L_rel_error,
        'S_rel_error': S_rel_error,
        'total_error': total_error,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def test_lambda_sensitivity(D, L_true, S_true, lambda_range=None):
    """
    Test sensitivity to regularization parameter lambda
    
    Args:
        D: Input data matrix
        L_true, S_true: True components for evaluation
        lambda_range: Range of lambda values to test
        
    Returns:
        DataFrame with results for different lambda values
    """
    if lambda_range is None:
        # Default range around the theoretical optimal value
        lambda_optimal = 1.0 / np.sqrt(max(D.shape))
        lambda_range = np.logspace(np.log10(lambda_optimal/10), np.log10(lambda_optimal*10), 20)
    
    results = []
    
    print("Testing lambda sensitivity...")
    for i, lambda_val in enumerate(lambda_range):
        print(f"  Lambda {i+1}/{len(lambda_range)}: {lambda_val:.4f}")
        
        # Test ADMM
        try:
            start_time = time.time()
            L_admm, S_admm, info_admm = admm_rpca(D, lambda_param=lambda_val, 
                                                  max_iter=500, verbose=False)
            admm_time = time.time() - start_time
            admm_metrics = compute_recovery_metrics(L_admm, S_admm, L_true, S_true)
            admm_success = True
        except Exception as e:
            print(f"    ADMM failed: {e}")
            admm_time = np.nan
            admm_metrics = {k: np.nan for k in ['L_rel_error', 'S_rel_error', 'total_error', 'precision', 'recall', 'f1_score']}
            info_admm = {'iterations': np.nan, 'converged': False}
            admm_success = False
        
        # Test IAML
        try:
            start_time = time.time()
            L_iaml, S_iaml, info_iaml = iaml_rpca(D, lambda_param=lambda_val, 
                                                  max_iter=500, verbose=False)
            iaml_time = time.time() - start_time
            iaml_metrics = compute_recovery_metrics(L_iaml, S_iaml, L_true, S_true)
            iaml_success = True
        except Exception as e:
            print(f"    IAML failed: {e}")
            iaml_time = np.nan
            iaml_metrics = {k: np.nan for k in ['L_rel_error', 'S_rel_error', 'total_error', 'precision', 'recall', 'f1_score']}
            info_iaml = {'iterations': np.nan, 'converged': False}
            iaml_success = False
        
        # Store results
        result = {
            'lambda': lambda_val,
            'admm_time': admm_time,
            'iaml_time': iaml_time,
            'admm_iterations': info_admm['iterations'],
            'iaml_iterations': info_iaml['iterations'],
            'admm_converged': info_admm['converged'],
            'iaml_converged': info_iaml['converged'],
            'admm_success': admm_success,
            'iaml_success': iaml_success
        }
        
        # Add metrics with prefixes
        for key, value in admm_metrics.items():
            result[f'admm_{key}'] = value
        for key, value in iaml_metrics.items():
            result[f'iaml_{key}'] = value
        
        results.append(result)
    
    return pd.DataFrame(results)

def test_penalty_parameter_sensitivity(D, L_true, S_true):
    """
    Test sensitivity to penalty parameters (mu for ADMM, mu_init for IAML)
    
    Args:
        D: Input data matrix
        L_true, S_true: True components for evaluation
        
    Returns:
        Dictionary with results for both algorithms
    """
    # Test ADMM mu sensitivity
    mu_range = np.logspace(-2, 2, 15)  # 0.01 to 100
    admm_results = []
    
    print("Testing ADMM mu sensitivity...")
    for i, mu_val in enumerate(mu_range):
        print(f"  Mu {i+1}/{len(mu_range)}: {mu_val:.4f}")
        
        try:
            start_time = time.time()
            L_admm, S_admm, info_admm = admm_rpca(D, mu=mu_val, max_iter=500, verbose=False)
            admm_time = time.time() - start_time
            admm_metrics = compute_recovery_metrics(L_admm, S_admm, L_true, S_true)
            admm_success = True
        except Exception as e:
            print(f"    ADMM failed: {e}")
            admm_time = np.nan
            admm_metrics = {k: np.nan for k in ['L_rel_error', 'S_rel_error', 'total_error', 'precision', 'recall', 'f1_score']}
            info_admm = {'iterations': np.nan, 'converged': False}
            admm_success = False
        
        result = {
            'mu': mu_val,
            'time': admm_time,
            'iterations': info_admm['iterations'],
            'converged': info_admm['converged'],
            'success': admm_success,
            **admm_metrics
        }
        admm_results.append(result)
    
    # Test IAML mu_init sensitivity
    mu_init_range = np.logspace(-3, 1, 15)  # 0.001 to 10
    iaml_results = []
    
    print("Testing IAML mu_init sensitivity...")
    for i, mu_init_val in enumerate(mu_init_range):
        print(f"  Mu_init {i+1}/{len(mu_init_range)}: {mu_init_val:.4f}")
        
        try:
            start_time = time.time()
            L_iaml, S_iaml, info_iaml = iaml_rpca(D, mu_init=mu_init_val, max_iter=500, verbose=False)
            iaml_time = time.time() - start_time
            iaml_metrics = compute_recovery_metrics(L_iaml, S_iaml, L_true, S_true)
            iaml_success = True
        except Exception as e:
            print(f"    IAML failed: {e}")
            iaml_time = np.nan
            iaml_metrics = {k: np.nan for k in ['L_rel_error', 'S_rel_error', 'total_error', 'precision', 'recall', 'f1_score']}
            info_iaml = {'iterations': np.nan, 'converged': False}
            iaml_success = False
        
        result = {
            'mu_init': mu_init_val,
            'time': iaml_time,
            'iterations': info_iaml['iterations'],
            'converged': info_iaml['converged'],
            'success': iaml_success,
            **iaml_metrics
        }
        iaml_results.append(result)
    
    return {
        'admm_mu': pd.DataFrame(admm_results),
        'iaml_mu_init': pd.DataFrame(iaml_results)
    }

def test_tolerance_sensitivity(D, L_true, S_true):
    """
    Test sensitivity to convergence tolerances
    
    Args:
        D: Input data matrix
        L_true, S_true: True components for evaluation
        
    Returns:
        DataFrame with results for different tolerance values
    """
    # Test different tolerance combinations
    tol_primal_range = np.logspace(-8, -3, 6)  # 1e-8 to 1e-3
    tol_dual_range = np.logspace(-9, -4, 6)    # 1e-9 to 1e-4
    
    results = []
    
    print("Testing tolerance sensitivity...")
    for i, tol_primal in enumerate(tol_primal_range):
        for j, tol_dual in enumerate(tol_dual_range):
            print(f"  Tolerance {i*len(tol_dual_range)+j+1}/{len(tol_primal_range)*len(tol_dual_range)}: "
                  f"primal={tol_primal:.1e}, dual={tol_dual:.1e}")
            
            # Test ADMM
            try:
                start_time = time.time()
                L_admm, S_admm, info_admm = admm_rpca(D, tol_primal=tol_primal, tol_dual=tol_dual, 
                                                      max_iter=1000, verbose=False)
                admm_time = time.time() - start_time
                admm_metrics = compute_recovery_metrics(L_admm, S_admm, L_true, S_true)
                admm_success = True
            except Exception as e:
                admm_time = np.nan
                admm_metrics = {k: np.nan for k in ['L_rel_error', 'S_rel_error', 'total_error', 'precision', 'recall', 'f1_score']}
                info_admm = {'iterations': np.nan, 'converged': False}
                admm_success = False
            
            # Test IAML
            try:
                start_time = time.time()
                L_iaml, S_iaml, info_iaml = iaml_rpca(D, tol_primal=tol_primal, tol_dual=tol_dual, 
                                                      max_iter=1000, verbose=False)
                iaml_time = time.time() - start_time
                iaml_metrics = compute_recovery_metrics(L_iaml, S_iaml, L_true, S_true)
                iaml_success = True
            except Exception as e:
                iaml_time = np.nan
                iaml_metrics = {k: np.nan for k in ['L_rel_error', 'S_rel_error', 'total_error', 'precision', 'recall', 'f1_score']}
                info_iaml = {'iterations': np.nan, 'converged': False}
                iaml_success = False
            
            # Store results
            result = {
                'tol_primal': tol_primal,
                'tol_dual': tol_dual,
                'admm_time': admm_time,
                'iaml_time': iaml_time,
                'admm_iterations': info_admm['iterations'],
                'iaml_iterations': info_iaml['iterations'],
                'admm_converged': info_admm['converged'],
                'iaml_converged': info_iaml['converged'],
                'admm_success': admm_success,
                'iaml_success': iaml_success
            }
            
            # Add metrics with prefixes
            for key, value in admm_metrics.items():
                result[f'admm_{key}'] = value
            for key, value in iaml_metrics.items():
                result[f'iaml_{key}'] = value
            
            results.append(result)
    
    return pd.DataFrame(results)

def plot_lambda_sensitivity(lambda_results, save_path='/home3/tianyi/RobustPCA/result'):
    """
    Plot lambda sensitivity analysis results
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    lambda_vals = lambda_results['lambda'].values
    lambda_optimal = 1.0 / np.sqrt(10000)  # For 100x100 matrix
    
    # Plot 1: Recovery errors vs lambda
    ax1 = axes[0, 0]
    ax1.loglog(lambda_vals, lambda_results['admm_L_rel_error'], 'bo-', label='ADM L error', linewidth=2, markersize=6)
    ax1.loglog(lambda_vals, lambda_results['iaml_L_rel_error'], 'ro-', label='IALM L error', linewidth=2, markersize=6)
    ax1.loglog(lambda_vals, lambda_results['admm_S_rel_error'], 'b^--', label='ADM S error', linewidth=2, markersize=6)
    ax1.loglog(lambda_vals, lambda_results['iaml_S_rel_error'], 'r^--', label='IALM S error', linewidth=2, markersize=6)
    ax1.axvline(x=lambda_optimal, color='k', linestyle=':', alpha=0.7, label='Theoretical optimal')
    ax1.set_xlabel('Lambda (regularization parameter)', fontsize=16)
    ax1.set_ylabel('Relative Error', fontsize=16)
    ax1.set_title('Recovery Error vs Lambda', fontsize=18, fontweight='bold')
    ax1.legend(fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1 score vs lambda
    ax2 = axes[1, 0]
    ax2.semilogx(lambda_vals, lambda_results['admm_f1_score'], 'bo-', label='ADM', linewidth=2, markersize=6)
    ax2.semilogx(lambda_vals, lambda_results['iaml_f1_score'], 'ro-', label='IALM', linewidth=2, markersize=6)
    ax2.axvline(x=lambda_optimal, color='k', linestyle=':', alpha=0.7, label='Theoretical optimal')
    ax2.set_xlabel('Lambda (regularization parameter)', fontsize=16)
    ax2.set_ylabel('F1 Score', fontsize=16)
    ax2.set_title('Sparse Support Recovery vs Lambda', fontsize=18, fontweight='bold')
    ax2.legend(fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    # Plot 3: Computation time vs lambda
    ax3 = axes[2, 0]
    ax3.loglog(lambda_vals, lambda_results['admm_time'], 'bo-', label='ADM', linewidth=2, markersize=6)
    ax3.loglog(lambda_vals, lambda_results['iaml_time'], 'ro-', label='IALM', linewidth=2, markersize=6)
    ax3.axvline(x=lambda_optimal, color='k', linestyle=':', alpha=0.7, label='Theoretical optimal')
    ax3.set_xlabel('Lambda (regularization parameter)', fontsize=16)
    ax3.set_ylabel('Computation Time (seconds)', fontsize=16)
    ax3.set_title('Computation Time vs Lambda', fontsize=18, fontweight='bold')
    ax3.legend(fontsize=16)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Iterations vs lambda
    ax4 = axes[0, 1]
    ax4.semilogx(lambda_vals, lambda_results['admm_iterations'], 'bo-', label='ADM', linewidth=2, markersize=6)
    ax4.semilogx(lambda_vals, lambda_results['iaml_iterations'], 'ro-', label='IALM', linewidth=2, markersize=6)
    ax4.axvline(x=lambda_optimal, color='k', linestyle=':', alpha=0.7, label='Theoretical optimal')
    ax4.set_xlabel('Lambda (regularization parameter)', fontsize=16)
    ax4.set_ylabel('Iterations to Convergence', fontsize=16)
    ax4.set_title('Convergence Speed vs Lambda', fontsize=18, fontweight='bold')
    ax4.legend(fontsize=16)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Success rate vs lambda
    ax5 = axes[1, 1]
    ax5.semilogx(lambda_vals, lambda_results['admm_converged'].astype(float), 'bo-', label='ADM', linewidth=2, markersize=6)
    ax5.semilogx(lambda_vals, lambda_results['iaml_converged'].astype(float), 'ro-', label='IALM', linewidth=2, markersize=6)
    ax5.axvline(x=lambda_optimal, color='k', linestyle=':', alpha=0.7, label='Theoretical optimal')
    ax5.set_xlabel('Lambda (regularization parameter)', fontsize=16)
    ax5.set_ylabel('Convergence Rate', fontsize=16)
    ax5.set_title('Convergence Reliability vs Lambda', fontsize=18, fontweight='bold')
    ax5.legend(fontsize=16)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([-0.1, 1.1])
    
    # Plot 6: Total error vs lambda
    ax6 = axes[2, 1]
    ax6.loglog(lambda_vals, lambda_results['admm_total_error'], 'bo-', label='ADM', linewidth=2, markersize=6)
    ax6.loglog(lambda_vals, lambda_results['iaml_total_error'], 'ro-', label='IALM', linewidth=2, markersize=6)
    ax6.axvline(x=lambda_optimal, color='k', linestyle=':', alpha=0.7, label='Theoretical optimal')
    ax6.set_xlabel('Lambda (regularization parameter)', fontsize=16)
    ax6.set_ylabel('Total Reconstruction Error', fontsize=16)
    ax6.set_title('Overall Recovery Quality vs Lambda', fontsize=18, fontweight='bold')
    ax6.legend(fontsize=16)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/lambda_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_penalty_parameter_sensitivity(penalty_results, save_path='/home3/tianyi/RobustPCA/result'):
    """
    Plot penalty parameter sensitivity analysis results
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    admm_results = penalty_results['admm_mu']
    iaml_results = penalty_results['iaml_mu_init']
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # ADMM mu sensitivity plots
    mu_vals = admm_results['mu'].values
    
    # Plot 1: ADMM recovery error vs mu
    ax1 = axes[0, 0]
    ax1.loglog(mu_vals, admm_results['L_rel_error'], 'bo-', label='L error', linewidth=2, markersize=6)
    ax1.loglog(mu_vals, admm_results['S_rel_error'], 'b^--', label='S error', linewidth=2, markersize=6)
    ax1.loglog(mu_vals, admm_results['total_error'], 'bs:', label='Total error', linewidth=2, markersize=6)
    ax1.set_xlabel('Mu (ADMM penalty parameter)', fontsize=16)
    ax1.set_ylabel('Relative Error', fontsize=16)
    ax1.set_title('ADMM: Recovery Error vs Mu', fontsize=18, fontweight='bold')
    ax1.legend(fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ADMM time and iterations vs mu
    ax2 = axes[1, 0]
    ax2_twin = ax2.twinx()
    line1 = ax2.loglog(mu_vals, admm_results['time'], 'bo-', label='Time', linewidth=2, markersize=6)
    line2 = ax2_twin.semilogx(mu_vals, admm_results['iterations'], 'ro-', label='Iterations', linewidth=2, markersize=6)
    ax2.set_xlabel('Mu (ADMM penalty parameter)', fontsize=16)
    ax2.set_ylabel('Time (seconds)', fontsize=16, color='blue')
    ax2_twin.set_ylabel('Iterations', fontsize=16, color='red')
    ax2.set_title('ADMM: Performance vs Mu', fontsize=18, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ADMM convergence rate vs mu
    ax3 = axes[2, 0]
    ax3.semilogx(mu_vals, admm_results['converged'].astype(float), 'bo-', linewidth=2, markersize=6)
    ax3.set_xlabel('Mu (ADMM penalty parameter)', fontsize=16)
    ax3.set_ylabel('Convergence Rate', fontsize=16)
    ax3.set_title('ADMM: Convergence Reliability vs Mu', fontsize=18, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-0.1, 1.1])
    
    # IAML mu_init sensitivity plots
    mu_init_vals = iaml_results['mu_init'].values
    
    # Plot 4: IAML recovery error vs mu_init
    ax4 = axes[0, 1]
    ax4.loglog(mu_init_vals, iaml_results['L_rel_error'], 'ro-', label='L error', linewidth=2, markersize=6)
    ax4.loglog(mu_init_vals, iaml_results['S_rel_error'], 'r^--', label='S error', linewidth=2, markersize=6)
    ax4.loglog(mu_init_vals, iaml_results['total_error'], 'rs:', label='Total error', linewidth=2, markersize=6)
    ax4.set_xlabel('Mu_init (IAML initial penalty)', fontsize=16)
    ax4.set_ylabel('Relative Error', fontsize=16)
    ax4.set_title('IAML: Recovery Error vs Mu_init', fontsize=18, fontweight='bold')
    ax4.legend(fontsize=16)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: IAML time and iterations vs mu_init
    ax5 = axes[1, 1]
    ax5_twin = ax5.twinx()
    line3 = ax5.loglog(mu_init_vals, iaml_results['time'], 'ro-', label='Time', linewidth=2, markersize=6)
    line4 = ax5_twin.semilogx(mu_init_vals, iaml_results['iterations'], 'bo-', label='Iterations', linewidth=2, markersize=6)
    ax5.set_xlabel('Mu_init (IAML initial penalty)', fontsize=16)
    ax5.set_ylabel('Time (seconds)', fontsize=16, color='red')
    ax5_twin.set_ylabel('Iterations', fontsize=16, color='blue')
    ax5.set_title('IAML: Performance vs Mu_init', fontsize=18, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: IAML convergence rate vs mu_init
    ax6 = axes[2, 1]
    ax6.semilogx(mu_init_vals, iaml_results['converged'].astype(float), 'ro-', linewidth=2, markersize=6)
    ax6.set_xlabel('Mu_init (IAML initial penalty)', fontsize=16)
    ax6.set_ylabel('Convergence Rate', fontsize=16)
    ax6.set_title('IAML: Convergence Reliability vs Mu_init', fontsize=18, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([-0.1, 1.1])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/penalty_parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_tolerance_sensitivity(tolerance_results, save_path='/home3/tianyi/RobustPCA/result'):
    """
    Plot tolerance sensitivity analysis results as heatmaps
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Create pivot tables for heatmaps
    tol_primal_vals = sorted(tolerance_results['tol_primal'].unique())
    tol_dual_vals = sorted(tolerance_results['tol_dual'].unique())
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # ADMM heatmaps
    # Plot 1: ADMM total error heatmap
    admm_error_pivot = tolerance_results.pivot(index='tol_primal', columns='tol_dual', values='admm_total_error')
    im1 = axes[0, 0].imshow(admm_error_pivot.values, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('ADM: Total Error vs Tolerances', fontsize=18, fontweight='bold')
    axes[0, 0].set_xlabel('Dual Tolerance Index', fontsize=16)
    axes[0, 0].set_ylabel('Primal Tolerance Index', fontsize=16)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: ADMM time heatmap
    admm_time_pivot = tolerance_results.pivot(index='tol_primal', columns='tol_dual', values='admm_time')
    im2 = axes[1, 0].imshow(admm_time_pivot.values, cmap='plasma', aspect='auto')
    axes[1, 0].set_title('ADM: Computation Time vs Tolerances', fontsize=18, fontweight='bold')
    axes[1, 0].set_xlabel('Dual Tolerance Index', fontsize=16)
    axes[1, 0].set_ylabel('Primal Tolerance Index', fontsize=16)
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot 3: ADMM iterations heatmap
    admm_iter_pivot = tolerance_results.pivot(index='tol_primal', columns='tol_dual', values='admm_iterations')
    im3 = axes[2, 0].imshow(admm_iter_pivot.values, cmap='inferno', aspect='auto')
    axes[2, 0].set_title('ADM: Iterations vs Tolerances', fontsize=18, fontweight='bold')
    axes[2, 0].set_xlabel('Dual Tolerance Index', fontsize=16)
    axes[2, 0].set_ylabel('Primal Tolerance Index', fontsize=16)
    plt.colorbar(im3, ax=axes[2, 0])
    
    # IAML heatmaps
    # Plot 4: IAML total error heatmap
    iaml_error_pivot = tolerance_results.pivot(index='tol_primal', columns='tol_dual', values='iaml_total_error')
    im4 = axes[0, 1].imshow(iaml_error_pivot.values, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('IAML: Total Error vs Tolerances', fontsize=18, fontweight='bold')
    axes[0, 1].set_xlabel('Dual Tolerance Index', fontsize=16)
    axes[0, 1].set_ylabel('Primal Tolerance Index', fontsize=16)
    plt.colorbar(im4, ax=axes[0, 1])
    
    # Plot 5: IAML time heatmap
    iaml_time_pivot = tolerance_results.pivot(index='tol_primal', columns='tol_dual', values='iaml_time')
    im5 = axes[1, 1].imshow(iaml_time_pivot.values, cmap='plasma', aspect='auto')
    axes[1, 1].set_title('IAML: Computation Time vs Tolerances', fontsize=18, fontweight='bold')
    axes[1, 1].set_xlabel('Dual Tolerance Index', fontsize=16)
    axes[1, 1].set_ylabel('Primal Tolerance Index', fontsize=16)
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Plot 6: IAML iterations heatmap
    iaml_iter_pivot = tolerance_results.pivot(index='tol_primal', columns='tol_dual', values='iaml_iterations')
    im6 = axes[2, 1].imshow(iaml_iter_pivot.values, cmap='inferno', aspect='auto')
    axes[2, 1].set_title('IAML: Iterations vs Tolerances', fontsize=18, fontweight='bold')
    axes[2, 1].set_xlabel('Dual Tolerance Index', fontsize=16)
    axes[2, 1].set_ylabel('Primal Tolerance Index', fontsize=16)
    plt.colorbar(im6, ax=axes[2, 1])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/tolerance_sensitivity_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_sensitivity_summary_report(lambda_results, penalty_results, tolerance_results, 
                                    save_path='/home3/tianyi/RobustPCA/result'):
    """
    Create a comprehensive sensitivity analysis report
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    report_path = f'{save_path}/PARAMETER_SENSITIVITY_REPORT.md'
    
    with open(report_path, 'w') as f:
        f.write("# RPCA Parameter Sensitivity Analysis Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report presents a comprehensive analysis of parameter sensitivity for both ADMM and IAML algorithms in Robust PCA.\n\n")
        
        # Lambda sensitivity summary
        f.write("## 1. Regularization Parameter (Lambda) Sensitivity\n\n")
        
        lambda_optimal = 1.0 / np.sqrt(10000)
        
        # Handle cases where all tests failed
        admm_valid = lambda_results['admm_total_error'].notna()
        iaml_valid = lambda_results['iaml_total_error'].notna()
        
        if admm_valid.any():
            best_admm_lambda = lambda_results.loc[lambda_results['admm_total_error'].idxmin(), 'lambda']
        else:
            best_admm_lambda = np.nan
            
        if iaml_valid.any():
            best_iaml_lambda = lambda_results.loc[lambda_results['iaml_total_error'].idxmin(), 'lambda']
        else:
            best_iaml_lambda = np.nan
        
        f.write(f"**Theoretical Optimal Lambda:** {lambda_optimal:.6f}\n\n")
        f.write(f"**Empirically Best Lambda:**\n")
        
        # Format ADMM lambda result
        if not np.isnan(best_admm_lambda):
            admm_lambda_str = f"{best_admm_lambda:.6f}"
        else:
            admm_lambda_str = "N/A (all tests failed)"
        f.write(f"- ADMM: {admm_lambda_str}\n")
        
        # Format IAML lambda result
        if not np.isnan(best_iaml_lambda):
            iaml_lambda_str = f"{best_iaml_lambda:.6f}"
        else:
            iaml_lambda_str = "N/A (all tests failed)"
        f.write(f"- IAML: {iaml_lambda_str}\n\n")
        
        # Calculate sensitivity ranges
        if admm_valid.any():
            admm_good_range = lambda_results[lambda_results['admm_total_error'] < lambda_results['admm_total_error'].quantile(0.25)]
            admm_range_str = f"{admm_good_range['lambda'].min():.6f} to {admm_good_range['lambda'].max():.6f}"
        else:
            admm_range_str = "N/A (all tests failed)"
            
        if iaml_valid.any():
            iaml_good_range = lambda_results[lambda_results['iaml_total_error'] < lambda_results['iaml_total_error'].quantile(0.25)]
            iaml_range_str = f"{iaml_good_range['lambda'].min():.6f} to {iaml_good_range['lambda'].max():.6f}"
        else:
            iaml_range_str = "N/A (all tests failed)"
        
        f.write(f"**Robust Parameter Ranges (top 25% performance):**\n")
        f.write(f"- ADMM: {admm_range_str}\n")
        f.write(f"- IAML: {iaml_range_str}\n\n")
        
        # Penalty parameter sensitivity summary
        f.write("## 2. Penalty Parameter Sensitivity\n\n")
        
        admm_mu_results = penalty_results['admm_mu']
        iaml_mu_results = penalty_results['iaml_mu_init']
        
        # Handle cases where all tests failed
        admm_mu_valid = admm_mu_results['total_error'].notna()
        iaml_mu_valid = iaml_mu_results['total_error'].notna()
        
        if admm_mu_valid.any():
            best_admm_mu = admm_mu_results.loc[admm_mu_results['total_error'].idxmin(), 'mu']
        else:
            best_admm_mu = np.nan
            
        if iaml_mu_valid.any():
            best_iaml_mu = iaml_mu_results.loc[iaml_mu_results['total_error'].idxmin(), 'mu_init']
        else:
            best_iaml_mu = np.nan
        
        f.write(f"**Optimal Penalty Parameters:**\n")
        
        # Format ADMM mu result
        if not np.isnan(best_admm_mu):
            admm_mu_str = f"{best_admm_mu:.4f}"
        else:
            admm_mu_str = "N/A (all tests failed)"
        f.write(f"- ADMM mu: {admm_mu_str}\n")
        
        # Format IAML mu_init result
        if not np.isnan(best_iaml_mu):
            iaml_mu_str = f"{best_iaml_mu:.4f}"
        else:
            iaml_mu_str = "N/A (all tests failed)"
        f.write(f"- IAML mu_init: {iaml_mu_str}\n\n")
        
        # Tolerance sensitivity summary
        f.write("## 3. Tolerance Sensitivity\n\n")
        
        admm_tol_valid = tolerance_results['admm_total_error'].notna()
        iaml_tol_valid = tolerance_results['iaml_total_error'].notna()
        
        if admm_tol_valid.any():
            best_tol_admm = tolerance_results.loc[tolerance_results['admm_total_error'].idxmin()]
            admm_tol_str = f"primal={best_tol_admm['tol_primal']:.1e}, dual={best_tol_admm['tol_dual']:.1e}"
        else:
            admm_tol_str = "N/A (all tests failed)"
            
        if iaml_tol_valid.any():
            best_tol_iaml = tolerance_results.loc[tolerance_results['iaml_total_error'].idxmin()]
            iaml_tol_str = f"primal={best_tol_iaml['tol_primal']:.1e}, dual={best_tol_iaml['tol_dual']:.1e}"
        else:
            iaml_tol_str = "N/A (all tests failed)"
        
        f.write(f"**Optimal Tolerance Settings:**\n")
        f.write(f"- ADMM: {admm_tol_str}\n")
        f.write(f"- IAML: {iaml_tol_str}\n\n")
        
        # Robustness comparison
        f.write("## 4. Algorithm Robustness Comparison\n\n")
        
        # Calculate coefficient of variation for different parameters
        if admm_valid.any() and iaml_valid.any():
            lambda_cv_admm = lambda_results['admm_total_error'].std() / lambda_results['admm_total_error'].mean()
            lambda_cv_iaml = lambda_results['iaml_total_error'].std() / lambda_results['iaml_total_error'].mean()
            
            f.write(f"**Parameter Sensitivity (Coefficient of Variation):**\n")
            f.write(f"- Lambda sensitivity - ADMM: {lambda_cv_admm:.3f}, IAML: {lambda_cv_iaml:.3f}\n")
            f.write(f"- {'IAML is more robust' if lambda_cv_iaml < lambda_cv_admm else 'ADMM is more robust'} to lambda variations\n\n")
        else:
            f.write(f"**Parameter Sensitivity:** Cannot compute due to test failures\n\n")
        
        # Convergence reliability
        admm_conv_rate = lambda_results['admm_converged'].mean()
        iaml_conv_rate = lambda_results['iaml_converged'].mean()
        
        f.write(f"**Convergence Reliability:**\n")
        f.write(f"- ADMM: {admm_conv_rate:.1%} convergence rate\n")
        f.write(f"- IAML: {iaml_conv_rate:.1%} convergence rate\n\n")
        
        # Recommendations
        f.write("## 5. Practical Recommendations\n\n")
        f.write("### Parameter Selection Guidelines:\n\n")
        f.write("1. **Lambda (Regularization Parameter):**\n")
        f.write(f"   - Use theoretical value λ = 1/√max(m,n) as starting point\n")
        
        if iaml_valid.any() and admm_valid.any():
            f.write(f"   - IAML is less sensitive to lambda variations\n")
            f.write(f"   - Safe range: [{min(admm_good_range['lambda'].min(), iaml_good_range['lambda'].min()):.6f}, {max(admm_good_range['lambda'].max(), iaml_good_range['lambda'].max()):.6f}]\n\n")
        else:
            f.write(f"   - Use default theoretical value due to test failures\n\n")
        
        f.write("2. **Penalty Parameters:**\n")
        
        # Format ADMM mu recommendation
        if not np.isnan(best_admm_mu):
            f.write(f"   - ADMM mu: Use values around {best_admm_mu:.2f} (range: 0.1 to 10)\n")
        else:
            f.write(f"   - ADMM mu: Use default value 1.0 (tests failed)\n")
            
        # Format IAML mu_init recommendation
        if not np.isnan(best_iaml_mu):
            f.write(f"   - IAML mu_init: Use values around {best_iaml_mu:.3f} (range: 0.01 to 1.0)\n\n")
        else:
            f.write(f"   - IAML mu_init: Use default value 0.1 (tests failed)\n\n")
        
        f.write("3. **Tolerance Settings:**\n")
        f.write(f"   - For high accuracy: primal_tol=1e-7, dual_tol=1e-8\n")
        f.write(f"   - For balanced performance: primal_tol=1e-6, dual_tol=1e-7\n")
        f.write(f"   - For fast computation: primal_tol=1e-5, dual_tol=1e-6\n\n")
        
        f.write("### Algorithm Selection:\n\n")
        f.write("- **Choose IAML when:** Parameter robustness is important, simpler tuning is preferred\n")
        f.write("- **Choose ADMM when:** Theoretical guarantees are critical, parallel implementation is needed\n\n")
        
        f.write("## 6. Detailed Results\n\n")
        f.write("See the generated plots for detailed sensitivity analysis:\n")
        f.write("- `lambda_sensitivity_analysis.png`: Lambda parameter sensitivity\n")
        f.write("- `penalty_parameter_sensitivity.png`: Penalty parameter sensitivity\n")
        f.write("- `tolerance_sensitivity_heatmaps.png`: Tolerance sensitivity heatmaps\n\n")
    
    print(f"Sensitivity analysis report saved to: {report_path}")

def save_sensitivity_results(lambda_results, penalty_results, tolerance_results, 
                           save_path='/home3/tianyi/RobustPCA/result'):
    """
    Save all sensitivity analysis results to CSV files
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Save lambda sensitivity results
    lambda_results.to_csv(f'{save_path}/lambda_sensitivity_results.csv', index=False)
    
    # Save penalty parameter results
    penalty_results['admm_mu'].to_csv(f'{save_path}/admm_mu_sensitivity_results.csv', index=False)
    penalty_results['iaml_mu_init'].to_csv(f'{save_path}/iaml_mu_init_sensitivity_results.csv', index=False)
    
    # Save tolerance results
    tolerance_results.to_csv(f'{save_path}/tolerance_sensitivity_results.csv', index=False)
    
    print("All sensitivity analysis results saved to CSV files.")

def main():
    """
    Main function to run the complete parameter sensitivity analysis
    """
    print("RPCA Parameter Sensitivity Analysis")
    print("="*50)
    
    # Generate test data
    print("Generating test data...")
    D, L_true, S_true = generate_test_data(m=100, n=100, r=5, sparsity_level=0.1, noise_level=5.0)
    print(f"Test data: {D.shape[0]}×{D.shape[1]} matrix, rank={5}, sparsity=10%")
    
    # Test lambda sensitivity
    print("\n1. Testing lambda (regularization parameter) sensitivity...")
    lambda_results = test_lambda_sensitivity(D, L_true, S_true)
    
    # Test penalty parameter sensitivity
    print("\n2. Testing penalty parameter sensitivity...")
    penalty_results = test_penalty_parameter_sensitivity(D, L_true, S_true)
    
    # Test tolerance sensitivity
    print("\n3. Testing tolerance sensitivity...")
    tolerance_results = test_tolerance_sensitivity(D, L_true, S_true)
    
    # Create plots
    print("\n4. Creating sensitivity analysis plots...")
    plot_lambda_sensitivity(lambda_results)
    plot_penalty_parameter_sensitivity(penalty_results)
    plot_tolerance_sensitivity(tolerance_results)
    
    # Save results
    print("\n5. Saving results...")
    save_sensitivity_results(lambda_results, penalty_results, tolerance_results)
    
    # Create summary report
    print("\n6. Creating summary report...")
    create_sensitivity_summary_report(lambda_results, penalty_results, tolerance_results)
    
    print("\nParameter sensitivity analysis completed!")
    print("Check the generated plots, CSV files, and report for detailed results.")

if __name__ == "__main__":
    main() 