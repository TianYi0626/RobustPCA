import numpy as np
import matplotlib.pyplot as plt
import time
from admm_rpca import admm_rpca
from iaml_rpca import iaml_rpca

def generate_synthetic_data(m, n, r, sparsity_level=0.1, noise_level=5.0, seed=42):
    """
    Generate synthetic RPCA data
    
    Args:
        m, n: Matrix dimensions
        r: Rank of low-rank component
        sparsity_level: Fraction of entries that are corrupted
        noise_level: Standard deviation of sparse corruption
        seed: Random seed
        
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

def compute_metrics(L_recovered, S_recovered, L_true, S_true):
    """
    Compute performance metrics
    
    Returns:
        Dictionary with various metrics
    """
    # Relative errors
    L_rel_error = np.linalg.norm(L_recovered - L_true, 'fro') / np.linalg.norm(L_true, 'fro')
    S_rel_error = np.linalg.norm(S_recovered - S_true, 'fro') / np.linalg.norm(S_true, 'fro')
    
    # Support recovery for sparse component
    S_true_support = np.abs(S_true) > 1e-10
    S_recovered_support = np.abs(S_recovered) > 1e-6
    
    true_positives = np.sum(S_true_support & S_recovered_support)
    false_positives = np.sum(~S_true_support & S_recovered_support)
    false_negatives = np.sum(S_true_support & ~S_recovered_support)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    support_recovery = true_positives / np.sum(S_true_support) if np.sum(S_true_support) > 0 else 0
    
    return {
        'L_rel_error': L_rel_error,
        'S_rel_error': S_rel_error,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'support_recovery': support_recovery
    }

def compare_algorithms(D, L_true, S_true, lambda_param=None):
    """
    Compare ADMM and IAML algorithms on the same data
    
    Returns:
        Dictionary with results from both algorithms
    """
    print("Running ADMM-RPCA...")
    start_time = time.time()
    L_admm, S_admm, info_admm = admm_rpca(D, lambda_param=lambda_param, verbose=True)
    admm_time = time.time() - start_time
    
    print("\nRunning IAML-RPCA...")
    start_time = time.time()
    L_iaml, S_iaml, info_iaml = iaml_rpca(D, lambda_param=lambda_param, verbose=True)
    iaml_time = time.time() - start_time
    
    # Compute metrics
    admm_metrics = compute_metrics(L_admm, S_admm, L_true, S_true)
    iaml_metrics = compute_metrics(L_iaml, S_iaml, L_true, S_true)
    
    results = {
        'ADMM': {
            'L': L_admm,
            'S': S_admm,
            'info': info_admm,
            'metrics': admm_metrics,
            'time': admm_time
        },
        'IAML': {
            'L': L_iaml,
            'S': S_iaml,
            'info': info_iaml,
            'metrics': iaml_metrics,
            'time': iaml_time
        }
    }
    
    return results

def plot_convergence(results):
    """
    Plot convergence comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Objective function
    axes[0, 0].semilogy(results['ADM']['info']['history']['objective'], 'b-', label='ADM', linewidth=2)
    axes[0, 0].semilogy(results['IALM']['info']['history']['objective'], 'r--', label='IALM', linewidth=2)
    axes[0, 0].set_xlabel('Iteration', fontsize=16)
    axes[0, 0].set_ylabel('Objective Value', fontsize=16)
    axes[0, 0].set_title('Objective Function Convergence', fontsize=18)
    axes[0, 0].legend(fontsize=16)
    axes[0, 0].grid(True)
    
    # Primal residual
    axes[0, 1].semilogy(results['ADM']['info']['history']['primal_residual'], 'b-', label='ADM', linewidth=2)
    axes[0, 1].semilogy(results['IALM']['info']['history']['primal_residual'], 'r--', label='IALM', linewidth=2)
    axes[0, 1].set_xlabel('Iteration', fontsize=16)
    axes[0, 1].set_ylabel('Primal Residual', fontsize=16)
    axes[0, 1].set_title('Primal Residual Convergence', fontsize=18)
    axes[0, 1].legend(fontsize=16)
    axes[0, 1].grid(True)
    
    # Dual residual
    axes[1, 0].semilogy(results['ADM']['info']['history']['dual_residual'], 'b-', label='ADM', linewidth=2)
    axes[1, 0].semilogy(results['IALM']['info']['history']['dual_residual'], 'r--', label='IALM', linewidth=2)
    axes[1, 0].set_xlabel('Iteration', fontsize=16)
    axes[1, 0].set_ylabel('Dual Residual', fontsize=16)
    axes[1, 0].set_title('Dual Residual Convergence', fontsize=18)
    axes[1, 0].legend(fontsize=16)
    axes[1, 0].grid(True)
    
    # Time vs objective
    axes[1, 1].semilogy(results['ADM']['info']['history']['time'], 
                       results['ADM']['info']['history']['objective'], 'b-', label='ADM', linewidth=2)
    axes[1, 1].semilogy(results['IALM']['info']['history']['time'], 
                       results['IALM']['info']['history']['objective'], 'r--', label='IALM', linewidth=2)
    axes[1, 1].set_xlabel('Time (seconds)', fontsize=16)
    axes[1, 1].set_ylabel('Objective Value', fontsize=16)
    axes[1, 1].set_title('Objective vs Time', fontsize=18)
    axes[1, 1].legend(fontsize=16)
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/home3/tianyi/RobustPCA/result/convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_decomposition(D, L_true, S_true, results):
    """
    Plot the decomposition results
    """
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    # Original data
    im1 = axes[0, 0].imshow(D, cmap='viridis')
    axes[0, 0].set_title('Original Data D', fontsize=16)
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # True components
    im2 = axes[0, 1].imshow(L_true, cmap='viridis')
    axes[0, 1].set_title('True Low-rank L', fontsize=16)
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(S_true, cmap='viridis')
    axes[0, 2].set_title('True Sparse S', fontsize=16)
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # ADMM results
    im4 = axes[1, 1].imshow(results['ADM']['L'], cmap='viridis')
    axes[1, 1].set_title('ADM Low-rank L', fontsize=16)
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1])
    
    im5 = axes[1, 2].imshow(results['ADM']['S'], cmap='viridis')
    axes[1, 2].set_title('ADM Sparse S', fontsize=16)
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2])
    
    # ADMM errors
    im6 = axes[1, 3].imshow(results['ADM']['L'] - L_true, cmap='RdBu')
    axes[1, 3].set_title('ADM L Error', fontsize=16)
    axes[1, 3].axis('off')
    plt.colorbar(im6, ax=axes[1, 3])
    
    im7 = axes[1, 4].imshow(results['ADM']['S'] - S_true, cmap='RdBu')
    axes[1, 4].set_title('ADM S Error', fontsize=16)
    axes[1, 4].axis('off')
    plt.colorbar(im7, ax=axes[1, 4])
    
    # IAML results
    im8 = axes[2, 1].imshow(results['IALM']['L'], cmap='viridis')
    axes[2, 1].set_title('IALM Low-rank L', fontsize=16)
    axes[2, 1].axis('off')
    plt.colorbar(im8, ax=axes[2, 1])
    
    im9 = axes[2, 2].imshow(results['IALM']['S'], cmap='viridis')
    axes[2, 2].set_title('IALM Sparse S', fontsize=16)
    axes[2, 2].axis('off')
    plt.colorbar(im9, ax=axes[2, 2])
    
    # IAML errors
    im10 = axes[2, 3].imshow(results['IALM']['L'] - L_true, cmap='RdBu')
    axes[2, 3].set_title('IALM L Error', fontsize=16)
    axes[2, 3].axis('off')
    plt.colorbar(im10, ax=axes[2, 3])
    
    im11 = axes[2, 4].imshow(results['IALM']['S'] - S_true, cmap='RdBu')
    axes[2, 4].set_title('IALM S Error', fontsize=16)
    axes[2, 4].axis('off')
    plt.colorbar(im11, ax=axes[2, 4])
    
    # Remove unused subplots
    axes[0, 3].axis('off')
    axes[0, 4].axis('off')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home3/tianyi/RobustPCA/result/decomposition_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_comparison_table(results):
    """
    Print a comparison table of the results
    """
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON RESULTS")
    print("="*80)
    
    print(f"{'Metric':<25} {'ADM':<15} {'IALM':<15} {'Winner':<10}")
    print("-"*70)
    
    # Performance metrics
    metrics = ['L_rel_error', 'S_rel_error', 'support_recovery', 'f1_score']
    metric_names = ['L Relative Error', 'S Relative Error', 'Support Recovery', 'F1 Score']
    
    for metric, name in zip(metrics, metric_names):
        admm_val = results['ADMM']['metrics'][metric]
        iaml_val = results['IAML']['metrics'][metric]
        
        if metric in ['L_rel_error', 'S_rel_error']:
            winner = 'ADM' if admm_val < iaml_val else 'IALM'
        else:
            winner = 'ADM' if admm_val > iaml_val else 'IALM'
            
        print(f"{name:<25} {admm_val:<15.4f} {iaml_val:<15.4f} {winner:<10}")
    
    # Convergence metrics
    print("-"*70)
    print(f"{'Iterations':<25} {results['ADMM']['info']['iterations']:<15} {results['IAML']['info']['iterations']:<15} {'ADMM' if results['ADMM']['info']['iterations'] < results['IAML']['info']['iterations'] else 'IAML':<10}")
    print(f"{'Total Time (s)':<25} {results['ADMM']['time']:<15.2f} {results['IAML']['time']:<15.2f} {'ADMM' if results['ADMM']['time'] < results['IAML']['time'] else 'IAML':<10}")
    print(f"{'Converged':<25} {results['ADMM']['info']['converged']:<15} {results['IAML']['info']['converged']:<15} {'-':<10}")
    
    print("="*80)

def main():
    """
    Main comparison function
    """
    # Generate synthetic data
    print("Generating synthetic data...")
    m, n, r = 100, 100, 5
    sparsity_level = 0.1
    noise_level = 5.0
    
    D, L_true, S_true = generate_synthetic_data(m, n, r, sparsity_level, noise_level)
    
    print(f"Data dimensions: {m} x {n}")
    print(f"True rank: {r}")
    print(f"Sparsity level: {sparsity_level:.1%}")
    print(f"Noise level: {noise_level}")
    
    # Run comparison
    results = compare_algorithms(D, L_true, S_true)
    
    # Print results
    print_comparison_table(results)
    
    # Plot results
    plot_convergence(results)
    plot_decomposition(D, L_true, S_true, results)
    
    print("\nComparison complete! Check the generated plots for visual results.")

if __name__ == "__main__":
    main() 