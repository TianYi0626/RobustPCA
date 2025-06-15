import numpy as np
from scipy.linalg import svd
import time

def singular_value_thresholding(X, tau):
    """
    Singular Value Thresholding operator D_tau(X)
    
    Args:
        X: Input matrix
        tau: Threshold parameter
    
    Returns:
        Thresholded matrix
    """
    U, s, Vt = svd(X, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0)
    return U @ np.diag(s_thresh) @ Vt

def soft_thresholding(X, tau):
    """
    Soft thresholding operator S_tau(X) applied element-wise
    
    Args:
        X: Input matrix
        tau: Threshold parameter
    
    Returns:
        Soft-thresholded matrix
    """
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def projection_complement_omega(X, Omega):
    """
    Projection operator P_{Omega^c}[X] onto the complement of observed entries
    
    Args:
        X: Input matrix
        Omega: Set of observed entry indices (boolean mask)
    
    Returns:
        Projected matrix (zeros at observed entries, keeps unobserved entries)
    """
    result = X.copy()
    if isinstance(Omega, np.ndarray) and Omega.dtype == bool:
        # Omega is a boolean mask - zero out observed entries (keep unobserved)
        result[Omega] = 0
    else:
        # Convert to boolean mask if needed
        mask = np.zeros_like(X, dtype=bool)
        if len(Omega) > 0:
            mask[Omega] = True
            result[mask] = 0
    return result

def projection_omega(X, Omega):
    """
    Projection operator P_Omega[X] onto observed entries
    
    Args:
        X: Input matrix
        Omega: Set of observed entry indices (boolean mask)
    
    Returns:
        Projected matrix (keeps observed entries, zeros unobserved entries)
    """
    result = np.zeros_like(X)
    if isinstance(Omega, np.ndarray) and Omega.dtype == bool:
        # Omega is a boolean mask - keep observed entries
        result[Omega] = X[Omega]
    else:
        # Convert to boolean mask if needed
        if len(Omega) > 0:
            result[Omega] = X[Omega]
    return result

def projection_box_constraint(X, gamma_over_mu):
    """
    Projection operator P_{Ω^{γ/μ}_∞}[X] for box constraint
    Projects onto the set {X | -γ/μ ≤ X_ij ≤ γ/μ}
    
    Args:
        X: Input matrix
        gamma_over_mu: Box constraint parameter γ/μ
    
    Returns:
        Projected matrix
    """
    return np.clip(X, -gamma_over_mu, gamma_over_mu)

class ADMM_SLRMD:
    """
    Alternating Direction Method of Multipliers for Sparse and Low-Rank Matrix Decomposition (SLRMD)
    
    Solves: 
    - Standard RPCA: min ||L||_* + lambda ||S||_1 subject to D = L + S
    - Matrix Completion: min ||L||_* + lambda ||S||_1 subject to P_Omega[C] = P_Omega[L + S]
    """
    
    def __init__(self, lambda_param=None, mu=1.0, gamma=None, max_iter=1000, 
                 tol_primal=1e-6, tol_dual=1e-7, verbose=False):
        """
        Initialize ADMM-SLRMD solver
        
        Args:
            lambda_param: Regularization parameter (if None, uses 1/sqrt(max(m,n)))
            mu: Penalty parameter (renamed from beta to match algorithm description)
            gamma: Box constraint parameter for S-subproblem (if None, uses lambda)
            max_iter: Maximum number of iterations
            tol_primal: Tolerance for primal variables
            tol_dual: Tolerance for constraint violation
            verbose: Print convergence information
        """
        self.lambda_param = lambda_param
        self.mu = mu
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol_primal = tol_primal
        self.tol_dual = tol_dual
        self.verbose = verbose
        
    def fit(self, D, Omega=None):
        """
        Decompose matrix D into sparse S and low-rank L components
        
        Args:
            D: Input data matrix (m x n) with possible missing entries
            Omega: Set of observed entry indices (boolean mask). If None, all entries are observed (standard RPCA).
            
        Returns:
            S: Sparse component
            L: Low-rank component
            info: Dictionary with convergence information
        """
        m, n = D.shape
        
        # Set default parameters
        if self.lambda_param is None:
            self.lambda_param = 1.0 / np.sqrt(max(m, n))
        if self.gamma is None:
            self.gamma = self.lambda_param
            
        # Determine if this is standard RPCA or matrix completion
        if Omega is None:
            # Standard RPCA case - all entries observed
            is_standard_rpca = True
            Omega = np.ones_like(D, dtype=bool)
        else:
            # Matrix completion case - some entries missing
            is_standard_rpca = False
            if not isinstance(Omega, np.ndarray):
                Omega = np.array(Omega)
            if Omega.dtype != bool:
                # Convert indices to boolean mask
                mask = np.zeros_like(D, dtype=bool)
                mask[Omega] = True
                Omega = mask
                
        # Initialize variables - only S, L, Z (no Y1, Y2)
        S = np.zeros_like(D)
        L = np.zeros_like(D)
        Z = np.zeros_like(D)
        
        # Store convergence history
        history = {
            'primal_residual': [],
            'dual_residual': [],
            'objective': [],
            'time': []
        }
        
        start_time = time.time()
        
        for k in range(self.max_iter):
            S_old = S.copy()
            L_old = L.copy()
            Z_old = Z.copy()
            
            if is_standard_rpca:
                # Standard RPCA using the algorithm from the description
                
                # Step 1: Generate S^{k+1}
                # S^{k+1} = (1/μ)Z^k - L^k + D - P_{Ω^{γ/μ}_∞}[(1/μ)Z^k - L^k + D]
                temp = (1.0/self.mu) * Z - L + D
                S = temp - projection_box_constraint(temp, self.gamma/self.mu)
                
                # Step 2: Generate L^{k+1}
                # L^{k+1} = SVT_{1/μ}[D - S^{k+1} + (1/μ)Z^k]
                svd_input = D - S + (1.0/self.mu) * Z
                L = singular_value_thresholding(svd_input, 1.0/self.mu)
                
                # Step 3: Update the multiplier
                # Z^{k+1} = Z^k - μ(S^{k+1} + L^{k+1} - D)
                constraint_violation = S + L - D
                Z = Z - self.mu * constraint_violation
                
                primal_residual = np.linalg.norm(constraint_violation, 'fro')
                
            else:
                # Matrix completion case: use the SLRMD formulation from the image
                # This is for problems with missing entries
                
                # Step 1: Generate S^{k+1} (A^{k+1} in the image, where A=S)
                # S^{k+1} = (1/β)Z^k - L^k + P_Ω[C] - P_{Ω^c}[(1/β)Z^k - L^k + P_Ω[C]]
                temp = (1.0/self.mu) * Z - L + projection_omega(D, Omega)
                S = temp - projection_complement_omega(temp, Omega)
                
                # Step 2: Generate L^{k+1} (B^{k+1} in the image, where B=L)
                # L^{k+1} = SVT_{1/β}[C - S^{k+1} + (1/β)Z^k]
                svd_input = D - S + (1.0/self.mu) * Z
                L = singular_value_thresholding(svd_input, 1.0/self.mu)
                
                # Step 3: Update the multiplier
                # Z^{k+1} = Z^k - β(S^{k+1} + L^{k+1} - C)
                constraint_violation = S + L - D
                Z = Z - self.mu * constraint_violation
                
                primal_residual = np.linalg.norm(constraint_violation, 'fro')
            
            # Compute dual residual (change in variables)
            dual_residual = max(
                np.linalg.norm(S - S_old, 'fro'),
                np.linalg.norm(L - L_old, 'fro'),
                np.linalg.norm(Z - Z_old, 'fro')
            )
            
            # Compute objective value
            nuclear_norm = np.sum(svd(L, compute_uv=False))
            l1_norm = np.sum(np.abs(S))
            objective = nuclear_norm + self.lambda_param * l1_norm
            
            # Store history
            history['primal_residual'].append(primal_residual)
            history['dual_residual'].append(dual_residual)
            history['objective'].append(objective)
            history['time'].append(time.time() - start_time)
            
            if self.verbose and k % 10 == 0:
                print(f"Iter {k:4d}: Primal={primal_residual:.2e}, "
                      f"Dual={dual_residual:.2e}, Obj={objective:.4f}")
            
            # Check convergence
            if (dual_residual < self.tol_dual and 
                primal_residual < self.tol_primal):
                if self.verbose:
                    print(f"Converged at iteration {k}")
                break
                
        total_time = time.time() - start_time
        
        info = {
            'iterations': k + 1,
            'converged': k < self.max_iter - 1,
            'total_time': total_time,
            'final_primal_residual': primal_residual,
            'final_dual_residual': dual_residual,
            'final_objective': objective,
            'history': history
        }
        
        return S, L, info

# Keep backward compatibility
class ADMM_RPCA(ADMM_SLRMD):
    """Backward compatibility alias for ADMM_SLRMD"""
    pass

def admm_slrmd(C, lambda_param=None, Omega=None, **kwargs):
    """
    Convenience function for ADMM-SLRMD
    
    Args:
        C: Input data matrix
        lambda_param: Regularization parameter
        Omega: Set of observed entry indices
        **kwargs: Additional parameters for ADMM_SLRMD
        
    Returns:
        S: Sparse component
        L: Low-rank component
        info: Convergence information
    """
    solver = ADMM_SLRMD(lambda_param=lambda_param, **kwargs)
    return solver.fit(C, Omega)

def admm_rpca(D, lambda_param=None, **kwargs):
    """
    Convenience function for ADMM-RPCA (backward compatibility)
    
    Args:
        D: Input data matrix
        lambda_param: Regularization parameter
        **kwargs: Additional parameters for ADMM_SLRMD
        
    Returns:
        L: Low-rank component
        S: Sparse component
        info: Convergence information
    """
    solver = ADMM_SLRMD(lambda_param=lambda_param, **kwargs)
    S, L, info = solver.fit(D)
    return L, S, info  # Return in original order for backward compatibility

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic data
    m, n, r = 100, 100, 5
    U = np.random.randn(m, r)
    V = np.random.randn(n, r)
    L_true = U @ V.T
    
    # Add sparse corruption
    S_true = np.zeros((m, n))
    corruption_indices = np.random.choice(m*n, size=int(0.1*m*n), replace=False)
    S_true.flat[corruption_indices] = np.random.randn(len(corruption_indices)) * 5
    
    D = L_true + S_true
    
    print("Testing Standard RPCA (no missing entries)...")
    S_recovered, L_recovered, info = admm_slrmd(D, verbose=True)
    
    # Compute recovery errors
    L_error = np.linalg.norm(L_recovered - L_true, 'fro') / np.linalg.norm(L_true, 'fro')
    S_error = np.linalg.norm(S_recovered - S_true, 'fro') / np.linalg.norm(S_true, 'fro')
    
    print(f"\nStandard RPCA Results:")
    print(f"L relative error: {L_error:.4f}")
    print(f"S relative error: {S_error:.4f}")
    print(f"Iterations: {info['iterations']}")
    print(f"Total time: {info['total_time']:.2f}s")
    print(f"Converged: {info['converged']}")
    
    # Test with missing entries
    print("\n" + "="*50)
    print("Testing Matrix Completion (with missing entries)...")
    missing_ratio = 0.3
    total_entries = m * n
    missing_indices = np.random.choice(total_entries, size=int(missing_ratio * total_entries), replace=False)
    Omega = np.ones((m, n), dtype=bool)
    Omega.flat[missing_indices] = False
    
    # Create observed data
    D_observed = D.copy()
    D_observed[~Omega] = 0  # Zero out missing entries
    
    S_recovered_mc, L_recovered_mc, info_mc = admm_slrmd(D_observed, Omega=Omega, verbose=True)
    
    # Compute recovery errors
    L_error_mc = np.linalg.norm(L_recovered_mc - L_true, 'fro') / np.linalg.norm(L_true, 'fro')
    S_error_mc = np.linalg.norm(S_recovered_mc - S_true, 'fro') / np.linalg.norm(S_true, 'fro')
    
    print(f"\nMatrix Completion Results:")
    print(f"L relative error: {L_error_mc:.4f}")
    print(f"S relative error: {S_error_mc:.4f}")
    print(f"Iterations: {info_mc['iterations']}")
    print(f"Total time: {info_mc['total_time']:.2f}s")
    print(f"Converged: {info_mc['converged']}")
    
    # Test backward compatibility
    print("\nTesting backward compatibility...")
    L_recovered_old, S_recovered_old, info_old = admm_rpca(D, verbose=False)
    print(f"Backward compatibility test passed: {np.allclose(L_recovered, L_recovered_old)}") 