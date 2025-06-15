import numpy as np
from scipy.linalg import svd
import time

def singular_value_thresholding(X, tau):
    """
    Singular Value Thresholding operator S_tau[X]
    
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
    Soft thresholding operator S_tau[X] applied element-wise
    
    Args:
        X: Input matrix
        tau: Threshold parameter
    
    Returns:
        Soft-thresholded matrix
    """
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def frobenius_norm(X):
    """Compute Frobenius norm of matrix X"""
    return np.linalg.norm(X, 'fro')

def J_function(D):
    """
    Compute J(D) = ||D||_F / max(||D||_2, ||D||_∞ / λ)
    This is a common initialization for Y_0 in IALM
    """
    norm_F = frobenius_norm(D)
    norm_2 = np.linalg.norm(D, 2)  # Spectral norm
    norm_inf = np.linalg.norm(D, np.inf)  # Infinity norm
    
    # Use a default lambda for the J function if not specified
    lambda_default = 1.0 / np.sqrt(max(D.shape))
    
    return norm_F / max(norm_2, norm_inf / lambda_default)

class IALM_RPCA:
    """
    Inexact Augmented Lagrangian Method for Robust PCA
    
    Solves: min ||L||_* + lambda ||S||_1 subject to D = L + S
    """
    
    def __init__(self, lambda_param=None, mu_init=None, rho=1.1, 
                 max_iter=1000, tol_primal=1e-6, tol_dual=1e-7, verbose=False):
        """
        Initialize IALM-RPCA solver
        
        Args:
            lambda_param: Regularization parameter (if None, uses 1/sqrt(max(m,n)))
            mu_init: Initial penalty parameter (if None, computed automatically)
            rho: Penalty parameter increase factor (rho > 1)
            max_iter: Maximum number of iterations
            tol_primal: Tolerance for primal constraint violation
            tol_dual: Tolerance for dual variables
            verbose: Print convergence information
        """
        self.lambda_param = lambda_param
        self.mu_init = mu_init
        self.rho = rho
        self.max_iter = max_iter
        self.tol_primal = tol_primal
        self.tol_dual = tol_dual
        self.verbose = verbose
        
    def fit(self, D):
        """
        Decompose matrix D into low-rank L and sparse S components using IALM
        
        Args:
            D: Observation matrix (m x n)
            
        Returns:
            L: Low-rank component
            S: Sparse component
            info: Dictionary with convergence information
        """
        m, n = D.shape
        
        # Set default lambda parameter
        if self.lambda_param is None:
            self.lambda_param = 1.0 / np.sqrt(max(m, n))
            
        # Initialize variables following the algorithm
        # Y_0 = D/J(D); S_0 = 0; mu_0 > 0; rho > 1; k = 0
        Y = D / J_function(D)
        S = np.zeros_like(D)  # S_0 = 0 (corresponds to E_0 = 0 in the image)
        L = np.zeros_like(D)  # L will be computed in first iteration
        
        # Set initial penalty parameter
        if self.mu_init is None:
            # Common initialization: mu_0 = 1.25 / ||D||_2
            self.mu_init = 1.25 / np.linalg.norm(D, 2)
        mu = self.mu_init
        
        k = 0
        
        # Store convergence history
        history = {
            'primal_residual': [],
            'dual_residual': [],
            'objective': [],
            'penalty_param': [],
            'time': []
        }
        
        start_time = time.time()
        
        while k < self.max_iter:
            L_old = L.copy()
            S_old = S.copy()
            
            # Lines 4-5: solve L_{k+1} = arg min_L L(L, S_k, Y_k, mu_k)
            # (U, S_svd, V) = svd(D - S_k + mu_k^{-1} Y_k)
            svd_input = D - S + (1.0/mu) * Y
            U, s_values, Vt = svd(svd_input, full_matrices=False)
            
            # L_{k+1} = U S_{mu_k^{-1}}[S] V^T
            L = singular_value_thresholding(svd_input, 1.0/mu)
            
            # Line 7: solve S_{k+1} = arg min_S L(L_{k+1}, S, Y_k, mu_k)
            # S_{k+1} = S_{lambda * mu_k^{-1}}[D - L_{k+1} + mu_k^{-1} Y_k]
            S = soft_thresholding(D - L + (1.0/mu) * Y, self.lambda_param / mu)
            
            # Update dual variable: Y_{k+1} = Y_k + mu_k(D - L_{k+1} - S_{k+1})
            constraint_violation = D - L - S
            Y = Y + mu * constraint_violation
            
            # Update mu_k to mu_{k+1}
            # Common strategy: increase mu when constraint violation is large
            primal_residual = frobenius_norm(constraint_violation)
            if k > 0:  # Don't update mu in first iteration
                prev_primal = history['primal_residual'][-1] if history['primal_residual'] else float('inf')
                if primal_residual > 0.9 * prev_primal:
                    mu = self.rho * mu
            
            # Compute dual residual (change in variables)
            dual_residual = max(
                frobenius_norm(L - L_old),
                frobenius_norm(S - S_old)
            )
            
            # Compute objective value
            nuclear_norm = np.sum(svd(L, compute_uv=False))
            l1_norm = np.sum(np.abs(S))
            objective = nuclear_norm + self.lambda_param * l1_norm
            
            # Store history
            history['primal_residual'].append(primal_residual)
            history['dual_residual'].append(dual_residual)
            history['objective'].append(objective)
            history['penalty_param'].append(mu)
            history['time'].append(time.time() - start_time)
            
            if self.verbose and k % 10 == 0:
                print(f"Iter {k:4d}: Primal={primal_residual:.2e}, "
                      f"Dual={dual_residual:.2e}, Obj={objective:.4f}, mu={mu:.2e}")
            
            # Check convergence
            if (dual_residual < self.tol_dual and 
                primal_residual < self.tol_primal):
                if self.verbose:
                    print(f"Converged at iteration {k}")
                break
            
            # k <- k + 1
            k += 1
                
        total_time = time.time() - start_time
        
        info = {
            'iterations': k,
            'converged': k < self.max_iter,
            'total_time': total_time,
            'final_primal_residual': primal_residual,
            'final_dual_residual': dual_residual,
            'final_objective': objective,
            'final_penalty_param': mu,
            'history': history
        }
        
        # Output: (L_k, S_k)
        return L, S, info

# Keep backward compatibility
class IAML_RPCA(IALM_RPCA):
    """Backward compatibility alias for IALM_RPCA"""
    pass

def ialm_rpca(D, lambda_param=None, **kwargs):
    """
    Convenience function for IALM-RPCA
    
    Args:
        D: Observation matrix
        lambda_param: Regularization parameter
        **kwargs: Additional parameters for IALM_RPCA
        
    Returns:
        L: Low-rank component
        S: Sparse component
        info: Convergence information
    """
    solver = IALM_RPCA(lambda_param=lambda_param, **kwargs)
    return solver.fit(D)

def iaml_rpca(D, lambda_param=None, **kwargs):
    """
    Convenience function for IAML-RPCA (backward compatibility)
    
    Args:
        D: Input data matrix
        lambda_param: Regularization parameter
        **kwargs: Additional parameters for IALM_RPCA
        
    Returns:
        L: Low-rank component
        S: Sparse component
        info: Convergence information
    """
    solver = IALM_RPCA(lambda_param=lambda_param, **kwargs)
    return solver.fit(D)

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
    
    print("Running IALM-RPCA...")
    L_recovered, S_recovered, info = ialm_rpca(D, verbose=True)
    
    # Compute recovery errors
    L_error = np.linalg.norm(L_recovered - L_true, 'fro') / np.linalg.norm(L_true, 'fro')
    S_error = np.linalg.norm(S_recovered - S_true, 'fro') / np.linalg.norm(S_true, 'fro')
    
    print(f"\nResults:")
    print(f"L relative error: {L_error:.4f}")
    print(f"S relative error: {S_error:.4f}")
    print(f"Iterations: {info['iterations']}")
    print(f"Total time: {info['total_time']:.2f}s")
    print(f"Converged: {info['converged']}")
    print(f"Final penalty parameter: {info['final_penalty_param']:.2e}")
    
    # Test backward compatibility
    print("\nTesting backward compatibility...")
    L_recovered_old, S_recovered_old, info_old = iaml_rpca(D, verbose=False)
    print(f"Backward compatibility test passed: {np.allclose(L_recovered, L_recovered_old)}") 