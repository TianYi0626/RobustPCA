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

class ADMM_RPCA:
    """
    Alternating Direction Method of Multipliers for Robust PCA
    
    Solves: min ||L||_* + lambda ||S||_1 subject to D = L + S
    """
    
    def __init__(self, lambda_param=None, rho=1.0, max_iter=1000, 
                 tol_primal=1e-6, tol_dual=1e-7, verbose=False):
        """
        Initialize ADMM-RPCA solver
        
        Args:
            lambda_param: Regularization parameter (if None, uses 1/sqrt(max(m,n)))
            rho: Penalty parameter
            max_iter: Maximum number of iterations
            tol_primal: Tolerance for primal variables
            tol_dual: Tolerance for constraint violation
            verbose: Print convergence information
        """
        self.lambda_param = lambda_param
        self.rho = rho
        self.max_iter = max_iter
        self.tol_primal = tol_primal
        self.tol_dual = tol_dual
        self.verbose = verbose
        
    def fit(self, D):
        """
        Decompose matrix D into low-rank L and sparse S components
        
        Args:
            D: Input data matrix (m x n)
            
        Returns:
            L: Low-rank component
            S: Sparse component
            info: Dictionary with convergence information
        """
        m, n = D.shape
        
        # Set default lambda parameter
        if self.lambda_param is None:
            self.lambda_param = 1.0 / np.sqrt(max(m, n))
            
        # Initialize variables
        L = np.zeros_like(D)
        S = np.zeros_like(D)
        Z = np.zeros_like(D)
        Y1 = np.zeros_like(D)  # Dual variable for L + S - Z = 0
        Y2 = np.zeros_like(D)  # Dual variable for D - Z = 0
        
        # Store convergence history
        history = {
            'primal_residual': [],
            'dual_residual': [],
            'objective': [],
            'time': []
        }
        
        start_time = time.time()
        
        for k in range(self.max_iter):
            L_old = L.copy()
            S_old = S.copy()
            Z_old = Z.copy()
            
            # Update L (L-subproblem)
            # L^{k+1} = D_{1/rho}(Z^k - S^k - Y1^k/rho)
            L = singular_value_thresholding(Z - S - Y1/self.rho, 1.0/self.rho)
            
            # Update S (S-subproblem)
            # S^{k+1} = S_{lambda/rho}(Z^k - L^{k+1} - Y1^k/rho)
            S = soft_thresholding(Z - L - Y1/self.rho, self.lambda_param/self.rho)
            
            # Update Z (Z-subproblem)
            # Z^{k+1} = (L^{k+1} + S^{k+1} + Y1^k/rho + D + Y2^k/rho) / 2
            Z = 0.5 * (L + S + Y1/self.rho + D + Y2/self.rho)
            
            # Update dual variables
            Y1 = Y1 + self.rho * (L + S - Z)
            Y2 = Y2 + self.rho * (D - Z)
            
            # Compute residuals
            primal_residual1 = np.linalg.norm(L + S - Z, 'fro')
            primal_residual2 = np.linalg.norm(D - Z, 'fro')
            primal_residual = max(primal_residual1, primal_residual2)
            
            dual_residual = max(
                np.linalg.norm(L - L_old, 'fro'),
                np.linalg.norm(S - S_old, 'fro'),
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
            if (dual_residual < self.tol_primal and 
                primal_residual < self.tol_dual):
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
        
        return L, S, info

def admm_rpca(D, lambda_param=None, **kwargs):
    """
    Convenience function for ADMM-RPCA
    
    Args:
        D: Input data matrix
        lambda_param: Regularization parameter
        **kwargs: Additional parameters for ADMM_RPCA
        
    Returns:
        L: Low-rank component
        S: Sparse component
        info: Convergence information
    """
    solver = ADMM_RPCA(lambda_param=lambda_param, **kwargs)
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
    
    print("Running ADMM-RPCA...")
    L_recovered, S_recovered, info = admm_rpca(D, verbose=True)
    
    # Compute recovery errors
    L_error = np.linalg.norm(L_recovered - L_true, 'fro') / np.linalg.norm(L_true, 'fro')
    S_error = np.linalg.norm(S_recovered - S_true, 'fro') / np.linalg.norm(S_true, 'fro')
    
    print(f"\nResults:")
    print(f"L relative error: {L_error:.4f}")
    print(f"S relative error: {S_error:.4f}")
    print(f"Iterations: {info['iterations']}")
    print(f"Total time: {info['total_time']:.2f}s")
    print(f"Converged: {info['converged']}") 