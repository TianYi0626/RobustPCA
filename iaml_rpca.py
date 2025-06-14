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

class IAML_RPCA:
    """
    Iterative Augmented Lagrangian Method for Robust PCA
    
    Solves: min ||L||_* + lambda ||S||_1 subject to D = L + S
    """
    
    def __init__(self, lambda_param=None, mu_init=0.1, mu_max=1e6, 
                 rho_mu=1.1, max_iter=1000, tol_primal=1e-6, 
                 tol_dual=1e-7, verbose=False):
        """
        Initialize IAML-RPCA solver
        
        Args:
            lambda_param: Regularization parameter (if None, uses 1/sqrt(max(m,n)))
            mu_init: Initial penalty parameter
            mu_max: Maximum penalty parameter
            rho_mu: Penalty parameter increase factor
            max_iter: Maximum number of iterations
            tol_primal: Tolerance for primal variables
            tol_dual: Tolerance for constraint violation
            verbose: Print convergence information
        """
        self.lambda_param = lambda_param
        self.mu_init = mu_init
        self.mu_max = mu_max
        self.rho_mu = rho_mu
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
        Y = np.zeros_like(D)  # Dual variable
        mu = self.mu_init     # Penalty parameter
        
        # Store convergence history
        history = {
            'primal_residual': [],
            'dual_residual': [],
            'objective': [],
            'penalty_param': [],
            'time': []
        }
        
        start_time = time.time()
        
        for k in range(self.max_iter):
            L_old = L.copy()
            S_old = S.copy()
            
            # Update L (L-subproblem)
            # L^{k+1} = D_{1/mu}(D - S^k + Y^k/mu)
            L = singular_value_thresholding(D - S + Y/mu, 1.0/mu)
            
            # Update S (S-subproblem)
            # S^{k+1} = S_{lambda/mu}(D - L^{k+1} + Y^k/mu)
            S = soft_thresholding(D - L + Y/mu, self.lambda_param/mu)
            
            # Update dual variable
            # Y^{k+1} = Y^k + mu(D - L^{k+1} - S^{k+1})
            constraint_violation = D - L - S
            Y = Y + mu * constraint_violation
            
            # Update penalty parameter
            mu = min(self.rho_mu * mu, self.mu_max)
            
            # Compute residuals
            primal_residual = np.linalg.norm(constraint_violation, 'fro')
            
            dual_residual = max(
                np.linalg.norm(L - L_old, 'fro'),
                np.linalg.norm(S - S_old, 'fro')
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
            'final_penalty_param': mu,
            'history': history
        }
        
        return L, S, info

def iaml_rpca(D, lambda_param=None, **kwargs):
    """
    Convenience function for IAML-RPCA
    
    Args:
        D: Input data matrix
        lambda_param: Regularization parameter
        **kwargs: Additional parameters for IAML_RPCA
        
    Returns:
        L: Low-rank component
        S: Sparse component
        info: Convergence information
    """
    solver = IAML_RPCA(lambda_param=lambda_param, **kwargs)
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
    
    print("Running IAML-RPCA...")
    L_recovered, S_recovered, info = iaml_rpca(D, verbose=True)
    
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