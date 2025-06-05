import cupy as xp
import numpy as np
from cupyx.scipy.special import gamma as gam_cp
import skued

def get_struct_factor(crystal, qs):
    return skued.structure_factor(crystal, *qs, normalized=True)

def get_rot_matrix_rodriguez(ks, thetas):
    """
    Rodrigues Formula
    """
    zero_array = xp.zeros_like(ks[0])
    K = xp.array(((zero_array, -ks[2], ks[1]), 
                 (ks[2], zero_array,-ks[0]),
                 (-ks[1],ks[0], zero_array)))
    K2 = xp.einsum("ij...,jk...->ik...", K, K)
    a1 = xp.sin(thetas)
    a2 = (1-xp.cos(thetas))
    U = xp.diag(np.ones(3)) + xp.einsum("...,jk...->jk...", a1, K).T + xp.einsum("...,jk...->jk...", a2, K2).T
    
    return U.T

def comb_cupy(n, k):
    """
    Compute the binomial coefficient (n choose k) for arrays using CuPy on the GPU.
    
    Parameters:
        n (int or cupy.ndarray): Total number of items. Can be a scalar or array.
        k (int or cupy.ndarray): Number of selected items. Can be a scalar or array.
        
    Returns:
        cupy.ndarray: Binomial coefficients, broadcasted over input arrays.
    """
    n = xp.asarray(n)
    k = xp.asarray(k)
    
    # Ensure k and n have the same shape via broadcasting
    n, k = xp.broadcast_arrays(n, k)
    
    # Handle invalid cases: k > n or k < 0
    k = xp.where(k > n, 0, k)  # If k > n, comb is 0
    k = xp.where(k < 0, 0, k)  # If k < 0, comb is 0
    
    # Use symmetry: C(n, k) == C(n, n-k) to minimize computations
    k = xp.minimum(k, n - k)
    
    # Efficient computation using logarithms
    log_factorial = lambda x: xp.log(gam_cp(x + 1))
    
    log_comb = log_factorial(n) - (log_factorial(k) + log_factorial(n - k))
    
    # Return the result in exponential form
    return xp.exp(log_comb)

# Function to compute the Bernstein polynomial
def bernstein_poly(n, t):
    i = xp.arange(n+1)
    t_new = t[:, None]
    i_new = i[None,:]
    return comb_cupy(n, i_new) * (t_new ** i_new) * ((1 - t_new) ** (n - i_new))

def bezier_surface(u, v, control_points):
    N_r = control_points.shape[0]
    n = control_points.shape[2]
    m = control_points.shape[1]
    
    B_u = bernstein_poly(n-1, u)
    B_v = bernstein_poly(m-1, v)
    return xp.einsum("ij,km,njml->nlik", B_v, B_u, control_points)

def bezier_basis_change(u, v, n, m):
    """
    Inputs: 
        u, v: 1D xp.ndarray of size k,l
        n, m: number of control points in each dimension
    
    Returns:
        R_flatter: basis change matrix from control points to the image of size (k*l)x(n*m)
    """
    
    B_u = bernstein_poly(n - 1, u)
    B_v = bernstein_poly(m - 1, v)

    R = xp.einsum("ij,km->ikjm", B_u, B_v)
    
    R_flatter = R.reshape((-1, n*m))
    return R_flatter

