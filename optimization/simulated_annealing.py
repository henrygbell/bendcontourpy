import numpy as np
import cupy as xp
from tqdm import tqdm

def propose_new_state_all_z(p, 
                            dp, 
                            num_c
):  
    """
    Proposes a new state by updating all z-coordinates of the control points.

    Parameters:
        p (ndarray): Current control points.
        dp (float): Update amplitude.
        num_c (int): Number of control points in each dimension.
    """

    inds = np.array((0,0))
    while (inds == np.array((0,0))).all():
        inds = np.random.randint(low = 0, high = num_c, size = 2)
    num_grid = int(xp.sqrt(p.size/3))
    p_reshape = p.reshape((num_grid,num_grid,3)).copy()
    p_reshape[inds[0], inds[1], 2] += dp*(xp.random.rand()*2-1)
    return p_reshape.flatten()

def propose_new_state_middle_z(p, 
                             dp, 
                             num_c
):
    """
    Proposes a new state by updating the z-coordinates of the middle control points.

    Parameters:
        p (ndarray): Current control points.
        dp (float): Update amplitude.
    """
    inds = np.random.randint(low = 1, high = num_c-1, size = 2)
    num_grid = int(xp.sqrt(p.size/3))
    p_reshape = p.reshape((num_grid,num_grid,3)).copy()
    p_reshape[inds[0], inds[1], 2] += dp*(xp.random.rand()*2-1)
    return p_reshape.flatten()

def propose_new_state_middle_xyz(
                                 p,
                                 dp_xy,
                                 dp_z,
                                 num_c,
):
    """
    Proposes a new state by updating the x, y, and z-coordinates of the middle control points.

    Parameters:
        p (ndarray): Current control points.
        dp_xy (float): Update amplitude for x and y-coordinates.
        dp_z (float): Update amplitude for z-coordinate.
    """
    inds = np.random.randint(low = 1, high = num_c-1, size = 2)
    ind_xyz = np.clip(np.random.randint(low = 0, high = 5, size = 1), 0, 2)
    
    num_grid = int(xp.sqrt(p.size/3))
    p_reshape = p.reshape((num_grid,num_grid,3)).copy()
    
    dp_vec = xp.array((dp_xy, dp_xy, dp_z))
    
    p_reshape[inds[0], inds[1], ind_xyz] += dp_vec[ind_xyz]*(xp.random.rand()*2-1)

    return p_reshape.flatten()

def linear_cooling_schedule(ind, T0):
    """
    Implements a linear cooling schedule for the temperature.

    Parameters:
        ind (int): Current iteration index.
        T0 (float): Initial temperature.
    """
    return T0*ind

def simulated_annealing(cost_f, 
                        p0, 
                        T0, 
                        propose_new_state, 
                        k_max, 
                        num_c, 
                        cooling_schedule = linear_cooling_schedule,
                        printing = False,
                        *args,
                        **kwargs, 
):
    """
    Performs simulated annealing optimization for surface fitting.

    Parameters:
        cost_f (callable): Cost function to minimize.
        p0 (ndarray): Initial parameters.
        T0 (float): Initial temperature.
        propose_new_state (callable): Function to generate new states.
        k_max (int): Maximum number of iterations.
        num_c (int): Number of control points in each dimension.
        cooling_schedule (callable, optional): Temperature reduction function. Defaults to linear_cooling_schedule.
        printing (bool, optional): Whether to print progress. Defaults to False.
    
    Returns:
        tuple: (parameter_history, cost_history, iteration_indices).
    """
    
    def acceptance_prob(c_f, c_f_p, T):
        if c_f_p < c_f:
            return 1
        else:
            return np.exp(-(c_f_p - c_f)/T)
    
    cost_f_evals = []
    p = p0
    p_collect = []
    collect_ks = np.round(np.linspace(0, k_max, 100), 0)
    cost_f_eval = cost_f(p)
    for k in tqdm(range(k_max)):
        T = cooling_schedule(1 - k/(k_max-1), T0)
        p_prime = propose_new_state(p, num_c = num_c, *args, **kwargs)
        cost_f_eval_prime = cost_f(p_prime)
        
        cost_f_evals.append(cost_f_eval)
    
        a_prob = acceptance_prob(cost_f_eval, cost_f_eval_prime, T)

        if printing:
            print(f"{k = }, {T = }, {cost_f_eval = }, {a_prob = }, {(cost_f_eval - cost_f_eval_prime)/T =}, {cost_f_eval_prime = }, {cost_f_eval = }")
        
        if np.random.rand() < a_prob:
            p = p_prime
            cost_f_eval = cost_f_eval_prime
        if k in collect_ks:
            p_collect.append(p) 
    p_collect.append(p) 
    return  p_collect, cost_f_evals, collect_ks