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


class SimulatedAnnealingOptimizer:
    """
    A general simulated annealing optimizer that works with parameter dictionaries.
    
    This optimizer allows for:
    - Fixed vs. variable parameters
    - Individual step sizes (dp) for each parameter
    - Different parameter types (scalar, array, etc.)
    - Bounds enforcement
    """
    
    def __init__(self, param_dict, cost_function, verbose=False):
        """
        Initialize the optimizer.
        
        Parameters:
        -----------
        param_dict : dict
            Dictionary of parameters with format:
            {
                "param_name": {
                    "Fixed": bool,      # Whether parameter is fixed
                    "value": any,       # Current value of parameter
                    "bounds": (min, max),  # Bounds for parameter (optional)
                    "dp": float         # Step size for parameter updates
                },
                ...
            }
        cost_function : callable
            Function that takes a param_dict and returns a scalar cost
        verbose : bool
            Whether to print detailed progress
        """
        self.param_dict = param_dict
        self.cost_function = cost_function
        self.verbose = verbose
        self.best_params = None
        self.best_cost = float('inf')
        self.cost_history = []
        self.param_history = []
        
        # Validate param_dict and add default dp if not present
        self._validate_params()
    
    def _validate_params(self):
        """Validate parameters and add default step sizes if missing"""
        for name, param in self.param_dict.items():
            if "dp" not in param:
                # Set default step size based on parameter type
                if isinstance(param["value"], (int, float)):
                    # Default step size for scalars: 5% of range or 0.01
                    if "bounds" in param and param["bounds"] is not None:
                        param["dp"] = (param["bounds"][1] - param["bounds"][0]) * 0.05
                    else:
                        param["dp"] = max(abs(param["value"]) * 0.05, 0.01)
                elif hasattr(param["value"], "shape"):
                    # For arrays, default to 5% of range or 0.01 for each element
                    if "bounds" in param and param["bounds"] is not None:
                        param["dp"] = np.ones_like(param["value"]) * (param["bounds"][1] - param["bounds"][0]) * 0.05
                    else:
                        param["dp"] = np.maximum(np.abs(param["value"]) * 0.05, 0.01)
                else:
                    param["dp"] = 0.01
    
    def _propose_new_state(self, current_params):
        """
        Propose a new state by perturbing a randomly selected parameter.
        
        Parameters:
        -----------
        current_params : dict
            Current parameter dictionary
            
        Returns:
        --------
        dict
            New parameter dictionary with one parameter perturbed
        """
        # Make a deep copy of parameters
        new_params = {}
        for name, param in current_params.items():
            new_params[name] = {k: v.copy() if hasattr(v, "copy") else v 
                               for k, v in param.items()}
        
        # Get list of non-fixed parameters
        variable_params = [name for name, param in new_params.items() 
                          if not param["Fixed"]]
        
        if not variable_params:
            return current_params  # No variable parameters to modify
        
        # Select a random parameter to modify
        param_name = np.random.choice(variable_params)
        param = new_params[param_name]
        
        # Get the step size
        dp = param["dp"]
        
        # Handle different parameter types
        if isinstance(param["value"], (int, float)):
            # Scalar parameter - add random perturbation
            perturbation = np.random.normal(0, dp)
            new_value = param["value"] + perturbation
            
            # Enforce bounds if specified
            if "bounds" in param and param["bounds"] is not None:
                new_value = max(param["bounds"][0], min(param["bounds"][1], new_value))
                
            # Update parameter
            new_params[param_name]["value"] = new_value
            
        elif hasattr(param["value"], "shape"):
            # Array parameter - modify a random element
            shape = param["value"].shape
            flat_size = np.prod(shape)
            
            # Select random indices to modify
            if flat_size > 1:
                idx = tuple(np.random.randint(0, dim_size) for dim_size in shape)
                
                # Calculate perturbation
                if isinstance(dp, (int, float)):
                    perturbation = np.random.normal(0, dp)
                else:
                    # If dp is an array, use the corresponding element
                    perturbation = np.random.normal(0, dp[idx])
                
                # Apply perturbation
                new_value = param["value"].copy()
                new_value[idx] += perturbation
                
                # Enforce bounds if specified
                if "bounds" in param and param["bounds"] is not None:
                    low, high = param["bounds"]
                    new_value[idx] = max(low, min(high, new_value[idx]))
            else:
                # Single element array
                perturbation = np.random.normal(0, dp if isinstance(dp, (int, float)) else dp.item())
                new_value = param["value"] + perturbation
                
                # Enforce bounds if specified
                if "bounds" in param and param["bounds"] is not None:
                    low, high = param["bounds"]
                    new_value = max(low, min(high, new_value))
            
            # Update parameter
            new_params[param_name]["value"] = new_value
        
        return new_params
    
    def _extract_values(self, params_dict):
        """Extract just the values from a parameter dictionary"""
        return {name: param["value"] for name, param in params_dict.items()}
    
    def _calculate_cost(self, params_dict):
        """Calculate cost for a parameter set"""
        values_dict = self._extract_values(params_dict)
        return self.cost_function(values_dict)
    
    def _acceptance_probability(self, current_cost, new_cost, temperature):
        """Calculate probability of accepting a new state"""
        if new_cost < current_cost:
            return 1.0
        else:
            if temperature <= 0:
                return 0.0
            return np.exp(-(new_cost - current_cost) / temperature)
    
    def _cooling_schedule(self, fraction, initial_temp):
        """Linear cooling schedule from T0 to 0"""
        return initial_temp * (1 - fraction)
    
    def optimize(self, initial_temp=None, max_iterations=1000, 
                 cooling_schedule=None, callback=None):
        """
        Run the simulated annealing optimization.
        
        Parameters:
        -----------
        initial_temp : float
            Initial temperature
        max_iterations : int
            Maximum number of iterations
        cooling_schedule : callable, optional
            Function that takes (iteration_fraction, initial_temp) and returns current temperature
        callback : callable, optional
            Function called after each iteration with (iteration, params, cost, temperature)
            
        Returns:
        --------
        dict
            Optimization results with best parameters, cost history, etc.
        """
        if cooling_schedule is None:
            cooling_schedule = self._cooling_schedule
        if initial_temp is None:
            initial_temp = self.calculate_initial_temperature()
            print("Using auto-calculated initial temperature:", initial_temp)
        
        # Initialize
        current_params = self.param_dict
        current_cost = self._calculate_cost(current_params)
        
        self.best_params = {name: {k: v.copy() if hasattr(v, "copy") else v 
                                  for k, v in param.items()}
                            for name, param in current_params.items()}
        self.best_cost = current_cost
        
        self.cost_history = [current_cost]
        self.param_history = [{name: param["value"].copy() if hasattr(param["value"], "copy") else param["value"]
                              for name, param in current_params.items()}]
        
        # Run optimization
        iterator = tqdm(range(max_iterations)) if self.verbose else range(max_iterations)
        for i in iterator:
            # Calculate current temperature
            fraction = i / (max_iterations - 1) if max_iterations > 1 else 1
            temperature = cooling_schedule(fraction, initial_temp)
            
            # Propose new state
            new_params = self._propose_new_state(current_params)
            new_cost = self._calculate_cost(new_params)
            
            # Decide whether to accept the new state
            if np.random.random() < self._acceptance_probability(current_cost, new_cost, temperature):
                current_params = new_params
                current_cost = new_cost
                
                # Update best if needed
                if new_cost < self.best_cost:
                    self.best_params = {name: {k: v.copy() if hasattr(v, "copy") else v 
                                             for k, v in param.items()}
                                       for name, param in new_params.items()}
                    self.best_cost = new_cost
            
            # Track history
            self.cost_history.append(current_cost)
            self.param_history.append({name: param["value"].copy() if hasattr(param["value"], "copy") else param["value"]
                                     for name, param in current_params.items()})
            
            # Call callback if provided
            if callback is not None:
                callback(i, current_params, current_cost, temperature)
            
            # Update progress bar
            if self.verbose and isinstance(iterator, tqdm):
                iterator.set_description(f"Cost: {current_cost:.6f}, Best: {self.best_cost:.6f}, T: {temperature:.6f}")
        
        # Return optimization results
        result = {
            'best_params': self._extract_values(self.best_params),
            'best_cost': self.best_cost,
            'cost_history': self.cost_history,
            'param_history': self.param_history,
            'final_temperature': cooling_schedule(1.0, initial_temp),
            'iterations': max_iterations
        }
        
        return result
    
    def calculate_initial_temperature(self, num_samples=100, acceptance_ratio=0.5):
        """
        Calculate an appropriate initial temperature automatically.
        
        This method performs random sampling of the parameter space to determine
        a good initial temperature that will accept worse solutions with the
        target probability.
        
        Parameters:
        -----------
        num_samples : int
            Number of random moves to test
        acceptance_ratio : float
            Target initial acceptance probability for negative moves (0.0-1.0)
            
        Returns:
        --------
        float
            Recommended initial temperature
        """
        if acceptance_ratio <= 0 or acceptance_ratio >= 1:
            raise ValueError("acceptance_ratio must be between 0 and 1")
        
        # Start with current parameters
        current_params = self.param_dict.copy()
        current_cost = self._calculate_cost(current_params)
        
        # Collect cost differences from random perturbations
        cost_diffs = []
        
        for _ in range(num_samples):
            # Generate a random neighbor
            new_params = self._propose_new_state(current_params)
            new_cost = self._calculate_cost(new_params)
            
            # Store the cost difference if it's a worse solution
            diff = new_cost - current_cost
            if diff > 0:  # Only consider uphill moves
                cost_diffs.append(diff)
        
        # If no uphill moves found, use a default
        if not cost_diffs:
            if self.verbose:
                print("Warning: No uphill moves found in sampling. Using default temperature.")
            # Return a default based on current cost
            return abs(current_cost) * 0.1 + 1e-3
        
        # Calculate average uphill move
        avg_diff = np.mean(cost_diffs)
        
        # Calculate temperature for desired acceptance ratio
        # From P(accept) = exp(-diff/T), we get T = -diff/ln(P)
        temp = avg_diff / -np.log(acceptance_ratio)
        
        if self.verbose:
            print(f"Auto temperature: {temp:.6f} (from {len(cost_diffs)} uphill samples, avg diff: {avg_diff:.6f})")
        
        return temp