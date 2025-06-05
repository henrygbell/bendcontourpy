from ..utils import get_rot_matrix_rodriguez, cross_correlation_registration
from ..surfaces import Bezier_Surfaces
from ..experiment import Experiment
import numpy as np
import cupy as xp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import h5py
import os
import datetime


def get_exp_surface(param_dict, info_dict):
    rot_ip = get_rot_matrix_rodriguez(ks = xp.array([0,0,1]), thetas = xp.array([param_dict["theta_lattice"]]))

    control_points_start = info_dict["control_points_start_xyz"].copy()
    
    control_points_start[0,:,:,0] += xp.array(param_dict["control_points_update_x"])
    control_points_start[0,:,:,1] += xp.array(param_dict["control_points_update_y"])
    control_points_start[0,:,:,2] += xp.array(param_dict["control_points_update_z"])
    
    control_points_start[0,:,:,0:2] *= param_dict["scale_xy"]
    
    if "control_points_dalpha" in param_dict:
        control_points_alpha = xp.array(param_dict["control_points_dalpha"])[None,:,:,None]
        control_points_beta = xp.array(param_dict["control_points_dbeta"])[None,:,:,None]
        
        control_points_start = np.concatenate((control_points_start, control_points_alpha, control_points_beta), axis = 3)

    surf_init = Bezier_Surfaces(
        control_points = control_points_start,
        material = info_dict["material"],
        num_samples = info_dict["num_samples"],
        width = param_dict["width"],
        U = rot_ip[:,:,0],
        reference_config = {"x_range": param_dict["x_range"], "y_range": param_dict["y_range"]},
    )

    # rotation axis
    U_rot_m90 = get_rot_matrix_rodriguez(ks = xp.array([0,0,1]), thetas = -xp.pi/2) # to rotate the image onto the correct axis 

    theta_alpha = param_dict["theta_alpha"]

    k_alpha = U_rot_m90 @ xp.array((xp.sin(theta_alpha), xp.cos(theta_alpha), 0*xp.cos(theta_alpha)))
    k_beta = U_rot_m90 @ xp.array((xp.cos(theta_alpha), -xp.sin(theta_alpha), 0*xp.cos(theta_alpha)))

    angles_alpha_beta = info_dict["angles_alpha_beta"]

    # axis of rotation
    rotation_matrices_alpha = get_rot_matrix_rodriguez(ks = k_alpha, thetas = xp.deg2rad(xp.array(angles_alpha_beta[0])))
    rotation_matrices_beta = get_rot_matrix_rodriguez(ks = k_beta, thetas = xp.deg2rad(xp.array(angles_alpha_beta[1])))

    rotation_matrices = xp.einsum("ijl,jkl->ikl", rotation_matrices_alpha, rotation_matrices_beta)

    exp_init = Experiment.from_energy(
            energy_kev = info_dict["energy"],
            qs = info_dict["qs"],
            Surfs = surf_init,
            intensity_param = xp.array(param_dict["intensity_param"]),
            rotation_matrices = rotation_matrices,
    )
    
    return surf_init, exp_init

class OptimizerFramework:
    def __init__(self, param_dict, data_info, settings):
        """
        param_dict: dict of form {
            "param_name": {"Fixed": bool, "value": float, "bounds": (low, high)}
        }
        data: optional, e.g., images or other input used in the cost function
        """
        self.param_dict = param_dict
        self.data_info = data_info
        self.settings = settings
        self.cost_function = None
        self.optimized_result = None

    def get_bf_df(self, full_params):
            surf, exp = get_exp_surface(full_params, self.data_info)

            I_df, I_bf = exp.get_bright_field()

            if "pre_mask" in self.data_info:
                exp.I_bf = exp.I_bf*self.data_info["pre_mask"]

            I_df = I_df.transpose((3, 0, 4, 1, 2))
            
            if self.settings["projection"]:
                if self.settings["include_df"]:
                    I_bf, I_df = exp.get_bright_field_projected(
                        x_reg = self.data_info["x_reg"], 
                        y_reg = self.data_info["y_reg"], 
                        project_df = True,
                        interpolation_method = "nearest",
                        )
                else:
                    I_bf = exp.get_bright_field_projected(
                        x_reg = self.data_info["x_reg"], 
                        y_reg = self.data_info["y_reg"],
                        project_df = False,
                        interpolation_method = "nearest",
                        )
                    # print(self.data_info["x_reg"], "\n", self.data_info["y_reg"])
                    I_bf = I_bf.transpose((0,1,3,2))

                if "post_mask" in self.data_info:
                    I_bf = I_bf*self.data_info["post_mask"]
            
            xp.nan_to_num(I_bf, copy = False, nan = 0, posinf = 1, neginf = 0)
    
            return I_df, I_bf

    def define_cost_function(self):
        """
        func should take a single argument: a dictionary of parameter values,
        plus optionally access self.data through self.
        """
        if self.settings["track_cost_function"]:
            self.cost_function_history = []
        
        if "weights" not in self.data_info:
            self.weights = xp.ones(self.data_info["BF_data"].shape[0])
        else: 
            self.weights = self.data_info["weights"]
            
        def cost_function(full_params):
            I_df, I_bf = self.get_bf_df(full_params)
    
            cost_f_ans = 0
            
            if self.settings["cost_function_type"] == "cross_correlation":
                _, vals_BF = cross_correlation_registration(self.data_info["BF_data"], I_bf[0])
                vals_BF = xp.abs(vals_BF)
                
                loss_per_frame = 1000*(1-vals_BF)
                loss = float(xp.sum(loss_per_frame*self.weights/self.data_info["BF_data"].shape[0]))
                
                cost_f_ans += loss
            elif self.settings["cost_function_type"] == "abs_sq_difference":
                cost_f_ans += float(xp.sum((self.data_info["BF_data"] - I_bf[0])**2))
                
                if self.settings["include_df"]:
                    cost_f_ans += float(xp.sum(xp.abs(self.data_info["DF_data"] - I_df[0])**2))
            else:
                print("Please specify a cost function_type in settings dict")
                return
            
            if self.settings["track_cost_function"]:
                self.cost_function_history.append(cost_f_ans)

            
            return cost_f_ans
            
        self.cost_function = cost_function

    @classmethod
    
    def from_file(cls, filename, data_info=None, settings=None):
        """
        Create an OptimizerFramework instance from saved HDF5 file.
        
        Parameters:
        -----------
        filename : str
            Path to the HDF5 file containing saved optimization results
        data_info : dict, optional
            Data information dictionary. If None, will attempt to load from file
            (but experimental data may not be included in the file)
        settings : dict, optional
            Settings dictionary. If None, will load available settings from file
            
        Returns:
        --------
        OptimizerFramework
            New instance with parameters loaded from file
        """
        
        with h5py.File(filename, 'r') as f:
            # Load parameters
            param_dict = {}
            for name in f['parameters']:
                param_group = f['parameters'][name]
                param_info = {
                    'Fixed': bool(param_group.attrs['fixed'])
                }
                
                # Load value - prefer optimized if available
                if 'optimized_value' in param_group.attrs:
                    param_info['value'] = param_group.attrs['optimized_value']
                elif 'optimized_value' in param_group:
                    param_info['value'] = np.array(param_group['optimized_value'])
                elif 'initial_value' in param_group.attrs:
                    param_info['value'] = param_group.attrs['initial_value']
                elif 'initial_value' in param_group:
                    param_info['value'] = np.array(param_group['initial_value'])
                    
                # Load bounds if available
                if 'bound_lower' in param_group.attrs and 'bound_upper' in param_group.attrs:
                    lower = param_group.attrs['bound_lower']
                    upper = param_group.attrs['bound_upper']
                    param_info['bounds'] = (lower if not np.isnan(lower) else None, 
                                        upper if not np.isnan(upper) else None)
                elif 'bounds_lower' in param_group and 'bounds_upper' in param_group:
                    param_info['bounds'] = (np.array(param_group['bounds_lower']),
                                        np.array(param_group['bounds_upper']))
                    
                param_dict[name] = param_info
            
            # Load settings if not provided
            if settings is None:
                settings = {}
                if 'metadata' in f:
                    meta = f['metadata']
                    # Load known settings
                    if 'cost_function_type' in meta.attrs:
                        settings['cost_function_type'] = meta.attrs['cost_function_type']
                
                # Set some reasonable defaults
                settings.setdefault('track_cost_function', True)
                settings.setdefault('include_df', False)
                settings.setdefault('verbose', True)
                settings.setdefault('projection', True)
                
            # Create instance
            instance = cls(param_dict, data_info or {}, settings)
            
            # Populate history if available
            if 'cost_function_history' in f:
                instance.cost_function_history = list(f['cost_function_history'][:])
                
            print(f"Loaded parameters from {filename}")
            
            # Note: we cannot fully restore the optimization state
            # without the original data_info
            if data_info is None:
                print("Warning: data_info not provided. You'll need to set this before optimizing.")
            
        return instance

    def _unpack_params(self):
        free_names = []
        x0 = []
        bounds = []
        self._param_shapes = {}  # Store shapes of array-valued params

        for name, info in self.param_dict.items():
            if not info["Fixed"]:
                value = info["value"]
                if isinstance(value, np.ndarray):
                    self._param_shapes[name] = value.shape
                    shape = value.shape
                    flat_value = value.ravel()
                    x0.extend(flat_value)
                    n = flat_value.size
                    b = info.get("bounds", (None, None))
                    if isinstance(b[0], (list, tuple, np.ndarray)) and len(b[0]) == n:
                        bounds.extend(zip(b[0], b[1]))  # element-wise bounds
                    else:
                        bound_list = [b] * n
                        
                        if "control_point" in name:
                            
                            if self.settings["fix_center_cp"]:
                                bound_list[shape[0]//2*shape[0] + shape[1]//2] = [0,0]
                            
                        bounds.extend(bound_list)  # scalar bounds broadcasted
                    
                else:
                    x0.append(value)
                    bounds.append(info.get("bounds", (None, None)))
                free_names.append(name)

        return free_names, x0, bounds


    def _reconstruct_full_params(self, free_names, x_opt):
        full_params = {}
        i = 0
        for name in self.param_dict:
            if self.param_dict[name]["Fixed"]:
                full_params[name] = self.param_dict[name]["value"]
            else:
                if name in getattr(self, "_param_shapes", {}):
                    shape = self._param_shapes[name]
                    size = np.prod(shape)
                    full_params[name] = np.array(x_opt[i:i+size]).reshape(shape)
                    i += size
                else:
                    full_params[name] = x_opt[i]
                    i += 1
        return full_params


    def _wrapped_cost(self, x_opt):
        if self.cost_function is None:
            raise RuntimeError("Cost function not defined.")
        full_params = self._reconstruct_full_params(self.free_names, x_opt)
        # print(x_opt)
        return self.cost_function(full_params)

    def optimize(self, method='L-BFGS-B', tol = 1e-6, options = None):
        # if self.settings["verbose"]:
        #     print("Starting optimization with method: ", method)
        #     print("Initial parameters: ", self.param_dict)
        #     print("Initial cost: ", self.cost_function(self._reconstruct_full_params(self.param_dict)))
        #     print("Free parameters: ", self.free_names)
        #     print("Bounds: ", bounds)
        self.intermediate_results = []
        self.free_names, x0, bounds = self._unpack_params()
            # Define callback function
        def callback(x):
            # Store current parameter values
            current_params = self._reconstruct_full_params(self.free_names, x)
            
            # Store cost function value
            if hasattr(self, 'cost_function_history') and self.cost_function_history:
                current_cost = self.cost_function_history[-1]
            else:
                current_cost = self._wrapped_cost(x)
                
            # Store in intermediate results
            self.intermediate_results.append({
                'params': current_params,
                'cost': current_cost,
                'iteration': len(self.intermediate_results),
            })
            
            # Optional: Print progress
            if self.settings.get("verbose", False):
                print(f"Iteration {len(self.intermediate_results)}: cost = {current_cost}")
        
        result = minimize(self._wrapped_cost, 
                          x0, 
                          bounds = bounds, 
                          method = method, 
                          tol = tol,
                          callback = callback,
                          options = options,
                          )
        self.optimized_result = result
        return result
        
    
    def plot_optimization_path(self, ground_truth_dict = None):
        """Plot the optimization path of selected parameters."""
        if not hasattr(self, 'intermediate_results') or not self.intermediate_results:
            raise RuntimeError("No intermediate results available.")
        
        # Extract costs
        costs = [result['cost'] for result in self.intermediate_results]
        iterations = [result['iteration'] for result in self.intermediate_results]
        
        # Plot cost function history
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(iterations, costs, 'o-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost Function Value')
        ax.set_title('Optimization Progress')
        ax.grid(True)
        
        # You could also plot the evolution of specific parameters
        # For example, pick a few key parameters to track
        key_params = [p for p in self.free_names if not isinstance(self.param_dict[p]["value"], np.ndarray)]
        
        if key_params:
            fig2, axes = plt.subplots(len(key_params), 1, figsize=(10, 2*len(key_params)), sharex=True)
            if len(key_params) == 1:
                axes = [axes]
            
            for i, param in enumerate(key_params):
                values = [result['params'][param] for result in self.intermediate_results]
                if "theta" in param:
                    values = [np.rad2deg(v) for v in values]
                axes[i].plot(iterations, values, 'o-')
                axes[i].set_ylabel(param)
                axes[i].grid(True)
                if ground_truth_dict is not None and param in ground_truth_dict:
                    axes[i].axhline(ground_truth_dict[param]['value'], color='r', linestyle='--', label='Ground Truth')
                    axes[i].legend()
                
            axes[-1].set_xlabel('Iteration')
            fig2.suptitle('Parameter Evolution During Optimization')
            fig2.tight_layout()
            
        return fig

    def reset_optimization(self):
        surf, exp = self.get_result_objects()
        
        self.data_info["control_points_start_xyz"] = surf.control_points
        
        result_dict = self.get_result()
        
        for key in result_dict:
            if key in self.param_dict:
                self.param_dict[key]["value"] = result_dict[key]
            else:
                print(f"Warning: Parameter {key} not found in param_dict. Skipping reset for this parameter.")

    def setup_pyramid_one_step(self):
        surf, exp = self.get_result_objects()
        
        cp_num = self.data_info["control_points_start_xyz"].shape[1]
        
        new_cps = surf.updown_sample_cp(cp_num*2)[None]
        
        self.data_info["control_points_start_xyz"] = new_cps
        
        self.param_dict["control_points_update_x"]["value"] = np.zeros((cp_num*2, cp_num*2))
        self.param_dict["control_points_update_y"]["value"] = np.zeros((cp_num*2, cp_num*2))
        self.param_dict["control_points_update_z"]["value"] = np.zeros((cp_num*2, cp_num*2))
        self.param_dict["control_points_dalpha"]["value"] = np.zeros((cp_num*2, cp_num*2))
        self.param_dict["control_points_dbeta"]["value"] = np.zeros((cp_num*2, cp_num*2))

    def get_result(self):
        if self.optimized_result is None:
            raise RuntimeError("No optimization run yet.")
        return self._reconstruct_full_params(self.free_names, self.optimized_result.x)
    
    def get_result_objects(self):
        params = self.get_result()
        
        return get_exp_surface(params, self.data_info)
        
    
    def plot_result(self):
        params = self._reconstruct_full_params(self.free_names, self.optimized_result.x)
        I_df, I_bf = self.get_bf_df(params)
        
        num_imgs = self.data_info["BF_data"].shape[0]
        
        fig, ax = plt.subplots(num_imgs, 3, figsize = (4, 37), sharex = True, sharey = True)
        for i in range(num_imgs):
            ax[i,0].imshow(self.data_info["BF_data"][i].get(), vmin = 0, vmax = 1)
            ax[i,1].imshow(I_bf[0,i].get(), vmin = 0, vmax = 1)
            
            diff = I_bf[0,i].get() - self.data_info["BF_data"][i].get()
            diff_vmax = diff.max()
            diff_vmin = diff.min()
            vmin_true = min(diff_vmin, -diff_vmax)
            ax[i,2].imshow(I_bf[0,i].get() - self.data_info["BF_data"][i].get(), cmap = "RdBu_r", vmin = vmin_true, vmax = -vmin_true)
            
            # ax[i,0].set_ylabel(rf"$\alpha$ = {xp.rad2deg(self.data_info["angles_alpha_beta"][0,i])}, $\beta$ = {xp.rad2deg(self.data_info["angles_alpha_beta"][1,i])}")
        
        ax[0,0].set_title("Data")
        ax[0,1].set_title("Reconstruction")
        ax[0,2].set_title("Difference")
        
        
        
        fig, ax = plt.subplots()
        ax.plot(self.cost_function_history)
        
        return fig, ax
    
    def save_result(self, filename = 'optimization_results.h5'):
        """
        Save optimization results to an HDF5 file.
        
        Parameters:
        -----------
        filename : str
            Name of the output file (with .h5 extension)
        
        Returns:
        --------
        str
            Path to the saved file
        """
        
        # Get results and parameters
        if self.optimized_result is None:
            raise RuntimeError("No optimization run yet.")
            
        result_params = self.get_result()
        
        # Create file
        with h5py.File(filename, 'w') as f:
            # Save metadata
            meta = f.create_group('metadata')
            meta.attrs['timestamp'] = datetime.datetime.now().isoformat()
            meta.attrs['cost_function_type'] = self.settings.get('cost_function_type', 'unknown')
            
            # Save optimization result details
            opt_grp = f.create_group('optimization_result')
            opt_grp.attrs['success'] = bool(self.optimized_result.success)
            opt_grp.attrs['status'] = int(self.optimized_result.status)
            opt_grp.attrs['message'] = str(self.optimized_result.message)
            opt_grp.attrs['nfev'] = int(self.optimized_result.nfev)
            opt_grp.attrs['nit'] = int(self.optimized_result.nit) if hasattr(self.optimized_result, 'nit') else 0
            opt_grp.attrs['final_cost'] = float(self.optimized_result.fun)
            
            # Save parameters and their details
            params_grp = f.create_group('parameters')
            for name, info in self.param_dict.items():
                param_grp = params_grp.create_group(name)
                param_grp.attrs['fixed'] = bool(info['Fixed'])
                
                # Save bounds if available
                if 'bounds' in info:
                    if isinstance(info['bounds'], tuple) and len(info['bounds']) == 2:
                        if isinstance(info['bounds'][0], (list, tuple, np.ndarray)):
                            # Element-wise bounds for array params
                            param_grp.create_dataset('bounds_lower', data=np.array(info['bounds'][0]))
                            param_grp.create_dataset('bounds_upper', data=np.array(info['bounds'][1]))
                        else:
                            # Scalar bounds
                            param_grp.attrs['bound_lower'] = info['bounds'][0] if info['bounds'][0] is not None else np.nan
                            param_grp.attrs['bound_upper'] = info['bounds'][1] if info['bounds'][1] is not None else np.nan
                
                # Save initial and optimized values
                if isinstance(info['value'], np.ndarray):
                    param_grp.create_dataset('initial_value', data=info['value'])
                    if name in result_params:
                        param_grp.create_dataset('optimized_value', data=result_params[name])
                else:
                    param_grp.attrs['initial_value'] = info['value']
                    if name in result_params:
                        param_grp.attrs['optimized_value'] = result_params[name]
            
            # Save cost function history if available
            if hasattr(self, 'cost_function_history') and self.cost_function_history:
                f.create_dataset('cost_function_history', data=np.array(self.cost_function_history))
            
            # Save intermediate results if available
            if hasattr(self, 'intermediate_results') and self.intermediate_results:
                inter_grp = f.create_group('intermediate_results')
                inter_grp.create_dataset('iterations', 
                                        data=np.array([r['iteration'] for r in self.intermediate_results]))
                inter_grp.create_dataset('costs', 
                                        data=np.array([r['cost'] for r in self.intermediate_results]))
                
                # Save parameter evolution for scalar parameters
                params_evol_grp = inter_grp.create_group('parameters_evolution')
                for param_name in self.free_names:
                    if not isinstance(self.param_dict[param_name]["value"], np.ndarray):
                        params_evol_grp.create_dataset(
                            param_name, 
                            data=np.array([r['params'][param_name] for r in self.intermediate_results])
                        )
        
        print(f"Results saved to {os.path.abspath(filename)}")
        return os.path.abspath(filename)
