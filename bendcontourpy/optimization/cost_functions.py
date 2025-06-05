import cupy as xp
import numpy as np

def define_cost_function_surface(bright_field_images,
                                 dark_field_images,
                                 Experiment, 
                                 weight_df = 1,
                                 weight_bf = 1
):
    """
    Defines the cost function for surface optimization.

    Parameters:
        bright_field_images (ndarray): Array of bright field images.
        dark_field_images (ndarray): Array of dark field images.
    """
    def cost_f(Surfaces):
        Experiment.set_surfs(Surfaces)
        
        I_df, I_bf = Experiment.get_bright_field()
        
        dif_bf = I_bf - bright_field_images
        dif_df = I_df - dark_field_images
        
        return 1000*weight_bf*xp.sum(dif_bf**2, axis = (1,2,3))/dif_bf.size + 1000*weight_df*xp.sum(dif_df**2, axis = (0,1,2,4))/dif_df.size
    return cost_f

def define_cost_function_R(bright_field_images,
                           dark_field_images, 
                           Experiment, 
                           weight_df = 1, 
                           weight_bf = 1
):
    """
    Defines the cost function for R optimization.

    Parameters:
        bright_field_images (ndarray): Array of bright field images.
        dark_field_images (ndarray): Array of dark field images.
    """
    cost_f_surface = define_cost_function_surface(bright_field_images,
                                                  dark_field_images, 
                                                  Experiment, 
                                                  weight_df = weight_df, 
                                                  weight_bf = weight_bf,
    )
    
    def cost_f_R(R):
        Experiment.Surfs.set_surface(R)
        return cost_f_surface(Experiment.Surfs)
    
    return cost_f_R

def define_cost_function_control_points(bright_field_images,
                                        dark_field_images, 
                                        Experiment, 
                                        weight_df = 1, 
                                        weight_bf = 1,
):
    """
    Defines the cost function for control point optimization.

    Parameters:
        bright_field_images (ndarray): Array of bright field images.
        dark_field_images (ndarray): Array of dark field images.
    """
    cost_f_surface = define_cost_function_surface(bright_field_images,
                                                  dark_field_images, 
                                                  Experiment, 
                                                  weight_df = weight_df, 
                                                  weight_bf = weight_bf,
    )
    
    def cost_f_cp(cp_list):
        Experiment.Surfs.set_control_points_list(cp_list)
        return cost_f_surface(Experiment.Surfs)
    
    return cost_f_cp