from typing import Union
import numpy as np
import cupy as xp
from .surfaces import Surfaces
import matplotlib
import matplotlib.pyplot as plt
from numpy import ndarray
from .utils.math_utils import get_struct_factor, get_rot_matrix_rodriguez


class Experiment:
    """
    Simulates TEM experiments with bent crystals.

    Parameters:
        k_i (ndarray): Incident electron beam wavevector.
        qs (ndarray): Array of reciprocal lattice vectors to consider.
        Surfs (type[Surfaces]): Surface object containing crystal geometry.
        intensity_param (float): Scaling parameter for diffraction intensity.
        rotation_axis (ndarray): Axis about which the sample is rotated.
        rotation_angles (ndarray): Array of rotation angles in radians.
    """
    @classmethod
    def from_energy(cls, 
                   energy_kev: float,
                   qs: Union[np.ndarray, xp.ndarray],
                   Surfs: 'Surfaces',
                   intensity_param: float,
                   rotation_axis: Union[np.ndarray, xp.ndarray] = None,
                   rotation_angles: Union[np.ndarray, xp.ndarray] = None,
                   rotation_matrices: Union[np.ndarray, xp.ndarray] = None,
    ) -> 'Experiment':
        """
        Create Experiment instance from electron energy.

        Parameters:
            energy_kev: Electron energy in keV
            qs: Array of reciprocal lattice vectors
            Surfs: Surface object containing crystal geometry
            intensity_param: Scaling parameter for diffraction intensity
            rotation_axis: Axis about which the sample is rotated
            rotation_angles: Array of rotation angles in radians
        """
        k_i = cls._energy_to_wavevector(energy_kev)
        return cls(k_i=k_i, 
                   qs=qs, 
                   Surfs=Surfs, 
                  intensity_param=intensity_param,
                  rotation_axis=rotation_axis, 
                  rotation_angles=rotation_angles,
                  rotation_matrices=rotation_matrices)

    @staticmethod
    def _energy_to_wavevector(energy_kev: float) -> Union[np.ndarray, xp.ndarray]:
        """Convert electron energy to wavevector."""
        E_electron_joule = energy_kev * 1e3 * 1.60218e-19
        m_electron = 9.1093837e-31  # kg
        hbar = 1.054571817e-34  # J s
        k_magnitude = xp.sqrt(2 * m_electron * E_electron_joule) / hbar / 1e10
        return k_magnitude * xp.array([0, 0, -1])  # Default direction -z

    def __init__(
        self,
        k_i: ndarray,
        qs: ndarray,
        Surfs: type[Surfaces],
        intensity_param: float,
        rotation_axis: ndarray = None,
        rotation_angles: ndarray = None,
        rotation_matrices: ndarray = None,
    ):
        """
        
        """

        self.k_i = k_i
        
        self.set_surfs(Surfs)
        
        self.set_qs(qs)
        
        self.intensity_param = intensity_param

        if rotation_matrices is None and rotation_axis is None and rotation_angles is None:
            raise ValueError("rotation_matrices must be provided if rotation_axis and rotation_angles are not provided")
        elif rotation_matrices is not None and rotation_axis is None and rotation_angles is None:
            self.set_rotation_matrices(rotation_matrices)
        elif rotation_matrices is None and rotation_axis is not None and rotation_angles is not None:
            self.set_rotation_axis(rotation_axis)
            self.set_rotation_angles(rotation_angles)
        else:
            raise ValueError("rotation_matrices must be provided if rotation_axis and rotation_angles are not provided")
    
    def set_surfs(
        self,
        Surfs,
    ):
        self.Surfs = Surfs
    
    def set_qs(
        self,
        qs,
    ):
        try:
            self.qs_np = np.array(qs)
            self.qs_xp = xp.array(qs)
        except:
            self.qs_np = qs.get()
            self.qs_xp = qs
            
        my_qs = np.transpose(self.qs_np, [len(self.qs_np.shape)-1, *np.arange(len(self.qs_np.shape)-1)])
        self.structure_factor = get_struct_factor(self.Surfs.material, my_qs)
        
    def set_rotation_axis(
        self,
        rotation_axis: ndarray,
    ):
        
        self.rotation_axis = rotation_axis

        
    def set_rotation_angles(
        self,
        rotation_angles: ndarray,
    ):
        
        self.rotation_angles = rotation_angles
        
        self.rotation_matrices = get_rot_matrix_rodriguez(self.rotation_axis, 
                                                          self.rotation_angles)

        self.k_is = xp.tensordot(self.rotation_matrices, 
                                 self.k_i, 
                                 axes = [[1],[0]])
    
    def set_rotation_matrices(
        self,
        rotation_matrices: ndarray,
    ):
        self.rotation_matrices = rotation_matrices

        self.k_is = xp.tensordot(self.rotation_matrices, 
                                 self.k_i, 
                                 axes = [[1],[0]])

        # TODO: set rotation axis and angles too
        
        
        
    def get_dark_field(
        self
    ):
        Q = np.tensordot(self.Surfs.UB, 
                         self.qs_xp, 
                         axes = [[1],[-1]])
        
        k_out = Q[:,None,:,:,:,:] + self.k_is[:, :, None, None, None, None]
        
        s = xp.abs(xp.linalg.norm(k_out, axis = 0) - xp.linalg.norm(self.k_i))
        
        # width = 2*xp.pi/4/700
        self.I_df = xp.exp(-s**2/self.Surfs.width**2)
        return self.I_df
    
    def get_bright_field(
        self
    ):
        I_df = self.get_dark_field()
        
        I = xp.abs(xp.array(self.structure_factor))**2
        
        self.I_bf = (1 - xp.sum(self.intensity_param*I_df*I, axis = -1)).transpose((3,0,1,2))

        return I_df, self.I_bf
    
    def mask_intensity(
        self,
        mask,
    ):
        self.I_bf[:,:,mask] = 0
        self.I_df[:,mask] = 0
    
    def get_bright_field_projected(
        self,
        x_reg,
        y_reg,
        interpolation_method = "nearest",
        project_df = False,
    ):
        # self.get_bright_field()

        num_angles = self.rotation_matrices.shape[2]

        x_y_R = self.rotation_matrices.transpose((1,0,2))[:,0:2] #(Transposed because we want to rotate by -theta) (columns of U^T)
        
        projected_in_plane_cs = xp.einsum("ijk,limn->ljkmn", x_y_R, self.Surfs.R)

        x_rot = projected_in_plane_cs[:,0]
        y_rot = projected_in_plane_cs[:,1]

        x_min = x_reg.min()
        x_max = x_reg.max()
        y_min = y_reg.min()
        y_max = y_reg.max()

        grid_size = x_reg.shape[0]
        
        # Find the nearest grid points and calculate weights for interpolation
        x_idx = (grid_size - 1) * (x_rot - x_min) / (x_max - x_min)
        y_idx = (grid_size - 1) * (y_rot - y_min) / (y_max - y_min)

        x0 = xp.floor(x_idx).astype(int)
        y0 = xp.floor(y_idx).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Clip indices to be within the valid range
        x0 = xp.clip(x0, 0, grid_size - 1)
        y0 = xp.clip(y0, 0, grid_size - 1)
        x1 = xp.clip(x1, 0, grid_size - 1)
        y1 = xp.clip(y1, 0, grid_size - 1)
        
        # Step 5: Apply the weights to interpolate I at the grid points
        I_bf_projected = xp.zeros((self.Surfs.R.shape[0], num_angles, grid_size, grid_size))
        normalization_bf = xp.zeros((self.Surfs.R.shape[0], num_angles, grid_size, grid_size))
        
        ind_surfs = xp.tile(xp.arange(self.Surfs.R.shape[0]), (*self.Surfs.R.shape[-2:], num_angles)).reshape((*self.Surfs.R.shape[-2:], num_angles, self.Surfs.R.shape[0])).T
        ind_angles = xp.tile(xp.arange(num_angles), (*self.Surfs.R.shape[-2:], self.Surfs.R.shape[0])).reshape((*self.Surfs.R.shape[-2:], num_angles, self.Surfs.R.shape[0])).T

        if interpolation_method == "nearest":
            xp.add.at(I_bf_projected, (ind_surfs, ind_angles, y0, x0), self.I_bf)
            xp.add.at(normalization_bf, (ind_surfs, ind_angles, y0, x0), 1)
        elif interpolation_method == "bilinear":
            #compute weights
            wa = (x1 - x_idx) * (y1 - y_idx)
            wb = (x1 - x_idx) * (y_idx - y0)
            wc = (x_idx - x0) * (y1 - y_idx)
            wd = (x_idx - x0) * (y_idx - y0)
        
            xp.add.at(I_bf_projected, (ind_surfs, ind_angles, y0, x0),self.I_bf*wa) # xp.add.at(I_projected, (y0, x0), wa * I)
            xp.add.at(I_bf_projected, (ind_surfs, ind_angles, y1, x0),self.I_bf*wb) # xp.add.at(I_projected, (y0, x0), wa * I)
            xp.add.at(I_bf_projected, (ind_surfs, ind_angles, y0, x1),self.I_bf*wc) # xp.add.at(I_projected, (y0, x0), wa * I)
            xp.add.at(I_bf_projected, (ind_surfs, ind_angles, y1, x1),self.I_bf*wd) # xp.add.at(I_projected, (y0, x0), wa * I)

            xp.add.at(normalization_bf, (ind_surfs, ind_angles, y0, x0), 1*wa)
            xp.add.at(normalization_bf, (ind_surfs, ind_angles, y1, x0), 1*wb)
            xp.add.at(normalization_bf, (ind_surfs, ind_angles, y0, x1), 1*wc)
            xp.add.at(normalization_bf, (ind_surfs, ind_angles, y1, x1), 1*wd)
        else:
            print("Choose either 'nearest' or 'bilinear' interpolation")
        
        I_bf_projected = I_bf_projected/normalization_bf

        # df projection
        if project_df:
            num_surfs = self.Surfs.R.shape[0]
            num_qs = self.qs_np.shape[0]
            image_num = self.Surfs.R.shape[-2:]
            
            I_df_projected = xp.zeros((num_surfs, num_angles, num_qs, grid_size, grid_size))
            normalization_df = xp.zeros((num_surfs, num_angles, num_qs, grid_size, grid_size))
        
            
            # ind_surfs_df = xp.tile(
            #     xp.arange(num_surfs), 
            #     (*image_num, num_qs, num_angles)
            # ).reshape((*image_num,  num_qs, num_angles, num_surfs)).T
            
            # ind_angles_df = xp.tile(
            #     xp.arange(num_angles), 
            #     (*image_num, num_qs, num_surfs)
            # ).reshape((*image_num, num_qs, num_angles, num_surfs)).T
            
            # ind_qs_df = xp.tile(
            #     xp.arange(self.qs_np.shape[0]), 
            #     (*image_num, num_angles, num_surfs)
            # ).reshape((*image_num,  num_qs, num_angles, num_surfs)).T
            
            ind_surfs_df, ind_angles_df, ind_qs_df = xp.meshgrid(
                xp.arange(num_surfs),   # Surface index
                xp.arange(num_angles),  # Angle index
                xp.arange(num_qs),      # Q index
                indexing="ij"           # Ensures proper shape alignment
            )  # Output shape: (num_surfs, num_angles, num_qs)

            # Expand dimensions to match (num_surfs, num_angles, num_qs, grid_size, grid_size)
            ind_surfs_df = ind_surfs_df[..., None, None]  # Shape: (num_surfs, num_angles, num_qs, 1, 1)
            ind_angles_df = ind_angles_df[..., None, None]
            ind_qs_df = ind_qs_df[..., None, None]

            # Broadcast to final shape: (num_surfs, num_angles, num_qs, grid_size, grid_size)
            ind_surfs_df = xp.broadcast_to(ind_surfs_df, (num_surfs, num_angles, num_qs, image_num[0], image_num[0]))
            ind_angles_df = xp.broadcast_to(ind_angles_df, (num_surfs, num_angles, num_qs, image_num[0], image_num[0]))
            ind_qs_df = xp.broadcast_to(ind_qs_df, (num_surfs, num_angles, num_qs, image_num[0], image_num[0]))
 
            y0_df = xp.repeat(y0[:,:,None,:,:,], num_qs, axis = 2)
            x0_df = xp.repeat(x0[:,:,None,:,:,], num_qs, axis = 2)
            y1_df = xp.repeat(y1[:,:,None,:,:,], num_qs, axis = 2)
            x1_df = xp.repeat(x1[:,:,None,:,:,], num_qs, axis = 2)
            

            if interpolation_method == "nearest":
                xp.add.at(I_df_projected, (ind_surfs_df, ind_angles_df, ind_qs_df, y0_df, x0_df), self.I_df.transpose((3, 0, 4, 1, 2)))
                xp.add.at(normalization_df, (ind_surfs_df, ind_angles_df, ind_qs_df, y0_df, x0_df), 1)
                
            elif interpolation_method == "bilinear":
                xp.add.at(I_df_projected, (ind_surfs_df, ind_angles_df, ind_qs_df, y0_df, x0_df), self.I_df.transpose((3, 0, 4, 1, 2))* xp.repeat(wa[:,:,None,:,:], self.qs_np.shape[0], axis = 2)) 
                xp.add.at(I_df_projected, (ind_surfs_df, ind_angles_df, ind_qs_df, y1_df, x0_df), self.I_df.transpose((3, 0, 4, 1, 2))* xp.repeat(wa[:,:,None,:,:], self.qs_np.shape[0], axis = 2)) 
                xp.add.at(I_df_projected, (ind_surfs_df, ind_angles_df, ind_qs_df, y0_df, x1_df), self.I_df.transpose((3, 0, 4, 1, 2))* xp.repeat(wa[:,:,None,:,:], self.qs_np.shape[0], axis = 2)) 
                xp.add.at(I_df_projected, (ind_surfs_df, ind_angles_df, ind_qs_df, y1_df, x1_df), self.I_df.transpose((3, 0, 4, 1, 2))* xp.repeat(wa[:,:,None,:,:], self.qs_np.shape[0], axis = 2)) 
                
                xp.add.at(normalization_df, (ind_surfs_df, ind_angles_df, ind_qs_df, y0_df, x0_df), xp.repeat(wa[:,:,None,:,:], self.qs_np.shape[0], axis = 2))
                xp.add.at(normalization_df, (ind_surfs_df, ind_angles_df, ind_qs_df, y1_df, x0_df), xp.repeat(wb[:,:,None,:,:], self.qs_np.shape[0], axis = 2))
                xp.add.at(normalization_df, (ind_surfs_df, ind_angles_df, ind_qs_df, y0_df, x1_df), xp.repeat(wc[:,:,None,:,:], self.qs_np.shape[0], axis = 2))
                xp.add.at(normalization_df, (ind_surfs_df, ind_angles_df, ind_qs_df, y1_df, x1_df), xp.repeat(wd[:,:,None,:,:], self.qs_np.shape[0], axis = 2))
            else:
                print("Choose either 'nearest' or 'bilinear' interpolation")
                
            I_df_projected = I_df_projected/normalization_df
            
            return I_bf_projected, I_df_projected
        else:
            return I_bf_projected
    
    # def get_dark_field_projected(
    #     self,
    #     x_reg,
    #     y_reg,
    #     interpolation_method = "nearest",
    #     project_df = False,
    # ):
    #     self.get_bright_field()

    #     num_angles = self.rotation_matrices.shape[2]

    #     x_y_R = self.rotation_matrices.transpose((1,0,2))[:,0:2] #(Transposed because we want to rotate by -theta) (columns of U^T)
        
    #     projected_in_plane_cs = xp.einsum("ijk,limn->ljkmn", x_y_R, self.Surfs.R)

    #     x_rot = projected_in_plane_cs[:,0]
    #     y_rot = projected_in_plane_cs[:,1]

    #     x_min = x_reg.min()
    #     x_max = x_reg.max()
    #     y_min = y_reg.min()
    #     y_max = y_reg.max()

    #     grid_size = x_reg.shape[0]
        
    #     # Find the nearest grid points and calculate weights for interpolation
    #     x_idx = (grid_size - 1) * (x_rot - x_min) / (x_max - x_min)
    #     y_idx = (grid_size - 1) * (y_rot - y_min) / (y_max - y_min)

    #     x0 = xp.floor(x_idx).astype(int)
    #     y0 = xp.floor(y_idx).astype(int)
    #     x1 = x0 + 1
    #     y1 = y0 + 1
        
    #     # Clip indices to be within the valid range
    #     x0 = xp.clip(x0, 0, grid_size - 1)
    #     y0 = xp.clip(y0, 0, grid_size - 1)
    #     x1 = xp.clip(x1, 0, grid_size - 1)
    #     y1 = xp.clip(y1, 0, grid_size - 1)
        
    #     # Step 5: Apply the weights to interpolate I at the grid points
    #     I_bf_projected = xp.zeros((self.Surfs.R.shape[0], num_angles, grid_size, grid_size))
    #     normalization_bf = xp.zeros((self.Surfs.R.shape[0], num_angles, grid_size, grid_size))
        
    #     ind_surfs = xp.tile(xp.arange(self.Surfs.R.shape[0]), (*self.Surfs.R.shape[-2:], num_angles)).reshape((*self.Surfs.R.shape[-2:], num_angles, self.Surfs.R.shape[0])).T
    #     ind_angles = xp.tile(xp.arange(num_angles), (*self.Surfs.R.shape[-2:], self.Surfs.R.shape[0])).reshape((*self.Surfs.R.shape[-2:], num_angles, self.Surfs.R.shape[0])).T

    #     if interpolation_method == "nearest":
    #         xp.add.at(I_bf_projected, (ind_surfs, ind_angles, y0, x0), self.I_bf)
    #         xp.add.at(normalization_bf, (ind_surfs, ind_angles, y0, x0), 1)
    #     elif interpolation_method == "bilinear":
    #         #compute weights
    #         wa = (x1 - x_idx) * (y1 - y_idx)
    #         wb = (x1 - x_idx) * (y_idx - y0)
    #         wc = (x_idx - x0) * (y1 - y_idx)
    #         wd = (x_idx - x0) * (y_idx - y0)
        
    #         xp.add.at(I_bf_projected, (ind_surfs, ind_angles, y0, x0),self.I_bf*wa) # xp.add.at(I_projected, (y0, x0), wa * I)
    #         xp.add.at(I_bf_projected, (ind_surfs, ind_angles, y1, x0),self.I_bf*wb) # xp.add.at(I_projected, (y0, x0), wa * I)
    #         xp.add.at(I_bf_projected, (ind_surfs, ind_angles, y0, x1),self.I_bf*wc) # xp.add.at(I_projected, (y0, x0), wa * I)
    #         xp.add.at(I_bf_projected, (ind_surfs, ind_angles, y1, x1),self.I_bf*wd) # xp.add.at(I_projected, (y0, x0), wa * I)

    #         xp.add.at(normalization_bf, (ind_surfs, ind_angles, y0, x0), 1*wa)
    #         xp.add.at(normalization_bf, (ind_surfs, ind_angles, y1, x0), 1*wb)
    #         xp.add.at(normalization_bf, (ind_surfs, ind_angles, y0, x1), 1*wc)
    #         xp.add.at(normalization_bf, (ind_surfs, ind_angles, y1, x1), 1*wd)
    #     else:
    #         print("Choose either 'nearest' or 'bilinear' interpolation")
        
    #     I_bf_projected = I_bf_projected/normalization_bf

    #     # df projection
    #     if project_df:
    #         num_surfs = self.Surfs.R.shape[0]
    #         num_qs = self.qs_np.shape[0]
    #         image_num = self.Surfs.R.shape[-2:]
            
    #         I_df_projected = xp.zeros((num_surfs, num_angles, num_qs, grid_size, grid_size))
    #         normalization_df = xp.zeros((num_surfs, num_angles, num_qs, grid_size, grid_size))
        
            
    #         # ind_surfs_df = xp.tile(
    #         #     xp.arange(num_surfs), 
    #         #     (*image_num, num_qs, num_angles)
    #         # ).reshape((*image_num,  num_qs, num_angles, num_surfs)).T
            
    #         # ind_angles_df = xp.tile(
    #         #     xp.arange(num_angles), 
    #         #     (*image_num, num_qs, num_surfs)
    #         # ).reshape((*image_num, num_qs, num_angles, num_surfs)).T
            
    #         # ind_qs_df = xp.tile(
    #         #     xp.arange(self.qs_np.shape[0]), 
    #         #     (*image_num, num_angles, num_surfs)
    #         # ).reshape((*image_num,  num_qs, num_angles, num_surfs)).T
            
    #         ind_surfs_df, ind_angles_df, ind_qs_df = xp.meshgrid(
    #             xp.arange(num_surfs),   # Surface index
    #             xp.arange(num_angles),  # Angle index
    #             xp.arange(num_qs),      # Q index
    #             indexing="ij"           # Ensures proper shape alignment
    #         )  # Output shape: (num_surfs, num_angles, num_qs)

    #         # Expand dimensions to match (num_surfs, num_angles, num_qs, grid_size, grid_size)
    #         ind_surfs_df = ind_surfs_df[..., None, None]  # Shape: (num_surfs, num_angles, num_qs, 1, 1)
    #         ind_angles_df = ind_angles_df[..., None, None]
    #         ind_qs_df = ind_qs_df[..., None, None]

    #         # Broadcast to final shape: (num_surfs, num_angles, num_qs, grid_size, grid_size)
    #         ind_surfs_df = xp.broadcast_to(ind_surfs_df, (num_surfs, num_angles, num_qs, image_num[0], image_num[0]))
    #         ind_angles_df = xp.broadcast_to(ind_angles_df, (num_surfs, num_angles, num_qs, image_num[0], image_num[0]))
    #         ind_qs_df = xp.broadcast_to(ind_qs_df, (num_surfs, num_angles, num_qs, image_num[0], image_num[0]))
 
    #         y0_df = xp.repeat(y0[:,:,None,:,:,], num_qs, axis = 2)
    #         x0_df = xp.repeat(x0[:,:,None,:,:,], num_qs, axis = 2)
    #         y1_df = xp.repeat(y1[:,:,None,:,:,], num_qs, axis = 2)
    #         x1_df = xp.repeat(x1[:,:,None,:,:,], num_qs, axis = 2)
            
            
    #         if interpolation_method == "nearest":
    #             xp.add.at(I_df_projected, (ind_surfs_df, ind_angles_df, ind_qs_df, y0_df, x0_df), self.I_df.transpose((3, 0, 4, 1, 2)))
    #             xp.add.at(normalization_df, (ind_surfs_df, ind_angles_df, ind_qs_df, y0_df, x0_df), 1)
                
    #         elif interpolation_method == "bilinear":
    #             xp.add.at(I_df_projected, (ind_surfs_df, ind_angles_df, ind_qs_df, y0_df, x0_df), self.I_df.transpose((3, 0, 4, 1, 2))* xp.repeat(wa[:,:,None,:,:], self.qs_np.shape[0], axis = 2)) 
    #             xp.add.at(I_df_projected, (ind_surfs_df, ind_angles_df, ind_qs_df, y1_df, x0_df), self.I_df.transpose((3, 0, 4, 1, 2))* xp.repeat(wa[:,:,None,:,:], self.qs_np.shape[0], axis = 2)) 
    #             xp.add.at(I_df_projected, (ind_surfs_df, ind_angles_df, ind_qs_df, y0_df, x1_df), self.I_df.transpose((3, 0, 4, 1, 2))* xp.repeat(wa[:,:,None,:,:], self.qs_np.shape[0], axis = 2)) 
    #             xp.add.at(I_df_projected, (ind_surfs_df, ind_angles_df, ind_qs_df, y1_df, x1_df), self.I_df.transpose((3, 0, 4, 1, 2))* xp.repeat(wa[:,:,None,:,:], self.qs_np.shape[0], axis = 2)) 
                
    #             xp.add.at(normalization_df, (ind_surfs_df, ind_angles_df, ind_qs_df, y0_df, x0_df), xp.repeat(wa[:,:,None,:,:], self.qs_np.shape[0], axis = 2))
    #             xp.add.at(normalization_df, (ind_surfs_df, ind_angles_df, ind_qs_df, y1_df, x0_df), xp.repeat(wb[:,:,None,:,:], self.qs_np.shape[0], axis = 2))
    #             xp.add.at(normalization_df, (ind_surfs_df, ind_angles_df, ind_qs_df, y0_df, x1_df), xp.repeat(wc[:,:,None,:,:], self.qs_np.shape[0], axis = 2))
    #             xp.add.at(normalization_df, (ind_surfs_df, ind_angles_df, ind_qs_df, y1_df, x1_df), xp.repeat(wd[:,:,None,:,:], self.qs_np.shape[0], axis = 2))
    #         else:
    #             print("Choose either 'nearest' or 'bilinear' interpolation")
                
    #         I_df_projected = I_df_projected/normalization_df
            
    #         return I_bf_projected, I_df_projected
    #     else:
    #         return I_bf_projected

    def plot_df_phase_map(
        self,
        i_R = 0,
        i_theta = 0,
    ):
        # (78, 128, 128, 1, 8)
        my_I_df = self.I_df[i_theta,:,:, i_theta,:]
        phase_factor = self.qs_np[:,0] + 1j*self.qs_np[:,1]

        df_phase_map = xp.zeros((my_I_df.shape[0], my_I_df.shape[1]), dtype = xp.complex128)

        for i in range(my_I_df.shape[2]):
            my_I_df_phase = my_I_df[:,:,i] *phase_factor[i]

            df_phase_map += my_I_df_phase
        
        amplitude = np.abs(df_phase_map)
        phase = np.angle(df_phase_map)  # radians, from -pi to pi

        # Normalize amplitude (optional, for better contrast)
        amplitude = amplitude / np.max(amplitude)

        # Build HSV image
        hue = (phase + np.pi) / (2 * np.pi)  # map from [−π,π] to [0,1]
        saturation = np.ones_like(hue)
        value = amplitude

        hsv_image = np.stack([hue, saturation, value], axis=-1)
        rgb_image = matplotlib.colors.hsv_to_rgb(hsv_image)

        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(rgb_image)
        ax.set_title("Dark Field Phase Plot")

        return fig, ax

    def plot_bf(
        self,
        i_R = 0,
    ):
        num_theta = self.I_bf.shape[1]

        n_row = int(np.ceil(num_theta/10)) 
        
        fig, ax = plt.subplots(n_row, 10, figsize = (20,10))
        ax = np.array(ax).flatten()
        for i in range(num_theta):
            ax[i].imshow(self.I_bf[i_R, i].get(), vmax = 1, vmin = 0)
            ax[i].set_title(fr"$\theta$ = {np.round(self.rotation_angles[i]*180/np.pi, 2)}$^\circ$")
    
class Apertures:
    def __init__(
        self,
        masks_flat, # n x img_x x img_y
        z_height, # n
        rotation_matrices, # num_angles x 3 x 3
    ): 
        self.masks_flat = masks_flat # n x img_x x img_y
        self.z_height = z_height # n
        self.rotation_matrices = rotation_matrices
        
class Detector:
    def __init__(
        self,
        R_pixels, #position of each pixe, 3 x pix_x x pix_y
        rotation_matrices, # rotation matrix for the sample's rotation angles
    ):
        self.R_pixels = R_pixels
        self.rotation_matrices = rotation_matrices

    
        
        
        
        
        