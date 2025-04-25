import numpy as np
import cupy as xp
from ..utils.math_utils import get_rot_matrix_rodriguez, bezier_basis_change
from ..surfaces import bezier_surface



class strain_free_solver:

    @classmethod
    def from_energy(cls, 
                    df_binary_images, 
                    qs, 
                    material,
                    energy_kev,
                    rotation_matrices,
                    dx,
                    dy,
                    U):
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
        return cls(k_i0=k_i, df_binary_images=df_binary_images, 
                  qs = qs, 
                  material=material,
                  rotation_matrices=rotation_matrices, 
                  dx=dx,
                  dy = dy,
                  U = U)

    @staticmethod
    def _energy_to_wavevector(energy_kev: float):
        """Convert electron energy to wavevector."""
        E_electron_joule = energy_kev * 1e3 * 1.60218e-19
        m_electron = 9.1093837e-31  # kg
        hbar = 1.054571817e-34  # J s
        k_magnitude = xp.sqrt(2 * m_electron * E_electron_joule) / hbar / 1e10
        return k_magnitude * xp.array([0, 0, -1])  # Default direction -z

    def __init__(
            self, 
            df_binary_images, 
            qs, 
            material,
            k_i0,
            rotation_matrices,
            dx,
            dy,
            U, 
        ):

        self.df_binary_images = df_binary_images
        self.qs_hkl = qs
        self.k_i0 = k_i0
        self.rotation_matrices = rotation_matrices
        self.dx = dx
        self.dy = dy
        self.U = U
        self.define_material(material)
    
    def define_material(self, 
            material
        ):

        self.material = material

        B_real = xp.array((self.material.a1, 
                            self.material.a2, 
                            self.material.a3))
        
        UB_real = self.U @ B_real
        
        self.UB = 2*xp.pi*xp.linalg.inv(UB_real)

        self.qs_xyz_plane = xp.einsum('ij, kj -> ik', self.qs_hkl, self.UB)

        qs_xyz_plane_norm = xp.linalg.norm(self.qs_xyz_plane, axis = 1)

        optimal_rot_thetas = xp.arccos((2*xp.linalg.norm(self.k_i0)**2 - qs_xyz_plane_norm**2)/(2*xp.linalg.norm(self.k_i0)**2))/2

        rot_around_vectors = xp.cross( self.qs_xyz_plane, xp.array([0,0,1])[None,:], axis = 1)
        rot_around_vectors = (rot_around_vectors.T/xp.linalg.norm(rot_around_vectors, axis = 1)).T

        R_rot_mat = get_rot_matrix_rodriguez(ks = rot_around_vectors.T, thetas = optimal_rot_thetas)

        self.R_diffs = xp.einsum("ijk,kj->ki", R_rot_mat, self.qs_xyz_plane)
        self.R_diffs_rot = xp.einsum("kj,jil->kli", self.R_diffs, self.rotation_matrices.transpose((1,0,2)))
    

    def _build_constraints_BC_one_order(
            self, 
            df_binary_image,
            R_diff,
        ):
        num_constraints = int(df_binary_image.sum())
        
        df_image_inds = xp.arange(df_binary_image.size)
        df_image_inds_2d = xp.reshape(df_image_inds, df_binary_image.shape)
        
        constraint_array = xp.zeros((num_constraints,df_binary_image.size))
        
        image_shape = df_binary_image.shape
        
        b = [] #rhs
        for i in range(num_constraints):
            init_array = xp.zeros(df_binary_image.size)

            df_img_coord = df_image_inds_2d[df_binary_image][i]
            x_i, y_i = np.unravel_index(df_img_coord, df_binary_image.shape)
            
            # Deal with boundary conditions
            onboundary = x_i == image_shape[0] - 1  or x_i  == 0 or y_i == image_shape[0]-1 or y_i == 0
            
            if not onboundary:
                init_array[df_image_inds_2d[x_i+1, y_i]] += -1/(2*self.dx)*R_diff[0]
                init_array[df_image_inds_2d[x_i-1, y_i]] += 1/(2*self.dx)*R_diff[0]
                init_array[df_image_inds_2d[x_i, y_i+1]] += -1/(2*self.dy)*R_diff[1]
                init_array[df_image_inds_2d[x_i, y_i-1]] += 1/(2*self.dy)*R_diff[1]
            # else:
            #     if x_i == 0:
            #         init_array[df_image_inds_2d[x_i+1, y_i]] += -1/(dx)*R_diff[0]
            #         init_array[df_image_inds_2d[x_i, y_i]] += 1/(dx)*R_diff[0]
            #     if x_i == (image_shape[0]-1):
            #         init_array[df_image_inds_2d[x_i, y_i]] += -1/(dx)*R_diff[0]
            #         init_array[df_image_inds_2d[x_i-1, y_i]] += 1/(dx)*R_diff[0]
            #     else:
            #         init_array[df_image_inds_2d[x_i+1, y_i]] += -1/(2*dx)*R_diff[0]
            #         init_array[df_image_inds_2d[x_i-1, y_i]] += 1/(2*dx)*R_diff[0]
                
            #     if y_i == 0:
            #         init_array[df_image_inds_2d[x_i, y_i+1]] += -1/(dy)*R_diff[1]
            #         init_array[df_image_inds_2d[x_i, y_i]] += 1/(dy)*R_diff[1]
            #     if y_i == (image_shape[0]-1):
            #         init_array[df_image_inds_2d[x_i, y_i]] += -1/(dy)*R_diff[1]
            #         init_array[df_image_inds_2d[x_i, y_i-1]] += 1/(dy)*R_diff[1]
            #     else:
            #         init_array[df_image_inds_2d[x_i, y_i+1]] += -1/(2*dy)*R_diff[1]
            #         init_array[df_image_inds_2d[x_i, y_i-1]] += 1/(2*dy)*R_diff[1]
            
            b.append(-R_diff[2])
            constraint_array[i] = init_array
        return constraint_array, xp.array(b)

    def _build_constraints_BC_all(
            self,
    ):
        my_constraints = []
        my_b = []
        for m in range(self.df_binary_images.shape[0]): # angles
            for j in range(self.qs_hkl.shape[0]): # bc orders
                constraints, b = self._build_constraints_BC_one_order(self.df_binary_images[m,:,:,j], self.R_diffs_rot[j, m])
                my_constraints.append(constraints)
                my_b.append(b)
        my_bs_combined = my_b[0]
        my_constraints_combined = my_constraints[0]

        for i in range(len(my_b)-1):
            my_constraints_combined = xp.vstack((my_constraints_combined, my_constraints[i+1]))
            my_bs_combined = xp.concatenate((my_bs_combined, my_b[i+1]))

        return xp.array(my_constraints_combined), xp.array(my_bs_combined)

    def _build_constraints_reg(
            self,
            image_shape,
        ):
        constraints_total = []
        b = []
        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                init_array = xp.zeros((image_shape[0], image_shape[1]))
                
                my_filter = xp.zeros((3,3))
                
                onboundary = (i == image_shape[0]-1 or i == 0 or j == image_shape[0]-1 or j == 0)
                
                if not onboundary:
                    my_filter = xp.array(((0, 1,  0), 
                                        (1, -4, 1), 
                                        (0, 1,  0)))
                
                
                
                if not onboundary:
                    init_array[(i-1):(i+2), (j-1):(j+2)] += my_filter
                    
                if onboundary:
                    if i == 0:
                        stencil_vertical = xp.array(((-1,0,0), 
                                                    (0,0,0), 
                                                    (1, 0, 0)))
                        i_min = 0
                        i_max = 3
                    elif i == image_shape[0]-1:
                        stencil_vertical = xp.array(((0,0,-1), 
                                                    (0,0,0), 
                                                    (0,0,1)))
                        i_max = image_shape[0]
                        i_min = image_shape[0]-3
                    else: # in the middle of i
                        stencil_vertical = xp.array(((0,1,0), 
                                                    (0,-2,0), 
                                                    (0,1,0)))
                        i_max = i+2
                        i_min = i-1
                        
                    if j == 0:
                        stencil_horiz = xp.array((((1,0,-1), 
                                                    (0,0,0), 
                                                    (0, 0, 0))))
                        j_min = 0
                        j_max = 3
                    elif j == image_shape[0]-1:
                        stencil_horiz = xp.array(((0,0,0), 
                                                    (0,0,0), 
                                                    (1,0,-1)))
                        j_max = image_shape[0]
                        j_min = image_shape[0] - 3
                    else: # in the middle of i
                        stencil_horiz = xp.array(((0,0,0), 
                                                    (1,-2,1), 
                                                    (0,0,0)))
                        j_min = j-1
                        j_max = j + 2
                    
                    stencil_total = stencil_horiz + stencil_vertical
                    init_array[i_min:i_max, j_min:j_max] += stencil_total
                    
                constraints_total.append(init_array.flatten())
                b.append(0)
                
        return xp.array(constraints_total), xp.array(b)

    def build_constraints(
            self,
            weights_reg = 1,
            weights_bc = 1,
            return_constraints = False,
        ):
        reg_constr, b_reg = self._build_constraints_reg(self.df_binary_images[0,:,:,0].shape)
        bc_constr, b_bc = self._build_constraints_BC_all()
        
        self.total_constraints = xp.vstack((bc_constr*weights_bc, reg_constr*weights_reg))
        self.total_b = xp.concatenate((b_bc*weights_bc, b_reg*weights_reg))
        if return_constraints:
            return self.total_constraints, self.total_b

    def solve(
            self,
            weights_reg = 1,
            weights_bc = 1,
            basis = "bezier",
            num_cp = None,
            method = "lsqr",
        ):
        if basis == "bezier":
            if num_cp is None:
                raise ValueError("num_cp must be specified for bezier basis")
            
            self.build_constraints(weights_reg, weights_bc)
            u = xp.linspace(0, 1, self.df_binary_images.shape[1])
            v = xp.linspace(0, 1, self.df_binary_images.shape[2])
            R = bezier_basis_change(u,v, num_cp, num_cp)
            
            constraints = self.total_constraints @ R
            b =  self.total_b
        
            if method == "lsqr":
                sol = xp.linalg.lstsq(constraints, b, rcond = None)[0]
                bez_height_map = (R @ sol).reshape((self.df_binary_images.shape[1],self.df_binary_images.shape[1]))
                return bez_height_map
            if method == "iterative":
                pass
        

        
        
        
        
        

        
        
        
        

