import numpy as np
import cupy as xp
from ..utils.math_utils import get_rot_matrix_rodriguez, bezier_basis_change
from ..surfaces import Bezier_Surfaces, Surfaces
from ..experiment import Experiment
import cupy as cp
import cupyx.scipy.sparse as sparse
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import lsmr


# class strain_free_solver:

#     @classmethod
#     def from_energy(cls, 
#                     df_binary_images, 
#                     qs, 
#                     material,
#                     energy_kev,
#                     rotation_matrices,
#                     dx,
#                     dy,
#                     U,
#                     boundary_constraints = "free",):
#         """
#         Create Experiment instance from electron energy.

#         Parameters:
#             energy_kev: Electron energy in keV
#             qs: Array of reciprocal lattice vectors
#             Surfs: Surface object containing crystal geometry
#             intensity_param: Scaling parameter for diffraction intensity
#             rotation_axis: Axis about which the sample is rotated
#             rotation_angles: Array of rotation angles in radians
#         """
#         k_i = cls._energy_to_wavevector(energy_kev)
#         return cls(k_i0=k_i, df_binary_images=df_binary_images, 
#                   qs = qs, 
#                   material=material,
#                   rotation_matrices=rotation_matrices, 
#                   dx=dx,
#                   dy = dy,
#                   U = U,
#                   boundary_constraints = boundary_constraints,)

#     @staticmethod
#     def _energy_to_wavevector(energy_kev: float):
#         """Convert electron energy to wavevector."""
#         E_electron_joule = energy_kev * 1e3 * 1.60218e-19
#         m_electron = 9.1093837e-31  # kg
#         hbar = 1.054571817e-34  # J s
#         k_magnitude = xp.sqrt(2 * m_electron * E_electron_joule) / hbar / 1e10
#         return k_magnitude * xp.array([0, 0, -1])  # Default direction -z

#     @staticmethod
#     def _wavevector_to_energy(k_i):
#         """Convert wavevector to electron energy."""
#         m_electron = 9.1093837e-31  # kg
#         hbar = 1.054571817e-34  # J s
#         k_magnitude = xp.linalg.norm(k_i)
#         E_electron_joule = k_magnitude**2 * hbar**2 *1e10 **2/ (2 * m_electron)
#         return E_electron_joule/1e3/1.60218e-19

#     def __init__(
#             self, 
#             df_binary_images, 
#             qs, 
#             material,
#             k_i0,
#             rotation_matrices,
#             dx,
#             dy,
#             U, 
#             boundary_constraints = "free",
#         ):

#         self.energy = strain_free_solver._wavevector_to_energy(k_i0)

#         self.df_binary_images = df_binary_images
#         self.qs_hkl = qs
#         self.k_i0 = k_i0
#         self.rotation_matrices = rotation_matrices
#         self.dx = dx
#         self.dy = dy
#         self.U = U
#         self.define_material(material)
#         self.bez_height_map = None
#         self.boundary_constraints = boundary_constraints
    
#     def define_material(self, 
#             material
#         ):
#         self.bez_points = xp.array([])
#         self.material = material

#         B_real = xp.array((self.material.a1, 
#                             self.material.a2, 
#                             self.material.a3))
        
#         UB_real = self.U @ B_real
        
#         self.UB = 2*xp.pi*xp.linalg.inv(UB_real)

#         self.qs_xyz_plane = xp.einsum('ij, jk -> ik', self.qs_hkl, self.UB)

#         qs_xyz_plane_norm = xp.linalg.norm(self.qs_xyz_plane, axis = 1)

#         optimal_rot_thetas = xp.arccos((2*xp.linalg.norm(self.k_i0)**2 - qs_xyz_plane_norm**2)/(2*xp.linalg.norm(self.k_i0)**2))/2

#         rot_around_vectors = xp.cross( self.qs_xyz_plane, xp.array([0,0,1])[None,:], axis = 1)
#         rot_around_vectors = (rot_around_vectors.T/xp.linalg.norm(rot_around_vectors, axis = 1)).T

#         R_rot_mat = get_rot_matrix_rodriguez(ks = rot_around_vectors.T, thetas = optimal_rot_thetas)

#         self.R_diffs = xp.einsum("ijk,kj->ki", R_rot_mat, self.qs_xyz_plane)
#         self.R_diffs_rot = xp.einsum("kj,jil->kli", self.R_diffs, self.rotation_matrices.transpose((1,0,2)))
    

#     def _build_constraints_BC_one_order(
#             self, 
#             df_binary_image,
#             R_diff,
#         ):
#         num_constraints = int(df_binary_image.sum())
        
#         df_image_inds = xp.arange(df_binary_image.size)
#         df_image_inds_2d = xp.reshape(df_image_inds, df_binary_image.shape)
        
#         constraint_array = xp.zeros((num_constraints,df_binary_image.size))
        
#         image_shape = df_binary_image.shape
        
#         b = [] #rhs
#         for i in range(num_constraints):
#             init_array = xp.zeros(df_binary_image.size)

#             df_img_coord = df_image_inds_2d[df_binary_image][i]
#             x_i, y_i = np.unravel_index(df_img_coord, df_binary_image.shape)
            
#             # Deal with boundary conditions
#             onboundary = x_i == image_shape[0] - 1  or x_i  == 0 or y_i == image_shape[0]-1 or y_i == 0
            
#             if not onboundary:
#                 init_array[df_image_inds_2d[x_i+1, y_i]] += -1/(2*self.dx)*R_diff[0]
#                 init_array[df_image_inds_2d[x_i-1, y_i]] += 1/(2*self.dx)*R_diff[0]
#                 init_array[df_image_inds_2d[x_i, y_i+1]] += -1/(2*self.dy)*R_diff[1]
#                 init_array[df_image_inds_2d[x_i, y_i-1]] += 1/(2*self.dy)*R_diff[1]
#             # else:
#             #     if x_i == 0:
#             #         init_array[df_image_inds_2d[x_i+1, y_i]] += -1/(dx)*R_diff[0]
#             #         init_array[df_image_inds_2d[x_i, y_i]] += 1/(dx)*R_diff[0]
#             #     if x_i == (image_shape[0]-1):
#             #         init_array[df_image_inds_2d[x_i, y_i]] += -1/(dx)*R_diff[0]
#             #         init_array[df_image_inds_2d[x_i-1, y_i]] += 1/(dx)*R_diff[0]
#             #     else:
#             #         init_array[df_image_inds_2d[x_i+1, y_i]] += -1/(2*dx)*R_diff[0]
#             #         init_array[df_image_inds_2d[x_i-1, y_i]] += 1/(2*dx)*R_diff[0]
                
#             #     if y_i == 0:
#             #         init_array[df_image_inds_2d[x_i, y_i+1]] += -1/(dy)*R_diff[1]
#             #         init_array[df_image_inds_2d[x_i, y_i]] += 1/(dy)*R_diff[1]
#             #     if y_i == (image_shape[0]-1):
#             #         init_array[df_image_inds_2d[x_i, y_i]] += -1/(dy)*R_diff[1]
#             #         init_array[df_image_inds_2d[x_i, y_i-1]] += 1/(dy)*R_diff[1]
#             #     else:
#             #         init_array[df_image_inds_2d[x_i, y_i+1]] += -1/(2*dy)*R_diff[1]
#             #         init_array[df_image_inds_2d[x_i, y_i-1]] += 1/(2*dy)*R_diff[1]
            
#             b.append(-R_diff[2])
#             constraint_array[i] = init_array
#         return constraint_array, xp.array(b)

#     def _build_constraints_BC_all(
#             self,
#     ):
#         my_constraints = []
#         my_b = []
#         for m in range(self.df_binary_images.shape[0]): # angles
#             for j in range(self.qs_hkl.shape[0]): # bc orders
#                 constraints, b = self._build_constraints_BC_one_order(self.df_binary_images[m,:,:,j], self.R_diffs_rot[j, m])
#                 my_constraints.append(constraints)
#                 my_b.append(b)
#         my_bs_combined = my_b[0]
#         my_constraints_combined = my_constraints[0]

#         for i in range(len(my_b)-1):
#             my_constraints_combined = xp.vstack((my_constraints_combined, my_constraints[i+1]))
#             my_bs_combined = xp.concatenate((my_bs_combined, my_b[i+1]))

#         return xp.array(my_constraints_combined), xp.array(my_bs_combined)

#     def _build_constraints_reg(
#             self,
#             image_shape,
#         ):
#         constraints_total = []
#         b = []
        
#         for i in range(image_shape[0]):
#             for j in range(image_shape[1]):
#                 init_array = xp.zeros((image_shape[0], image_shape[1]))
                
#                 my_filter = xp.zeros((3,3))
                
#                 onboundary = (i == image_shape[0]-1 or i == 0 or j == image_shape[0]-1 or j == 0)
                
#                 if not onboundary:
#                     my_filter = xp.array(((0, 1,  0), 
#                                         (1, -4, 1), 
#                                         (0, 1,  0)))
                
#                     init_array[(i-1):(i+2), (j-1):(j+2)] += my_filter
#                 if self.boundary_constraints == "free":
#                     if onboundary:
#                         if i == 0:
#                             stencil_vertical = xp.array(((-1,0,0), 
#                                                         (0,0,0), 
#                                                         (1, 0, 0)))
#                             i_min = 0
#                             i_max = 3
#                         elif i == image_shape[0]-1:
#                             stencil_vertical = xp.array(((0,0,-1), 
#                                                         (0,0,0), 
#                                                         (0,0,1)))
#                             i_max = image_shape[0]
#                             i_min = image_shape[0]-3
#                         else: # in the middle of i
#                             stencil_vertical = xp.array(((0,1,0), 
#                                                         (0,-2,0), 
#                                                         (0,1,0)))
#                             i_max = i+2
#                             i_min = i-1
                            
#                         if j == 0:
#                             stencil_horiz = xp.array((((1,0,-1), 
#                                                         (0,0,0), 
#                                                         (0, 0, 0))))
#                             j_min = 0
#                             j_max = 3
#                         elif j == image_shape[0]-1:
#                             stencil_horiz = xp.array(((0,0,0), 
#                                                         (0,0,0), 
#                                                         (1,0,-1)))
#                             j_max = image_shape[0]
#                             j_min = image_shape[0] - 3
#                         else: # in the middle of i
#                             stencil_horiz = xp.array(((0,0,0), 
#                                                         (1,-2,1), 
#                                                         (0,0,0)))
#                             j_min = j-1
#                             j_max = j + 2
                    
#                     stencil_total = stencil_horiz + stencil_vertical
#                     init_array[i_min:i_max, j_min:j_max] += stencil_total
                    
#                 if self.boundary_constraints == "fixed" and onboundary:
#                     init_array[i,j] = 1
                    
#                 constraints_total.append(init_array.flatten())
#                 b.append(0)
                
#         return xp.array(constraints_total), xp.array(b)

#     def build_constraints(
#             self,
#             weights_reg = 1,
#             weights_bc = 1,
#             return_constraints = False,
#         ):
#         reg_constr, b_reg = self._build_constraints_reg(self.df_binary_images[0,:,:,0].shape)
#         bc_constr, b_bc = self._build_constraints_BC_all()
        
#         self.total_constraints = xp.vstack((bc_constr*weights_bc, reg_constr*weights_reg))
#         self.total_b = xp.concatenate((b_bc*weights_bc, b_reg*weights_reg))
#         if return_constraints:
#             return self.total_constraints, self.total_b

#     def solve(
#             self,
#             weights_reg = 1,
#             weights_bc = 1,
#             basis = "bezier",
#             num_cp = None,
#             method = "lsqr",
#         ):
#         if basis == "bezier":
#             if num_cp is None:
#                 raise ValueError("num_cp must be specified for bezier basis")
            
#             self.num_cp = num_cp

#             self.build_constraints(weights_reg, weights_bc)
#             u = xp.linspace(0, 1, self.df_binary_images.shape[1])
#             v = xp.linspace(0, 1, self.df_binary_images.shape[2])
#             R = bezier_basis_change(u,v, num_cp, num_cp)
            
#             constraints = self.total_constraints @ R
#             b =  self.total_b
        
#             if method == "lsqr":
#                 print(constraints.shape, b.shape)
#                 results = xp.linalg.lstsq(constraints, b)
#                 self.bez_points = results[0]
#                 print(results)
#                 self.residual = results[1]
#                 self.bez_height_map = (R @ self.bez_points).reshape((self.df_binary_images.shape[1],self.df_binary_images.shape[1]))
#                 return self.bez_height_map, self.residual
#             if method == "iterative":
#                 pass
    
#     def _get_surface(
#             self,
#             **kwargs,
#     ):
#         # if self.bez_height_map is None:
#         #     raise ValueError("bez_height_map is not set")
        
#         # u = xp.linspace(0, 1, self.df_binary_images.shape[1])
#         # v = xp.linspace(0, 1, self.df_binary_images.shape[1])

#         # u, v = xp.meshgrid(u,v, indexing = "ij")

#         # u_new = u 
#         # v_new = v

#         # X = (2*u_new - 1)*self.dx*self.df_binary_images.shape[1]/2 #(2*u - 1)*3.5 # strain field in X
#         # Y = (2*v_new - 1)*self.dy*self.df_binary_images.shape[1]/2 # 0.03*xp.exp(-((u-0.5)**2 + (v-0.5)**2)/0.05)#
#         # Z = self.bez_height_map - self.bez_height_map.mean() #0.05e-1*xp.sin(10*((u - 0.5)**2 - 2*(v-0.5)**3)) #3e-2*xp.exp(-((v-0.5)**2 + (u-0.5)**2)/0.05)

#         # R1 = xp.array((X,Y,Z))

#         # my_surfs_solved = Surfaces(R1, 
#         #                     u, 
#         #                     v, 
#         #                     np.pi/2, 
#         #                     np.pi/2, 
#         #                     self.material,
#         #                     U = self.U,
#         #                     **kwargs)
        


#         if self.bez_points.size == 0:
#             raise ValueError("bez_height_map is not set")
        


#         shape_bez = self.num_cp

#         u = xp.linspace(0, 1, shape_bez)
#         v = xp.linspace(0, 1, shape_bez)

#         u, v = xp.meshgrid(u, v, indexing = "ij")

#         x = 2*(u - 0.5)*self.df_binary_images.shape[1]*self.dx/2
#         y = 2*(v - 0.5)*self.df_binary_images.shape[2]*self.dy/2

#         control_points = xp.zeros((x.shape[0], x.shape[0], 3))
#         control_points[:,:,0] = x[:,:]
#         control_points[:,:,1] = y[:,:]
#         control_points[:,:,2] = self.bez_points.reshape(shape_bez, shape_bez)

#         self.bez_surface = Bezier_Surfaces(
#             control_points = control_points,
#             # alpha = xp.pi/2, #hard coded, fix later
#             # beta = xp.pi/2, #hard coded, fix later
#             material = self.material,
#             num_samples = self.df_binary_images.shape[1],
#             U = self.U,
#             **kwargs,
#         )

#         return self.bez_surface

#     def get_exp(
#         self,
#         intensity_param = 0.05,
#         **kwargs,
#     ):
#         surface =self._get_surface(**kwargs)

#         self.exp_solved = Experiment.from_energy(
#             self.energy,
#             self.qs_hkl,
#             surface,
#             intensity_param = intensity_param,
#             rotation_matrices = self.rotation_matrices,
#         )
#         return self.exp_solved
        
        
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
                    U,
                    boundary_constraints = "free",
                    storage_method = "sparse",
                    correct_rotation = True,):
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
                  U = U,
                  boundary_constraints = boundary_constraints,
                  storage_method = "sparse",
                  correct_rotation = correct_rotation,
                  )

    @staticmethod
    def _energy_to_wavevector(energy_kev: float):
        """Convert electron energy to wavevector."""
        E_electron_joule = energy_kev * 1e3 * 1.60218e-19
        m_electron = 9.1093837e-31  # kg
        hbar = 1.054571817e-34  # J s
        k_magnitude = xp.sqrt(2 * m_electron * E_electron_joule) / hbar / 1e10
        return k_magnitude * xp.array([0, 0, -1])  # Default direction -z

    @staticmethod
    def _wavevector_to_energy(k_i):
        """Convert wavevector to electron energy."""
        m_electron = 9.1093837e-31  # kg
        hbar = 1.054571817e-34  # J s
        k_magnitude = xp.linalg.norm(k_i)
        E_electron_joule = k_magnitude**2 * hbar**2 *1e10 **2/ (2 * m_electron)
        return E_electron_joule/1e3/1.60218e-19

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
            boundary_constraints = "free",
            storage_method = "dense",
            correct_rotation = True,
        ):

        self.energy = strain_free_solver._wavevector_to_energy(k_i0)

        self.df_binary_images = df_binary_images
        self.qs_hkl = qs
        self.k_i0 = k_i0
        self.rotation_matrices = rotation_matrices
        self.dx = dx
        self.dy = dy
        self.U = U
        self.define_material(material)
        self.bez_height_map = None
        self.total_constraints = None
        self.boundary_constraints = boundary_constraints
        self.storage_method = storage_method
        self.correct_rotation = correct_rotation
        if self.storage_method not in ["dense", "sparse"]:
            raise ValueError("storage_method must be either 'dense' or 'sparse'")
    
    def define_material(self, 
            material
        ):
        self.bez_points = xp.array([])
        self.material = material

        B_real = xp.array((self.material.a1, 
                            self.material.a2, 
                            self.material.a3))
        
        UB_real = self.U @ B_real
        
        self.UB = 2*xp.pi*xp.linalg.inv(UB_real)

        self.qs_xyz_plane = xp.einsum('ij, jk -> ik', self.qs_hkl, self.UB)

        qs_xyz_plane_norm = xp.linalg.norm(self.qs_xyz_plane, axis = 1)

        optimal_rot_thetas = xp.arccos((2*xp.linalg.norm(self.k_i0)**2 - qs_xyz_plane_norm**2)/(2*xp.linalg.norm(self.k_i0)**2))/2

        rot_around_vectors = xp.cross( self.qs_xyz_plane, xp.array([0,0,1])[None,:], axis = 1)
        rot_around_vectors = (rot_around_vectors.T/xp.linalg.norm(rot_around_vectors, axis = 1)).T

        R_rot_mat = get_rot_matrix_rodriguez(ks = rot_around_vectors.T, thetas = optimal_rot_thetas)

        self.R_diffs = xp.einsum("ijk,kj->ki", R_rot_mat, self.qs_xyz_plane)
        self.R_diffs_rot = xp.einsum("kj,jil->kli", self.R_diffs, self.rotation_matrices.transpose((1,0,2)))
    

    def _build_constraints_BC_one_order(self, df_binary_image, R_diff, rotation_matrix):
        num_constraints = int(df_binary_image.sum())
        if num_constraints == 0:
            return None, xp.array([])
        
        # Get coordinates of all True values in the binary image
        coords = xp.where(df_binary_image)
        x_coords, y_coords = coords[0], coords[1]
        
        # Create indices for the full image
        df_image_inds = xp.arange(df_binary_image.size)
        df_image_inds_2d = xp.reshape(df_image_inds, df_binary_image.shape)
        
        # Check for boundary points
        image_shape = df_binary_image.shape
        onboundary = (x_coords == image_shape[0] - 1) | (x_coords == 0) | \
                    (y_coords == image_shape[1] - 1) | (y_coords == 0)
        
        # Right-hand side vector
        b = xp.full(num_constraints, -R_diff[2])
        
        if self.storage_method == "dense":
            constraint_array = xp.zeros((num_constraints, df_binary_image.size))
            
            # Only process non-boundary points
            non_boundary_idxs = xp.where(~onboundary)[0]
            if len(non_boundary_idxs) > 0:
                x_nb = x_coords[non_boundary_idxs]
                y_nb = y_coords[non_boundary_idxs]
                
                # Get indices for up, down, left, right
                ind_down = df_image_inds_2d[x_nb + 1, y_nb]
                ind_up = df_image_inds_2d[x_nb - 1, y_nb]
                ind_right = df_image_inds_2d[x_nb, y_nb + 1]
                ind_left = df_image_inds_2d[x_nb, y_nb - 1]
                
                # Set values in the constraint matrix
                for i, idx in enumerate(non_boundary_idxs):
                    constraint_array[idx, ind_down[i]] = -1/(2*self.dx)*R_diff[0]
                    constraint_array[idx, ind_up[i]] = 1/(2*self.dx)*R_diff[0]
                    constraint_array[idx, ind_right[i]] = -1/(2*self.dy)*R_diff[1]
                    constraint_array[idx, ind_left[i]] = 1/(2*self.dy)*R_diff[1]
            
        elif self.storage_method == "sparse":
            # Preallocate for better performance
            non_boundary_idxs = xp.where(~onboundary)[0]
            nnz_entries = len(non_boundary_idxs) * 4  # 4 entries per non-boundary point
            
            row_ind = xp.zeros(nnz_entries, dtype=xp.int32)
            col_ind = xp.zeros(nnz_entries, dtype=xp.int32)
            data = xp.zeros(nnz_entries, dtype=xp.float32)
            
            if len(non_boundary_idxs) > 0:
                x_nb = x_coords[non_boundary_idxs]
                y_nb = y_coords[non_boundary_idxs]

                r1 = rotation_matrix[0, :]
                r2 = rotation_matrix[1, :]
                r3 = rotation_matrix[2, :]

                if self.correct_rotation:
                    A_inv = xp.array(((1,0,float(-r3[0]/r3[2])),
                                    (0,1,float(-r3[1]/r3[2])),
                                    (0,0,float(-1/r3[2]))))
                    
                    B = xp.array((r1, r2)).T
                    r = (xp.array((x_nb,y_nb)).T - xp.array((image_shape[0]//2, image_shape[1]//2))).T
                    
                    # print(A_inv.shape, B.shape, r.shape)

                    r_rot_back = xp.einsum("ij,jk->ik",A_inv @ B, r)

                    x_new, y_new = r_rot_back[0].astype(xp.int32)+image_shape[0]//2, r_rot_back[1].astype(xp.int32) + image_shape[1]//2
                    
                    # backproject x and y coordinates to indices
                    x_out_of_bounds = (x_new < 0) | (x_new >= image_shape[0])
                    y_out_of_bounds = (y_new < 0) | (y_new >= image_shape[1]) 

                    out_of_bounds = xp.logical_or(x_out_of_bounds, y_out_of_bounds)

                    x_new = x_new[xp.logical_not(out_of_bounds)]
                    y_new = y_new[xp.logical_not(out_of_bounds)]
                    
                    # print(x_new[0:5], x_nb[0:5])
                    # print(y_new[0:5], y_nb[0:5])
                    
                else:
                    x_new = x_nb
                    y_new = y_nb
                
                # Calculate all indices at once
                ind_down = df_image_inds_2d[x_new + 1, y_new]
                ind_up = df_image_inds_2d[x_new - 1, y_new]
                ind_right = df_image_inds_2d[x_new, y_new + 1]
                ind_left = df_image_inds_2d[x_new, y_new - 1]
                
                # Set row indices (4 entries per row)
                row_ind[0::4] = non_boundary_idxs
                row_ind[1::4] = non_boundary_idxs
                row_ind[2::4] = non_boundary_idxs
                row_ind[3::4] = non_boundary_idxs
                
                # Set column indices
                col_ind[0::4] = ind_down
                col_ind[1::4] = ind_up
                col_ind[2::4] = ind_right
                col_ind[3::4] = ind_left
                
                # Set data values
                data[0::4] = -1/(2*self.dx)*R_diff[0]
                data[1::4] = 1/(2*self.dx)*R_diff[0]
                data[2::4] = -1/(2*self.dy)*R_diff[1]
                data[3::4] = 1/(2*self.dy)*R_diff[1]
                
            constraint_array = csr_matrix((data, (row_ind, col_ind)), 
                                        shape=(num_constraints, df_binary_image.size))
            
        return constraint_array, b

    def _build_constraints_BC_all(
            self,
    ):
        my_constraints = []
        my_b = []
        for m in range(self.df_binary_images.shape[0]): # angles
            for j in range(self.qs_hkl.shape[0]): # bc orders
                if self.df_binary_images[m,:,:,j].sum() > 0:
                    constraints, b = self._build_constraints_BC_one_order(self.df_binary_images[m,:,:,j],
                                                                           self.R_diffs_rot[j, m],
                                                                           self.rotation_matrices[:,:,m])
                    my_constraints.append(constraints)
                    my_b.append(b)
        my_bs_combined = my_b[0]
        my_constraints_combined = my_constraints[0]

        for i in range(len(my_b)-1):
            if self.storage_method == "dense":
                my_constraints_combined = xp.vstack((my_constraints_combined, my_constraints[i+1]))
                
            if self.storage_method == "sparse":
                my_constraints_combined = sparse.vstack((my_constraints_combined, my_constraints[i+1]))
            # my_constraints_combined = xp.vstack((my_constraints_combined, my_constraints[i+1]))
            my_bs_combined = xp.concatenate((my_bs_combined, my_b[i+1]))

        return my_constraints_combined, xp.array(my_bs_combined)

    def _build_constraints_reg(
            self,
            image_shape,
        ):
        constraints_total = []
        b = []
        
        if self.storage_method == "sparse":
            row_inds = []
            col_inds = []
            datas = []
        
        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                if self.storage_method == "dense":
                    init_array = xp.zeros((image_shape[0], image_shape[1]))
    
                my_filter = xp.zeros((3,3))
                
                row_ind = np.ravel_multi_index((i,j), image_shape)
                
                onboundary = (i == image_shape[0]-1 or i == 0 or j == image_shape[0]-1 or j == 0)
                
                if not onboundary:
                    if self.storage_method == "dense":
                        my_filter = xp.array(((0, 1,  0), 
                                            (1, -4, 1), 
                                            (0, 1,  0)))
                        init_array[(i-1):(i+2), (j-1):(j+2)] += my_filter
                    if self.storage_method == "sparse":
                        col_inds_2d = np.array(((i,j),(i-1,j),(i+1,j),(i,j+1),(i, j-1)))
                        col_data = np.array(((-4,1,1,1,1)))
                        
                        col_inds_1d = np.ravel_multi_index(col_inds_2d.T, image_shape)
                        row_inds.extend([row_ind]*5)
                        col_inds.extend(list(col_inds_1d))
                        datas.extend(col_data)
                        
                if self.boundary_constraints == "free" and self.storage_method == "dense":
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
                    
                if self.boundary_constraints == "fixed" and onboundary:
                    if self.storage_method == "dense":
                        init_array[i,j] = 1
                    if self.storage_method == "sparse":
                        col_inds_2d = np.array(((i,j)))
                        col_data = 1
                        
                        col_inds_1d = np.ravel_multi_index(col_inds_2d.T, image_shape)
                        row_inds.extend([row_ind])

                        col_inds.extend([col_inds_1d])
                        datas.extend([col_data])
                if self.storage_method == "dense":
                    constraints_total.append(init_array.flatten())
                    
                b.append(0)
                
        if self.storage_method == "sparse":
            data_xp = xp.array(datas, dtype = xp.float32)
            row_inds = xp.array(row_inds, dtype = xp.float32)
            col_inds = xp.array(col_inds, dtype = xp.float32)
            
            constraints_total = csr_matrix((data_xp, (row_inds, col_inds)), shape=(image_shape[0]*image_shape[1], image_shape[0]*image_shape[1]))
        
        elif self.storage_method == "dense":
            constraints_total = xp.array(constraints_total)
                
        return constraints_total, xp.array(b)
    def build_constraints(
            self,
            weights_reg = 1,
            weights_bc = 1,
            return_constraints = False,
        ):
        
        reg_constr, b_reg = self._build_constraints_reg(self.df_binary_images[0,:,:,0].shape)
        bc_constr, b_bc = self._build_constraints_BC_all()
        if self.storage_method == "dense":
            self.total_constraints = xp.vstack((bc_constr*weights_bc, reg_constr*weights_reg))
        elif self.storage_method == "sparse":
            # print(bc_constr.shape, reg_constr.shape)
            self.total_constraints = sparse.vstack((bc_constr*weights_bc, reg_constr*weights_reg))
        self.total_b = xp.concatenate((b_bc*weights_bc, b_reg*weights_reg))
        if return_constraints:
            return self.total_constraints, self.total_b

    def solve(
            self,
            weights_reg = 1,
            weights_bc = 1,
            basis = "bezier",
            num_cp = None,
            method = "lstsqr",
            tol = None,
        ):
        if method not in ["lstsq", "lsmr"]:
            raise ValueError("method must be either 'lstsq' or 'lsmr'")
        if basis == "bezier":
            if num_cp is None:
                raise ValueError("num_cp must be specified for bezier basis")
            
            self.num_cp = num_cp
            if self.total_constraints is None:
                self.build_constraints(weights_reg, weights_bc)
            u = xp.linspace(0, 1, self.df_binary_images.shape[1])
            v = xp.linspace(0, 1, self.df_binary_images.shape[2])
        
            
            R = bezier_basis_change(u,v, num_cp, num_cp)
            
            constraints = self.total_constraints @ R
            b = self.total_b
            
            # if self.storage_method == "sparse":
            #     block_result = []
            #     block_size = num_cp
            #     for i in range(num_cp):
            #         # R_sub = bezier_basis_change(u, v[i*block_size:(i+1)*block_size], num_cp, num_cp)
            #         R = bezier_basis_change(u,v, num_cp, num_cp)
            #         print(R.shape, self.total_constraints.shape)
            #         sub_mat = self.total_constraints @ R_sub
                    
            #         if tol is not None:
            #             sub_mat[xp.abs(sub_mat) < tol] = 0
                        
            #         block_result.append(sub_mat.tocsr())
                    
            #     constraints = sparse.hstack(block_result).tocsr()
        
            if method == "lstsq": #least squares
                # print(constraints.shape)
                results = xp.linalg.lstsq(constraints, b)
                self.bez_points = results[0]
                # print(results)
                self.residual = results[1]
                self.bez_height_map = (R @ self.bez_points).reshape((self.df_binary_images.shape[1],self.df_binary_images.shape[1]))
                return self.bez_height_map, self.residual
            
            if method == "lsmr": #least squares QR decomp for sparse matrices

                if tol is not None:
                    constraints[xp.abs(constraints) < tol] = 0 
                results = lsmr(sparse.csr_matrix(constraints), b)

                
                self.bez_points = results[0]

                self.residual = results[2]
                self.bez_height_map = (R @ self.bez_points).reshape((self.df_binary_images.shape[1],self.df_binary_images.shape[1]))
                return self.bez_height_map, self.residual
    
    def _get_surface(
            self,
            num_samples,
            **kwargs,
    ):
        # if self.bez_height_map is None:
        #     raise ValueError("bez_height_map is not set")
        
        # u = xp.linspace(0, 1, self.df_binary_images.shape[1])
        # v = xp.linspace(0, 1, self.df_binary_images.shape[1])

        # u, v = xp.meshgrid(u,v, indexing = "ij")

        # u_new = u 
        # v_new = v

        # X = (2*u_new - 1)*self.dx*self.df_binary_images.shape[1]/2 #(2*u - 1)*3.5 # strain field in X
        # Y = (2*v_new - 1)*self.dy*self.df_binary_images.shape[1]/2 # 0.03*xp.exp(-((u-0.5)**2 + (v-0.5)**2)/0.05)#
        # Z = self.bez_height_map - self.bez_height_map.mean() #0.05e-1*xp.sin(10*((u - 0.5)**2 - 2*(v-0.5)**3)) #3e-2*xp.exp(-((v-0.5)**2 + (u-0.5)**2)/0.05)

        # R1 = xp.array((X,Y,Z))

        # my_surfs_solved = Surfaces(R1, 
        #                     u, 
        #                     v, 
        #                     np.pi/2, 
        #                     np.pi/2, 
        #                     self.material,
        #                     U = self.U,
        #                     **kwargs)
        


        if self.bez_points.size == 0:
            raise ValueError("bez_height_map is not set")
        
        shape_bez = self.num_cp

        u = xp.linspace(0, 1, shape_bez)
        v = xp.linspace(0, 1, shape_bez)

        u, v = xp.meshgrid(u, v, indexing = "ij")

        x = 2*(u - 0.5)*self.df_binary_images.shape[1]*self.dx/2
        y = 2*(v - 0.5)*self.df_binary_images.shape[2]*self.dy/2

        control_points = xp.zeros((x.shape[0], x.shape[0], 3))
        control_points[:,:,0] = x[:,:]
        control_points[:,:,1] = y[:,:]
        control_points[:,:,2] = self.bez_points.reshape(shape_bez, shape_bez)

        self.bez_surface = Bezier_Surfaces(
            control_points = control_points,
            # alpha = xp.pi/2, #hard coded, fix later
            # beta = xp.pi/2, #hard coded, fix later
            material = self.material,
            num_samples = num_samples,
            U = self.U,
            **kwargs,
        )

        return self.bez_surface

    def get_exp(
        self,
        intensity_param = 0.05,
        num_samples = 128,
        **kwargs,
    ):
        surface =self._get_surface(num_samples, **kwargs)

        self.exp_solved = Experiment.from_energy(
            self.energy,
            self.qs_hkl,
            surface,
            intensity_param = intensity_param,
            rotation_matrices = self.rotation_matrices,
        )
        return self.exp_solved