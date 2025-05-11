import numpy as np
import cupy as xp
from .utils.math_utils import bezier_surface, bernstein_poly, bezier_basis_change
from numpy import ndarray
import matplotlib.pyplot as plt

# class Surfaces:
#     """
#     Represents and manages crystal surfaces for bend contour analysis in TEM.

#     Parameters:
#         R (ndarray): Surface coordinates in real space with shape (num_surfaces, 3, height, width).
#         u (ndarray): First parameter of surface parameterization.
#         v (ndarray): Second parameter of surface parameterization.
#         c (ndarray): Excitation error parameter.
#         alpha (ndarray): First rotation angle in radians.
#         beta (ndarray): Second rotation angle in radians.
#         a0 (float): First lattice parameter in real space.
#         b0 (float): Second lattice parameter in real space.
#         material: Crystal structure object containing material properties.
#         width (float, optional): Width parameter for diffraction intensity. Defaults to 2*π/4/700.
#     """
#     def __init__(
#         self, 
#         R: ndarray,
#         u: ndarray,
#         v: ndarray,
#         c: ndarray,
#         alpha: ndarray,
#         beta: ndarray,
#         a0: float,
#         b0: float,
#         material,
#         width: float = 2*xp.pi/4/700
#     ):
#         """
        
#         """
        
#         if len(u.shape) == 1:
#             self.u, self.v = xp.meshgrid(u, v)
#         if len(u.shape) == 2:
#             self.u = u
#             self.v = v
            

#         self.c = c
        
#         self.alpha = alpha
      
#         self.beta = beta
        
#         self.x_range_0 = R[:, 0].max() - R[:, 0].min()
#         self.y_range_0 = R[:, 1].max() - R[:, 1].min()
        
#         self.du = xp.abs(self.u[1,1] - self.u[0,0])
#         self.dv = xp.abs(self.v[1,1] - self.v[0,0])

#         self.a0 = a0
#         self.b0 = b0

#         self.material = material
#         self.width = width
#         self.set_R(R)
    
#     def set_width(self, width):
#         self.width = width

#     def set_R(
#         self, 
#         R
#     ):
#         if len(R.shape) == 3:
#             self.R = R[None,:,:,:]
#         if len(R.shape) == 4:
#             self.R = R
        
#         self.get_UB()
    
#     def get_UB(self):
#         r_u, r_v = xp.gradient(self.R, 
#                                axis = (-2,-1))
        
#         cart_axis = 1
        
#         normals = -xp.cross(r_u, 
#                             r_v, 
#                             axis = cart_axis)
        
#         normals_normed = xp.einsum("...jkl,...kl->...jkl", 
#                                    normals,
#                                    1/xp.linalg.norm(normals, axis = cart_axis))

#         a_hat = r_u/self.x_range_0/self.du
#         b_hat = r_v/self.y_range_0/self.dv

#         self.a_vec = a_hat*self.a0
#         self.b_vec = b_hat*self.b0
        
#         a_len = xp.linalg.norm(self.a_vec,
#                                axis = cart_axis)
        
#         b_len = xp.linalg.norm(self.b_vec,
#                                axis = cart_axis)
        
#         gamma_cos = xp.einsum("...ijk,...ijk->...jk", 
#                               a_hat, 
#                               b_hat)
#         det = a_len*b_len*(1 - gamma_cos**2)
        
#         A1_inv = xp.array(((self.b0*xp.ones_like(gamma_cos), 
#                             -b_len*gamma_cos), 
#                            (-self.a0*gamma_cos, a_len*xp.ones_like(gamma_cos))))/det
        
#         v1 = self.c*xp.array((xp.cos(self.beta), xp.cos(self.alpha)))
        
#         x, y = xp.einsum("ij...,j...->i...", 
#                          A1_inv, 
#                          v1)
        
#         z = xp.sqrt(self.c**2 - (x**2 * a_len**2 + y**2 * b_len**2 + x*y*a_len*b_len*gamma_cos))
        
#         self.c_vec = xp.einsum("i...jk,i...ljk->...ljk",
#                           xp.array((x, y, z)),
#                           xp.array((self.a_vec, self.b_vec, normals_normed))) 
        
#         V = xp.einsum("...jkl,...jkl->...kl", 
#                       xp.cross(self.b_vec, self.c_vec, axis = cart_axis),
#                       self.a_vec)

#         UB = xp.zeros((self.R.shape[0], 3, 3, *self.R.shape[-2:]))
        
#         a_rec = 2*xp.pi*xp.einsum("...kl, ...jkl->...jkl", 
#                                   1/V, 
#                                   xp.cross(self.b_vec, self.c_vec, axis = cart_axis))
        
#         b_rec = 2*xp.pi*xp.einsum("...kl, ...jkl->...jkl", 
#                                   1/V, 
#                                   xp.cross(self.c_vec, self.a_vec, axis = cart_axis))
        
#         c_rec = 2*xp.pi*xp.einsum("...kl, ...jkl->...jkl",
#                                   1/V, 
#                                   xp.cross(self.a_vec, self.b_vec, axis = cart_axis))

#         UB[:,:,0,:,:] = a_rec
#         UB[:,:,1,:,:] = b_rec
#         UB[:,:,2,:,:] = c_rec
        
#         self.UB = UB.transpose([1,2,3,4,0])
    
#     def test_image_axis(self, i = 0):
#         fig, axR = plt.subplots(1,3)
        
#         fig.suptitle(f"self.R[{i}]")
#         axR[0].set_title("x")
#         axR[0].imshow(self.R[i, 0, :,:].get())
#         axR[1].set_title("y")
#         axR[1].imshow(self.R[i, 1, :,:].get())
#         axR[2].set_title("z")
#         axR[2].imshow(self.R[i, 2, :,:].get())
        
#         fig, ax_a = plt.subplots(1,3)
#         fig.suptitle(f"self.a_vec[{i}]")
#         ax_a[2].set_title("x")
#         ax_a[0].imshow(self.a_vec[i, 0, :,:].get())
#         ax_a[2].set_title("y")
#         ax_a[1].imshow(self.a_vec[i, 1, :,:].get())
#         ax_a[2].set_title("z")
#         ax_a[2].imshow(self.a_vec[i, 2, :,:].get())
        
#         fig, ax_b = plt.subplots(1,3)
#         fig.suptitle(f"self.b_vec[{i}]")
#         ax_b[2].set_title("x")
#         ax_b[0].imshow(self.b_vec[i, 0, :,:].get())
#         ax_b[2].set_title("y")
#         ax_b[1].imshow(self.b_vec[i, 1, :,:].get())
#         ax_b[2].set_title("z")
#         ax_b[2].imshow(self.b_vec[i, 2, :,:].get())
        
#         fig, ax_c = plt.subplots(1,3)
#         fig.suptitle(f"self.c_vec[{i}]")
#         ax_c[2].set_title("x")
#         ax_c[0].imshow(self.c_vec[i, 0, :,:].get())
#         ax_c[2].set_title("y")
#         ax_c[1].imshow(self.c_vec[i, 1, :,:].get())
#         ax_c[2].set_title("z")
#         ax_c[2].imshow(self.c_vec[i, 2, :,:].get())

class Surfaces:
    """
    Represents and manages crystal surfaces for bend contour analysis in TEM.

    Parameters:
        R (ndarray): Surface coordinates in real space with shape (num_surfaces, 3, height, width).
        u (ndarray): First parameter of surface parameterization.
        v (ndarray): Second parameter of surface parameterization.
        c (ndarray): Excitation error parameter.
        alpha (ndarray): First rotation angle in radians.
        beta (ndarray): Second rotation angle in radians.
        a0 (float): First lattice parameter in real space.
        b0 (float): Second lattice parameter in real space.
        material: Crystal structure object containing material properties.
        width (float, optional): Width parameter for diffraction intensity. Defaults to 2*π/4/700.
    """
    def __init__(
        self, 
        R: ndarray,
        u: ndarray,
        v: ndarray,
        alpha: ndarray,
        beta: ndarray,
        material,
        width: float = 2*xp.pi/4/700,
        U: ndarray = xp.diag(xp.ones(3)), # the
    ):
        """
        Initialize the Surfaces class.

        Parameters:
            R (ndarray): Surface coordinates in real space with shape (num_surfaces, 3, height, width).
            u (ndarray): First parameter of surface parameterization.
            v (ndarray): Second parameter of surface parameterization.
        
        """
        
        if len(u.shape) == 1:
            self.u, self.v = xp.meshgrid(u, v)
        if len(u.shape) == 2:
            self.u = u
            self.v = v
            

        self.c = np.linalg.norm(material.a3)
        
        self.alpha = alpha
      
        self.beta = beta
        
        self.x_range_0 = R[:, 0].max() - R[:, 0].min()
        self.y_range_0 = R[:, 1].max() - R[:, 1].min()
        
        self.du = xp.abs(self.u[1,1] - self.u[0,0])
        self.dv = xp.abs(self.v[1,1] - self.v[0,0])

        self.a0 = np.linalg.norm(material.a1)
        self.b0 = np.linalg.norm(material.a2)
        self.c0 = np.linalg.norm(material.a3)

        self.material = material
        self.width = width
        self.U = U

        self.B_real0 = xp.array((self.material.a1, 
                           self.material.a2, 
                           self.material.a3)).T
        
        self.B_recip0 = 2*xp.pi*xp.linalg.inv(self.B_real0)

        self.set_R(R)
    
    def set_width(self, width):
        self.width = width

    def set_R(
        self, 
        R
    ):
        if len(R.shape) == 3:
            self.R = R[None,:,:,:]
        if len(R.shape) == 4:
            self.R = R
        
        self.get_UB()
    
    def get_UB(self):
        self.r_u, self.r_v = xp.gradient(self.R, 
                               axis = (-2,-1))
        
        cart_axis = 1
        
        normals = xp.cross(self.r_u, 
                            self.r_v, 
                            axis = cart_axis)
        
        normals_normed = xp.einsum("...jkl,...kl->...jkl", 
                                   normals,
                                   1/xp.linalg.norm(normals, axis = cart_axis))

        x_hat = self.r_u/self.x_range_0/self.du
        y_hat = self.r_v/self.y_range_0/self.dv
        z_hat = normals_normed
        self.z_hat = z_hat
        
        T = xp.array((x_hat, y_hat, z_hat))
        
        
        TUB_real = xp.einsum("ijklm,in,no->jkolm", T, self.U, self.B_real0)

        self.a_vec = TUB_real[:,:,0,:,:]
        self.b_vec = TUB_real[:,:,1,:,:]
        self.c_vec = TUB_real[:,:,2,:,:]
        
        # a_len = xp.linalg.norm(self.a_vec,
        #                        axis = cart_axis)
        
        # b_len = xp.linalg.norm(self.b_vec,
        #                        axis = cart_axis)
        
        # gamma_cos = xp.einsum("...ijk,...ijk->...jk", 
        #                       a_hat, 
        #                       b_hat)
        # det = a_len*b_len*(1 - gamma_cos**2)
        
        # A1_inv = xp.array(((self.b0*xp.ones_like(gamma_cos), 
        #                     -b_len*gamma_cos), 
        #                    (-self.a0*gamma_cos, a_len*xp.ones_like(gamma_cos))))/det
        
        # v1 = self.c*xp.array((xp.cos(self.beta), xp.cos(self.alpha)))
        
        # x, y = xp.einsum("ij...,j...->i...", 
        #                  A1_inv, 
        #                  v1)
        
        # z = xp.sqrt(self.c**2 - (x**2 * a_len**2 + y**2 * b_len**2 + x*y*a_len*b_len*gamma_cos))
        
        # self.c_vec = xp.einsum("i...jk,i...ljk->...ljk",
        #                   xp.array((x, y, z)),
        #                   xp.array((self.a_vec, self.b_vec, normals_normed))) 
        
        V = xp.einsum("...jkl,...jkl->...kl", 
                      xp.cross(self.b_vec, self.c_vec, axis = cart_axis),
                      self.a_vec)

        UB = xp.zeros((self.R.shape[0], 3, 3, *self.R.shape[-2:]))
        
        a_rec = 2*xp.pi*xp.einsum("...kl, ...jkl->...jkl", 
                                  1/V, 
                                  xp.cross(self.b_vec, self.c_vec, axis = cart_axis))
        
        b_rec = 2*xp.pi*xp.einsum("...kl, ...jkl->...jkl", 
                                  1/V, 
                                  xp.cross(self.c_vec, self.a_vec, axis = cart_axis))
        
        c_rec = 2*xp.pi*xp.einsum("...kl, ...jkl->...jkl",
                                  1/V, 
                                  xp.cross(self.a_vec, self.b_vec, axis = cart_axis))

        UB[:,:,0,:,:] = a_rec
        UB[:,:,1,:,:] = b_rec
        UB[:,:,2,:,:] = c_rec
        
        self.UB = UB.transpose([1,2,3,4,0])
    
    def get_strain_tensor(self):
        (2, 1, 3, 128, 128)
        #bez_surf.get_strain_tensor()
        R_strain = xp.array((self.r_u, self.r_v))
        print(R_strain.shape)
        eps = xp.einsum("isklm, jsklm->sijlm", R_strain, R_strain)

        return eps

    def test_image_axis(self, i = 0):
        fig, axR = plt.subplots(1,3)
        
        fig.suptitle(f"self.R[{i}]")
        axR[0].set_title("x")
        axR[0].imshow(self.R[i, 0, :,:].get())
        axR[1].set_title("y")
        axR[1].imshow(self.R[i, 1, :,:].get())
        axR[2].set_title("z")
        axR[2].imshow(self.R[i, 2, :,:].get())
        
        fig, ax_a = plt.subplots(1,3)
        fig.suptitle(f"self.a_vec[{i}]")
        ax_a[2].set_title("x")
        ax_a[0].imshow(self.a_vec[i, 0, :,:].get())
        ax_a[2].set_title("y")
        ax_a[1].imshow(self.a_vec[i, 1, :,:].get())
        ax_a[2].set_title("z")
        ax_a[2].imshow(self.a_vec[i, 2, :,:].get())
        
        fig, ax_b = plt.subplots(1,3)
        fig.suptitle(f"self.b_vec[{i}]")
        ax_b[2].set_title("x")
        ax_b[0].imshow(self.b_vec[i, 0, :,:].get())
        ax_b[2].set_title("y")
        ax_b[1].imshow(self.b_vec[i, 1, :,:].get())
        ax_b[2].set_title("z")
        ax_b[2].imshow(self.b_vec[i, 2, :,:].get())
        
        fig, ax_c = plt.subplots(1,3)
        fig.suptitle(f"self.c_vec[{i}]")
        ax_c[2].set_title("x")
        ax_c[0].imshow(self.c_vec[i, 0, :,:].get())
        ax_c[2].set_title("y")
        ax_c[1].imshow(self.c_vec[i, 1, :,:].get())
        ax_c[2].set_title("z")
        ax_c[2].imshow(self.c_vec[i, 2, :,:].get())
    
    def visualize_vectors(self, i = 0, fig = None, ax = None):

        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(projection = "3d")

        scale = 1000

        ax.plot_surface(self.R[i,0,:,:].get(), self.R[i,1,:,:].get(), self.R[i,2,:,:].get()*scale, alpha = 0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")


        a_vec = self.a_vec.get()
        b_vec = self.b_vec.get()
        c_vec = self.c_vec.get()

        # c_vecs = ax.quiver(my_surfs.R[0,0,::10,::10].get(), my_surfs.R[0,1,::10,::10].get(), my_surfs.R[0,2,::10,::10].get()*1000, 
        #                    scale*c_vec[0,0,::10,::10], scale*c_vec[0,1,::10,::10], c_vec[0,2,::10,::10], length = 1e-2, color = "blue")

        b_vecs = ax.quiver(self.R[0,0,5::10,5::10].get(), self.R[0,1,5::10,5::10].get(), self.R[0,2,5::10,5::10].get()*scale, 
                        b_vec[0,0,5::10,5::10], b_vec[0,1,5::10,5::10], b_vec[0,2,5::10,5::10]*scale, length = 4e-2, color = "green", label = "b")

        a_vecs = ax.quiver(self.R[0,0,5::10,5::10].get(), self.R[0,1,5::10,5::10].get(), self.R[0,2,5::10,5::10].get()*scale, 
                        a_vec[0,0,5::10,5::10], a_vec[0,1,5::10,5::10], a_vec[0,2,5::10,5::10]*scale, length = 4e-2, color = "red", label = "a")
        
        fig.legend()

class Bezier_Surfaces(Surfaces):
    """
    Models crystal surfaces using Bézier interpolation for smooth deformation.

    Parameters:
        control_points (ndarray): Control points defining the Bézier surface shape.
        c (ndarray): Excitation error parameter.
        alpha (ndarray): First rotation angle in radians.
        beta (ndarray): Second rotation angle in radians.
        a0 (float): First lattice parameter in real space.
        b0 (float): Second lattice parameter in real space.
        material: Crystal structure object containing material properties.
        num_samples (int): Number of points to sample in each direction.
        width (float, optional): Width parameter for diffraction intensity. Defaults to 2*π/4/700
    """
    def __init__(
        self, 
        control_points: ndarray,
        alpha: ndarray,
        beta: ndarray,
        material,
        num_samples: int,
        width: float = 2*xp.pi/4/700,
        U: ndarray = xp.diag(xp.ones(3)),
    ):  
        self.u_1d = xp.linspace(0, 1, num_samples)
        self.v_1d = xp.linspace(0, 1, num_samples)
        
        self.alpha = alpha
        self.beta = beta

        self.material = material
        self.width = width
        self.U = U

        self.set_control_points(control_points)
        
    def set_control_points(self, 
                           control_points
    ):
        if len(control_points.shape) == 3:
            self.control_points = control_points[None]
        elif len(control_points.shape) == 4:
            self.control_points = control_points
            
        assert self.control_points.shape[1] == self.control_points.shape[2]
        assert self.control_points.shape[3] == 3
        
        self.num_c = control_points.shape[1]
        
        R_bez = bezier_surface(self.u_1d,
                               self.v_1d, 
                               self.control_points)
        
        Surfaces.__init__(
            self, 
            R_bez,
            self.u_1d,
            self.v_1d,
            self.alpha,
            self.beta,
            self.material,
            width = self.width,
            U = self.U,
        )
    
    def set_control_points_list(self,
                                control_points_list: ndarray,
    ):
        """
        Sets the control points for the Bézier surface.

        Parameters:
            control_points_list (ndarray): List of control points defining the Bézier surface.
        """
        
        if len(control_points_list.shape) == 1:
            num_c_float = np.sqrt(control_points_list.size/3)
            control_points_list = control_points_list[None]
            
        num_c_float = np.sqrt(control_points_list[0].size/3)
        assert num_c_float % 1 == 0 # is an integer
        num_c = int(num_c_float)
        num_r = control_points_list.shape[0]
        
        control_points = control_points_list.reshape((num_r, num_c,num_c,3))
        
        self.set_control_points(control_points)


    def updown_sample_cp(self, cp_num):
        basis_R_cp_prime = xp.array(bezier_basis_change(self.u_1d, self.v_1d, cp_num, cp_num))
        basis_R_cp = xp.array(bezier_basis_change(self.u_1d, self.v_1d, self.num_c, self.num_c))

        new_control_points = xp.zeros((cp_num, cp_num, 3))

        for i in range(3): #xyz
            xyz = xp.array(self.control_points[0,:,:,i])
            xyz_flat = xyz.flatten()

            b = basis_R_cp @ xyz_flat

            A = basis_R_cp_prime

            xyz_new = xp.linalg.lstsq(A, b, rcond = None)[0]

            new_control_points[:,:,i] = xyz_new.reshape((cp_num, cp_num))
        return new_control_points


        