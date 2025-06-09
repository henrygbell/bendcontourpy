import numpy as np
import cupy as xp
from .utils.math_utils import bezier_surface, bernstein_poly, bezier_basis_change
from numpy import ndarray
import matplotlib.pyplot as plt

class Surfaces:
    """
    Stores information about the film surfaces and computes the UB matrix for bend contour simulation with the Experiment class.

    Parameters:
        R (ndarray): Surface coordinates in real space with shape (num_surfaces, 3, height, width).
        u (ndarray): First parameter of surface parameterization.
        v (ndarray): Second parameter of surface parameterization.
        c (ndarray): Excitation error parameter.
        material (Crystal class): Crystal structure object containing material properties.
        width (float, optional): Width parameter for diffraction intensity. Defaults to 2*π/4/700.
        dalpha (float, optional): tilt parameters from default defined by material 
        dbeta (float, optional): tilt parameters from default defined by material
    """
    
    def __init__(
        self, 
        R: ndarray,
        u: ndarray,
        v: ndarray,
        material,
        width: float = 2*xp.pi/4/700,
        U: ndarray = xp.diag(xp.ones(3)), 
        dalpha = 0,
        dbeta = 0,
        reference_config = None,
    ):
        
        self.reference_config = reference_config
        
        if len(u.shape) == 1:
            self.u, self.v = xp.meshgrid(u, v, indexing = "ij")
            self.u_1d = u
            self.v_1d = v
        if len(u.shape) == 2:
            self.u = u
            self.v = v
            self.u_1d = u[:,0]
            self.v_1d = v[0,:]
        
        alpha0 = np.arccos(np.dot(material.a1, material.a3)/(np.linalg.norm(material.a1)*np.linalg.norm(material.a3)))
        beta0 = np.arccos(np.dot(material.a2, material.a3)/(np.linalg.norm(material.a2)*np.linalg.norm(material.a3)))
        
        if type(dalpha) == float or type(dalpha) == int:
            self.dalpha = xp.ones_like(self.u)*dalpha
        else:
            if dalpha.ndim == 0:
                self.dalpha = xp.ones_like(self.u)*dalpha
            else:
                self.dalpha = dalpha
        
        self.alpha = alpha0 + self.dalpha
        
        if type(dbeta) == float or type(dbeta) == int:
            self.dbeta = xp.ones_like(self.u)*dbeta
        else:
            if dbeta.ndim == 0:
                self.dbeta = xp.ones_like(self.u)*dbeta
            else:
                self.dbeta = dbeta
        
        self.beta = beta0 + self.dbeta

        self.c = np.linalg.norm(material.a3)
        
        # self.x_range_0 = R[0].max() - R[0].min()
        # self.y_range_0 = R[1].max() - R[1].min()
        
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
        """
        Computes the UB matrix for the surface based on the real space coordinates R.
        """
        
        # gradients (tangential vectors)
        self.r_u, self.r_v = xp.gradient(self.R, 
                               axis = (-2,-1))
        
        cart_axis = 1
        
        #compute normal, normalize
        normals = xp.cross(self.r_u, 
                            self.r_v, 
                            axis = cart_axis)
        
        normals_normed = xp.einsum("...jkl,...kl->...jkl", 
                                   normals,
                                   1/xp.linalg.norm(normals, axis = cart_axis))
        
        #normalize by reference configuration (if available)
        if self.reference_config is not None:
            self.x_hat = self.r_u/self.du/self.reference_config['x_range']
            self.y_hat = self.r_v/self.dv/self.reference_config['y_range']
        else:
            self.x_hat = self.r_u/self.du
            self.y_hat = self.r_v/self.dv
            
            self.x_hat /= xp.linalg.norm(self.x_hat, axis = 1, keepdims = True)
            self.y_hat /= xp.linalg.norm(self.y_hat, axis = 1, keepdims = True)
        
        self.z_hat = normals_normed
        
        # compute frame transformation matrix T (represents the local coordinate system of each pixel)
        T = xp.array((self.x_hat, self.y_hat, self.z_hat))
        
        # TUB matrix is the local abc lattice vectors in real space
        TUB_real = xp.einsum("ijklm,in,no->jkolm", T, self.U, self.B_real0)

        #unpack
        self.a_vec = TUB_real[:,:,0,:,:]
        self.b_vec = TUB_real[:,:,1,:,:]
        self.c_vec = TUB_real[:,:,2,:,:]
        
        if (self.dalpha != 0).any() or (self.dbeta != 0).any(): # compute c given alpha and beta/alpha if they are nonzero
            a_hat = self.a_vec/xp.linalg.norm(self.a_vec, axis = 1, keepdims = True)
            b_hat = self.b_vec/xp.linalg.norm(self.b_vec, axis = 1, keepdims = True)
            n_hat = self.c_vec/xp.linalg.norm(self.c_vec, axis = 1, keepdims = True)
            
            cos_gamma = xp.einsum("ijkl,ijkl->ikl", a_hat, b_hat)
            
            A_inv = 1/(1-cos_gamma**2)*xp.array(((xp.ones_like(cos_gamma), -cos_gamma), 
                                                 (-cos_gamma, xp.ones_like(cos_gamma))))
            
            cos_alpha_beta = xp.array((xp.cos(self.alpha), xp.cos(self.beta)))
            
            c_a, c_b = xp.einsum("ij...,j...->i...", A_inv, cos_alpha_beta)
            
            c_n = xp.sqrt(1 - c_a**2 - c_b**2 - 2*c_a*c_b*cos_gamma)
            
            self.c_vec = a_hat*c_a + b_hat*c_b + n_hat*c_n
            
            self.c_vec = self.c0*self.c_vec/xp.linalg.norm(self.c_vec, axis = 1, keepdims = True)
            
            TUB_real[:,:,2,:,:] = self.c_vec
        
        # Compute reciprocal lattice vectors in real space
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

        # collect into UB matrix (used in diffraction intensity simulation)
        UB[:,:,0,:,:] = a_rec
        UB[:,:,1,:,:] = b_rec
        UB[:,:,2,:,:] = c_rec
        
        self.UB = UB.transpose([1,2,3,4,0])
    
    def get_strain_tensor(self):
        """
        Computes the Green-Lagrange strain tensor for the surface. (no refernce)
        
        Returns:
            eps (ndarray): The strain tensor with shape (num_surfaces, 2, 2, height, width).
        """
        #bez_surf.get_strain_tensor()
        R_strain = xp.array((self.x_hat, self.y_hat))
        eps = xp.einsum("isklm, jsklm->sijlm", R_strain, R_strain)

        return eps

    def get_strain_tensor_relative(self):
        eps = self.get_strain_tensor()
        
        # get strain tensor of flat surface
        eps_flat = xp.diag(xp.ones(2))[None, :,:,None, None] #identity matrix across surface!
        return 1/2*(eps - eps_flat)
    
    def get_cps(self, cp_num):
        """
        Computes the best fit for a cp_num x cp_num control point grid for the surface.
        
        Parameters:
            cp_num (int): Number of control points in each direction (creates square grid).
        Returns:
            new_control_points (ndarray): Control points for the Bézier surface with shape (cp_num, cp_num, 3).
        """
        basis_R_cp_prime = xp.array(bezier_basis_change(self.u_1d, self.v_1d, cp_num, cp_num))

        new_control_points = xp.zeros((cp_num, cp_num, 3))

        for i in range(3): #xyz
            b = self.R[0, i].flatten()
            # print(b.shape)

            A = basis_R_cp_prime
            # print(A.shape)

            xyz_new = xp.linalg.lstsq(A, b, rcond = None)[0]

            new_control_points[:,:,i] = xyz_new.reshape((cp_num, cp_num))
        return new_control_points

    def updown_sample_cp(self, cp_num):
        """
        Down or upsample the number of control points. 
        
        Only works on the first surface (i=0).
        
        Parameters:
            cp_num (int): Number of control points in each direction for the new Bézier surface.
        
        Returns:
            new_control_points (ndarray): Control points for the new Bézier surface with shape (cp_num, cp_num, 3).
        """
        basis_R_cp_prime = xp.array(bezier_basis_change(self.u_1d, self.v_1d, cp_num, cp_num))
        # basis_R_cp = xp.array(bezier_basis_change(self.u_1d, self.v_1d, self.num_c, self.num_c))

        new_control_points = xp.zeros((cp_num, cp_num, 3))

        for i in range(3): #xyz
            xyz = xp.array(self.control_points[0,:,:,i])
            xyz_flat = xyz.flatten()

            # b = basis_R_cp @ xyz_flat
            b = self.R[0,i,:,:].flatten()

            A = basis_R_cp_prime

            xyz_new = xp.linalg.lstsq(A, b, rcond = None)[0]

            new_control_points[:,:,i] = xyz_new.reshape((cp_num, cp_num))
        return new_control_points

    """
    Visualization Tools
    """
    
    
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
        
        ax.set_xlabel(r"X ($\mu m$)")
        ax.set_ylabel(r"Y ($\mu m$)")
        ax.set_zlabel(r"Z ($nm$)")
        
        fig.legend()
        
    def test_image_axis(self, i = 0):
        """
        Visualization of the surface vectors for debugging purposes.
        """
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

class Bezier_Surfaces(Surfaces):
    """
    Models crystal surfaces using Bézier interpolation for smooth deformation.

    Parameters:
        control_points (ndarray): Control points defining the Bézier surface shape.
        material: Crystal structure object containing material properties.
        num_samples (int): Number of points to sample in each direction.
        width (float, optional): Width parameter for diffraction intensity. Defaults to 2*π/4/700
        U (ndarray, optional): Transformation matrix for the crystal structure. Defaults to identity matrix.
        reference_config (optional): Reference configuration for the surface, if available. Format: {"x_range": float, "y_range": float}.
    """
    def __init__(
        self, 
        control_points: ndarray,
        material,
        num_samples: int,
        width: float = 2*xp.pi/4/700,
        U: ndarray = xp.diag(xp.ones(3)),
        reference_config = None,
    ):  
        self.u_1d = xp.linspace(0, 1, num_samples)
        self.v_1d = xp.linspace(0, 1, num_samples)

        self.material = material
        self.width = width
        self.U = U
        
        self.reference_config = reference_config

        self.set_control_points(control_points)
        
    def set_control_points(self, 
                           control_points
    ):
        """
        Sets new control points for the Bézier surface.
        
        Parameters:
            control_points: ndarray, Control points defining the Bézier surface shape with shape (num_surfaces, num_c, num_c, 3) or (num_c, num_c, 3).
        """
        if len(control_points.shape) == 3:
            self.control_points = control_points[None]
        elif len(control_points.shape) == 4:
            self.control_points = control_points
            
        assert self.control_points.shape[1] == self.control_points.shape[2]
        
        self.num_c = control_points.shape[1]
        
        R_bez = bezier_surface(self.u_1d,
                               self.v_1d, 
                               self.control_points)
        if R_bez.shape[1] >= 4:
            dalpha = R_bez[0,3,:,:]
            dbeta = R_bez[0,4,:,:]
            
            R_bez = R_bez[:,:3,:,:]
        else:
            dalpha = 0
            dbeta = 0
        
        # if self.control_points.shape[3] >= 4:
        #     R_bez[:,:]
        
        Surfaces.__init__(
            self, 
            R_bez,
            self.u_1d,
            self.v_1d,
            self.material,
            width = self.width,
            U = self.U,
            dalpha=dalpha,
            dbeta=dbeta,
            reference_config = self.reference_config,
        )
    
    def set_control_points_list(self,
                                control_points_list: ndarray,
    ):
        """
        Sets the control points for the Bézier surface with a list of 1D coordinates

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
        """
        Down or upsample the number of control points. 
        
        Only works on the first surface (i=0).
        
        Parameters:
            cp_num (int): Number of control points in each direction for the new Bézier surface.
        
        Returns:
            new_control_points (ndarray): Control points for the new Bézier surface with shape (cp_num, cp_num, 3).
        """
        basis_R_cp_prime = xp.array(bezier_basis_change(self.u_1d, self.v_1d, cp_num, cp_num))
        basis_R_cp = xp.array(bezier_basis_change(self.u_1d, self.v_1d, self.num_c, self.num_c))

        new_control_points = xp.zeros((cp_num, cp_num, 3))

        for i in range(3): #xyz
            xyz = xp.array(self.control_points[0,:,:,i])
            xyz_flat = xyz.flatten()

            # b = basis_R_cp @ xyz_flat
            b = self.R[0,i,:,:].flatten()

            A = basis_R_cp_prime

            xyz_new = xp.linalg.lstsq(A, b, rcond = None)[0]

            new_control_points[:,:,i] = xyz_new.reshape((cp_num, cp_num))
        return new_control_points


        