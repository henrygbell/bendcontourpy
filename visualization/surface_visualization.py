import numpy as np
import cupy as xp
import matplotlib.pyplot as plt

def plot_surfaces(
        surface,
        figax = None,
):
    num_Rs = surface.R.shape[0]
    if figax is None:
        fig = plt.figure(figsize=plt.figaspect(0.5))
        axs = [fig.add_subplot(1, num_Rs, 1+i, projection='3d') for i in range(num_Rs)] 
    else:
        fig, axs = figax

    assert np.array(axs).size == num_Rs
    
    for i in range(num_Rs):
        axs[i].plot_surface(surface.R[i, 0].get(), surface.R[i, 1].get(), 1000*surface.R[i, 2].get(), cmap = "magma")
        axs[i].set_xlabel("x, (um)")
        axs[i].set_ylabel("y, (um)")
        axs[i].set_zlabel("z, (nm)")
    return fig, axs

def plot_bf(
    experiment,
    i_R = 0,
):
    num_theta = experiment.I_bf.shape[1]

    n_row = int(np.ceil(num_theta/10)) 
    
    fig, ax = plt.subplots(n_row, 10, figsize = (14,10))
    ax = np.array(ax).flatten()
    for i in range(num_theta):
        ax[i].imshow(experiment.I_bf[i_R, i].get(), vmax = 1, vmin = 0)
        ax[i].set_title(fr"$\theta$ = {np.round(experiment.rotation_angles[i]*180/np.pi, 2)}$^\circ$")
    
    return fig, ax

def plot_df(
    experiment,
    i_R = 0,
):
    pass
