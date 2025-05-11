from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import numpy as np

def add_colorbar(fig, ax, im):
    """
    Adds a colorbar to a fig, ax, im.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical') 


import matplotlib

def get_RGB_image(complex_image):
    amplitude = np.abs(complex_image)
    phase = np.angle(complex_image)  # radians, from -pi to pi

    # Normalize amplitude (optional, for better contrast)
    amplitude = amplitude / np.max(amplitude)

    # Build HSV image
    hue = (phase + np.pi) / (2 * np.pi)  # map from [−π,π] to [0,1]
    saturation = np.ones_like(hue)
    value = amplitude

    hsv_image = np.stack([hue, saturation, value], axis=-1)
    rgb_image = matplotlib.colors.hsv_to_rgb(hsv_image)

    return rgb_image

def plot_df_phase_image(ax, DF_images, qs):
    assert DF_images.shape[0] == qs.shape[0]
    phase_factor = qs[:,0] + 1j*qs[:,1]

    df_phase_map = np.zeros((DF_images.shape[1], DF_images.shape[2]), dtype = np.complex128)

    for i in range(DF_images.shape[0]):
        my_I_df_phase = DF_images[i, :,:] *phase_factor[i]

        df_phase_map += my_I_df_phase

    rgb_image = get_RGB_image(df_phase_map)

    ax.imshow(rgb_image)