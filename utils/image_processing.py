import cupy as xp
import numpy as np

def cross_correlation_registration(
        fixed_images, 
        moving_images
):
    """
    Returns the  shift calculated through cross correlation (vectorized)

    Inputs:
        fixed_image: ndarray, N x img_x x img_y
        moving_image: ndarray, N x img_x x img_y
    """
    fixed_images_norm = xp.linalg.norm(fixed_images, axis = (1, 2))
    moving_images_norm = xp.linalg.norm(moving_images, axis = (1, 2))
    
    fixed_images = (fixed_images.T/fixed_images_norm).T
    moving_images = (moving_images.T/moving_images_norm).T
    
    image_product = xp.fft.fft2(fixed_images) * xp.fft.fft2(moving_images).conj()
    cc_images = xp.fft.fftshift(
        xp.fft.ifft2(image_product),
    )
    max_inds = cc_images.argmax(axis = (1,2))
    max_val = np.abs(cc_images).max(axis = (1,2))
    
    max_val = xp.nan_to_num(max_val, 0)
    
    return xp.array(xp.unravel_index(max_inds, fixed_images.shape[1:])).T - xp.array(fixed_images.shape[1:])/2, max_val
        
def bin_img(imgs, bin_width):
    """
    Bins an image by averaging over a specified bin width.

    Parameters:
        imgs (ndarray): Image to be binned.
        bin_width (int): Width of the bins.
    """
    img_side = imgs.shape[1]
    assert img_side % bin_width == 0
    if imgs.ndim == 2:
        binned_imgs = imgs.reshape(img_side//bin_width, bin_width, img_side//bin_width, bin_width).mean(axis = (1, 3))

    elif imgs.ndim == 3:
        N_imgs = imgs.shape[0]
        
        binned_imgs = imgs.reshape(N_imgs, img_side//bin_width, bin_width, img_side//bin_width, bin_width).mean(axis = (2, 4))

    return binned_imgs