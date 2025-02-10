"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi,Wi = image.shape
    Hk,Wk = kernel.shape
    out = np.zeros((Hi,Wi))
    for i in range(Hi):
        for j in range(Wi):
            for m in range(Hk):
                for n in range(Wk):
                    image_i= i + m - Hk//2
                    image_j = j + n -Wk//2 
                    if  0 <= image_i < Hi and 0<= image_j< Wi:
                        out[i,j] += image[image_i,image_j]*kernel[m,n]


    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros((H+2*pad_height,W+2*pad_width))
    out[pad_height:pad_height+H,pad_width:pad_width+W] = image

    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    image_padded= zero_pad(image,Hk//2,Wk//2)
    out = np.zeros((Hi, Wi))
    kernel = np.flip(kernel)
    for i in range(Hi):
        for j in range(Wi):
            out[i,j]= np.sum(kernel * image_padded[i:i+Hk,j:j+Wk])

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = conv_fast(f,np.flip(g))
    

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    g_mean= np.mean(g)
    M =g_mean*np.ones_like(g)
    g = g-M
    out = conv_fast(f,np.flip(g))

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.
    
    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    Hf,Wf = f.shape
    Hg,Wg = g.shape
    g_mean= np.mean(g)
    g_std= np.std(g)
    g_normalized=(g-g_mean)/g_std
    out = np.zeros((Hf,Wf))
    for m in range(Hf):
        for n in range(Wf):
            fmn = f[m-Hg//2:m+Hg//2,n-Wg//2:n+Wg//2]
            fmn_mean = np.mean(fmn)
            fmn_sd=np.std(fmn)
            M = cross_correlation((f-fmn_mean)/fmn_sd,g_normalized)
            out[m,n] = M[m,n]


    return out
