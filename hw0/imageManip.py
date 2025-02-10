import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
   
    out = io.imread(image_path)
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):

    out = image[start_row:start_row+num_rows, start_col:start_col+num_cols,:]

    return out


def dim_image(image):
    out = np.zeros_like(image)
    height = image.shape[0]
    width = image.shape[1]
    for i in range(height):
        for j in range(width):
            out[i,j] = 0.5*image[i,j]**2
    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape

    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!
    # Calculate scale factors
    row_scale_factor = input_rows / output_rows
    col_scale_factor = input_cols / output_cols

    # Initialize the output image
    output_image = np.zeros((output_rows, output_cols, channels), dtype=input_image.dtype)

    # Populate the output image
    for i in range(output_rows):
        for j in range(output_cols):
            # Map the output pixel (i, j) to the nearest input pixel
            input_i = int(i * row_scale_factor)
            input_j = int(j * col_scale_factor)

            # Copy the RGB values from input to output
            output_image[i, j, :] = input_image[input_i, input_j, :]

    return output_image



    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)
    x1 =point[0]
    y1 = point[1]
    x2 = np.cos(theta)*x1 - np.sin(theta)*y1
    y2 = np.sin(theta)*x1 + np.cos(theta)*y1
    return np.array([x2,y2])



def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """

    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    center =np.array([input_rows/2,input_cols/2])

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)
    for i in range(input_rows):
        for j in range(input_cols):
            p= np.array([i,j]) -center
            p = np.round(rotate2d(p,theta))
            p = p +center
            if 0<= p[0] < input_rows and 0 <= p[1] < input_cols:
                output_image[i,j,:] = input_image[p[0],p[1],:]
            else:
                output_image[i, j,:] = np.array([0,0,0])
                


    return output_image
