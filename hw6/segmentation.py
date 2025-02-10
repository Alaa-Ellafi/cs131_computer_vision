"""
CS131 - Computer Vision: Foundations and Applications
Assignment 6
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 10/9/2020
Python Version: 3.5+
"""

import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float
from skimage import filters
from scipy.ndimage.filters import convolve

### Clustering Methods

def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)
    assignments_iter = np.zeros(N, dtype=np.uint32)
    for n in range(num_iters):
        ### YOUR CODE HERE

        for i in range(N):
            min = float('inf')
            indx = 0
            for j in range(k):
                dist = np.linalg.norm(features[i]-centers[j])
                if dist < min : 
                    min = dist
                    indx = j
            assignments_iter[i] = indx
            
        if np.array_equal(assignments_iter, assignments):
            break
        assignments = assignments_iter.copy()
        for i in range(k):
            assigned_points = features[assignments_iter == i]
            centers[i] = np.mean(assigned_points, axis=0)
    return assignments



def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find cdist (imported from scipy.spatial.distance) and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        distances = cdist(features, centers)  
        assignments_iter = np.argmin(distances, axis=1)  
        if np.array_equal(assignments_iter, assignments):
            break
        assignments = assignments_iter.copy()
        for i in range(k):
            assigned_points = features[assignments_iter == i]
            if len(assigned_points) > 0:
                centers[i] = np.mean(assigned_points, axis=0)
        ### END YOUR CODE

    return assignments



def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Hints
    - You may find pdist (imported from scipy.spatial.distance) useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N, dtype=np.uint32)
    centers = np.copy(features)
    n_clusters = N
    n =0

    while n_clusters > k:
        import numpy as np
from scipy.spatial.distance import pdist, squareform

def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    Args:
        features - Array of N feature vectors. Each row represents a feature vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g., i-th point is assigned to cluster assignments[i])
    """
    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N, dtype=np.uint32)
    centers = np.copy(features)
    n_clusters = N

    while n_clusters > k:
        dist_matrix = squareform(pdist(centers, metric='euclidean'))
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist_idx = np.argmin(dist_matrix)
        i, j = np.unravel_index(min_dist_idx, dist_matrix.shape)
        assignments[assignments == j] = i
        merged_cluster_points = features[assignments == i]
        centers[i] = np.mean(merged_cluster_points, axis=0)
        centers = np.delete(centers, j, axis=0)
        n_clusters -= 1
    unique_clusters = np.unique(assignments)
    reindexed_assignments = np.zeros_like(assignments)
    for new_idx, old_idx in enumerate(unique_clusters):
        reindexed_assignments[assignments == old_idx] = new_idx
    assignments = reindexed_assignments

    return assignments

    


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))
    ### YOUR CODE HERE
    features = img.reshape(H * W, C)
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))
    color_features = img.reshape(H*W,C)

    ### YOUR CODE HERE
      # Flatten x and y
    x,y= np.mgrid[0:H, 0:W]
    # Normalize x and y coordinates
    coords = np.dstack((x,y))
    coords = coords.reshape(H*W,2)
    features= np.hstack((color_features,coords))
    features = (features - np.mean(features))/np.std(features)
    ### END YOUR CODE

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    ### I will transform the image to become zero-mean and then use harris corner detection algorithm to extract corners.
    img = img - np.mean(img)
    H,W,C = img.shape
    k = 0.04
    window = np.ones((3,3))
    features = np.zeros((H, W, C))

    for c in range(C):  # Process each channel separately
        Ix = filters.sobel_v(img[:, :, c])  # Vertical gradient
        Iy = filters.sobel_h(img[:, :, c])  # Horizontal gradient

        Ix2 = Ix ** 2
        Iy2 = Iy ** 2
        Ixy = Ix * Iy

        # Apply window smoothing using convolution
        Sx2 = convolve(Ix2, window, mode='constant', cval=0)
        Sy2 = convolve(Iy2, window, mode='constant', cval=0)
        Sxy = convolve(Ixy, window, mode='constant', cval=0)

        # Compute Harris response R = Det(M) - k * (Trace(M)^2)
        det = (Sx2 * Sy2) - (Sxy ** 2)
        trace = Sx2 + Sy2
        R = det - k * (trace ** 2)

        features[:, :, c] = R  # Store Harris response for each channel

    # Reshape to match expected output shape
    return features.reshape(H * W, C)
    ### END YOUR CODE



### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    correct_pixels = np.sum(mask_gt == mask)
    total_pixels = mask_gt.size
    accuracy = correct_pixels / total_pixels
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
