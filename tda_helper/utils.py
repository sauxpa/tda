import numpy as np

def triangular_from_linear_index(n, k):
    """Convert from a linear index describing the position in a flattened
    triangular array to the coordinates (i,j) in the full matrix.

    n : size of the matrix (size of the flattened list is n*(n-1)/2),
    k : linear index.
    """
    i = n - 2 - np.floor(np.sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5)
    j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
    return int(i), int(j)

def linear_index_from_triangular(n, i, j):
    """Convert from a coordinates (i,j) in a full matrix
    to a linear index describing the position in the flattened
    array.
    n : size of the matrix (size of the flattened list is n*(n-1)/2),
    i, j : coordinates in the matrix.
    """
    if j >= i:
        return int((n-2)*(n-1)/2-(n-2-i)*(n-i-1)/2+j-1)
    else:
        return int((n-2)*(n-1)/2-(n-2-j)*(n-j-1)/2+i-1)
