import numpy as np

def lower_approximation(matrix):
    """
    Lower approximation: conservative estimation.
    We use minimum of each row as lower bound.
    """
    lower = []

    for row in matrix:
        lower.append(np.min(row))

    # Expand back to matrix shape
    lower_matrix = np.tile(np.array(lower).reshape(-1, 1), (1, matrix.shape[1]))
    return lower_matrix


def upper_approximation(matrix):
    """
    Upper approximation: optimistic estimation.
    We use maximum of each row as upper bound.
    """
    upper = []

    for row in matrix:
        upper.append(np.max(row))

    # Expand back to matrix shape
    upper_matrix = np.tile(np.array(upper).reshape(-1, 1), (1, matrix.shape[1]))
    return upper_matrix
