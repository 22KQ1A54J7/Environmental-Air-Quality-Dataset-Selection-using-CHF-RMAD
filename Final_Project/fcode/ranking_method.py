import numpy as np

def normalize_matrix(matrix):
    """
    Normalize each column using vector normalization.
    """
    matrix = np.array(matrix, dtype=float)

    norm_matrix = np.zeros_like(matrix)

    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        denom = np.sqrt(np.sum(col ** 2))

        if denom == 0:
            norm_matrix[:, j] = 0
        else:
            norm_matrix[:, j] = col / denom

    return norm_matrix


def weighted_score(normalized_matrix, weights):
    """
    Compute final weighted score for each alternative.
    """
    weights = np.array(weights, dtype=float)

    # If number of weights != number of criteria, auto-fix
    if len(weights) != normalized_matrix.shape[1]:
        weights = np.ones(normalized_matrix.shape[1]) / normalized_matrix.shape[1]

    scores = normalized_matrix.dot(weights)

    return scores
