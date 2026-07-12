import numpy as np

def hesitant_score_matrix(df):
    """
    Convert each numeric value into a simple hesitant fuzzy set.
    Here we model hesitation by creating 3 close membership values.
    """

    data = df.values.astype(float)

    # Normalize data between 0 and 1 for fuzzy membership
    min_val = data.min()
    max_val = data.max()

    if max_val - min_val == 0:
        normalized = np.zeros_like(data)
    else:
        normalized = (data - min_val) / (max_val - min_val)

    # Create hesitant fuzzy matrix:
    # Each element becomes an average of small hesitant values
    hf_matrix = []

    for row in normalized:
        hf_row = []
        for val in row:
            # Create a small hesitant set around val
            h_set = [val * 0.9, val, min(val * 1.1, 1.0)]
            hf_row.append(np.mean(h_set))  # use mean as hesitant score
        hf_matrix.append(hf_row)

    return np.array(hf_matrix)
