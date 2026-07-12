import pandas as pd
from .hesitant_fuzzy import hesitant_score_matrix
from .fuzzy_rough import lower_approximation, upper_approximation
from .ranking_method import normalize_matrix, weighted_score

def run_chf_rough_madm(data_path, weights):
    df = pd.read_csv(data_path)

    # Step 1: Hesitant fuzzy score matrix
    hf_matrix = hesitant_score_matrix(df)

    # Step 2: Fuzzy rough approximations
    lower = lower_approximation(hf_matrix)
    upper = upper_approximation(hf_matrix)

    # Step 3: Combine
    combined = (lower + upper) / 2

    # Step 4: Normalize
    normalized = normalize_matrix(combined)

    # Step 5: Weighted score
    scores = weighted_score(normalized, weights)

    # Result DataFrame
    result_df = pd.DataFrame({
        "Alternative": df.index + 1,
        "Score": scores
    }).sort_values(by="Score", ascending=False)

    # Best score from this dataset
    best_score = result_df.iloc[0]["Score"]

    return result_df, best_score
