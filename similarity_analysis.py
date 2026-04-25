import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def compare(ref_csv, conv_csv):
    ref = pd.read_csv(ref_csv)
    conv = pd.read_csv(conv_csv)

    sim = cosine_similarity(ref, conv)[0][0]
    print(f"Similarity: {sim:.4f}")

compare("CSVs/spb_ref_features.csv", "CSVs/spb_conv_features.csv")
compare("CSVs/arr_ref_features.csv", "CSVs/arr_conv_features.csv")
compare("CSVs/mano_ref_features.csv", "CSVs/mano_conv_features.csv")
compare("CSVs/ilai_ref_features.csv", "CSVs/ilai_conv_features.csv")

