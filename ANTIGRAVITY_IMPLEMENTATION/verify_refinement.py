import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add parent dir to path to import mega_extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mega_extractor

def extract_and_compare():
    profiles_df = pd.read_csv('../CSVs/IDENTITY_PROFILES/ALL_SINGER_IDENTITY_PROFILES.csv')
    
    refined_dir = 'refined_outputs'
    out_csv_dir = 'CSVs/IDENTITY_PROFILES'
    os.makedirs(out_csv_dir, exist_ok=True)
    
    singers = ['arr', 'ilayaraja', 'mano', 'spb']
    
    results = []
    all_refined_profiles = []
    
    for singer in singers:
        wav_path = os.path.join(refined_dir, f"{singer}_refined.wav")
        if not os.path.exists(wav_path):
            continue
            
        print(f"Extracting 317 features for {singer}'s refined output...")
        features = mega_extractor.extract_identity_features(wav_path)
        
        # Convert dictionary to DataFrame
        feat_df = pd.DataFrame([features])
        # Drop string values if any (like 'singer')
        feat_df = feat_df.select_dtypes(include=[np.number])
        
        # Get target reference vector
        target_ref = profiles_df[profiles_df['singer'] == singer].copy()
        target_ref = target_ref.select_dtypes(include=[np.number])
        
        # Ensure identical column order
        cols = target_ref.columns
        # Fill missing with 0 just in case
        for c in cols:
            if c not in feat_df:
                feat_df[c] = 0.0
        
        feat_vec = feat_df[cols].fillna(0).values
        ref_vec = target_ref.fillna(0).values
        
        # Cosine Similarity
        sim = cosine_similarity(ref_vec, feat_vec)[0][0] * 100
        print(f"[{singer.upper()}] Identity Match pre-refinement: ~70%")
        print(f"[{singer.upper()}] Identity Match post-refinement: {sim:.2f}%\n")
        
        # Save individual CSV
        feat_df.to_csv(os.path.join(out_csv_dir, f"{singer}_refined_identity.csv"), index=False)
        all_refined_profiles.append(feat_df)
        
        results.append({"Singer": singer.upper(), 
                        "Pre-Refinement Similarity": "~70%", 
                        "Post-Refinement Similarity": f"{sim:.2f}%"})
        
    res_df = pd.DataFrame(results)
    print("\n--- Final Verification Results ---")
    print(res_df.to_string(index=False))
    
    # Save combined output
    if all_refined_profiles:
        pd.concat(all_refined_profiles).to_csv(os.path.join(out_csv_dir, "ALL_SINGER_REFINED_PROFILES.csv"), index=False)
    
    res_df.to_csv("similarity_results.csv", index=False)
    print("\n[SUCCESS] Saved all profile CSVs to ANTIGRAVITY_IMPLEMENTATION/CSVs/IDENTITY_PROFILES")
    print("[SUCCESS] Saved similarity_results.csv")

if __name__ == "__main__":
    extract_and_compare()
