import pandas as pd

try:
    df = pd.read_csv(r"c:\Users\ANEETA PETER\Documents\gt_training\project_5_streamlit\mediscan_ckd_diagnostic_P5.csv")
    print("Columns:")
    for col in df.columns:
        print(col)
    print("\nShape:", df.shape)
except Exception as e:
    print(f"Error reading CSV: {e}")
