import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from pathlib import Path
import os

def preprocess_data(x):
    df = pd.read_csv(x)
    
    missing_pct = round(df.isna().mean() * 100, 3).sort_values()
    df = df[missing_pct[missing_pct <= 15].index]
    
    df_clean = df.dropna()
    df_clean = df_clean.drop(columns=["Unnamed: 0", "submission_id"], errors='ignore')

    # Absolute path logic to prevent OSError
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "cleaned"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / "data_clean.csv"
    df_clean.to_csv(save_path, index=False)

    return df_clean