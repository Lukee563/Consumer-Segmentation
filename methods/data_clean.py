import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


def preprocess_data(x):
    # Import data as df
    df = pd.read_csv(x)

    # Define preliminary k cluster count
    clusters = 5

    # Compute "percent missing" metric 
    missing_pct = round(df.isna().mean()*100,3).sort_values()

    # Only include variables with no more than 15% incomplete observations to maintain integrity 
    df = df[missing_pct[missing_pct <= 15].index]

    df_clean = df.dropna()
    df_clean = df_clean.drop(columns=["Unnamed: 0","submission_id"]) # Drop Variables that are unique for each observaiton

    # Store cleaned data in the data folder for later access if needed
    df_clean.to_csv("data/data_clean")

    return df_clean
    # Data is now fully preprocessed and ready for K-Modes clustering