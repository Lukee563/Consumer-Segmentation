import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from methods.data_clean import preprocess_data
from kmodes.kmodes import KModes

def cluster_data(path, n_clusters):
    df = preprocess_data(path)
    
    km = KModes(n_clusters = n_clusters, init='Huang', n_init=50, random_state=42)
    labels = km.fit_predict(df)
    data_clustered = df.copy()
    data_clustered["cluster"] = labels
    
    return data_clustered
    