import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from methods.data_clean import preprocess_data

def evaluate_clusters(path, max_k=10):
    """
    Mathematically evaluates optimal k using Cost (Elbow) and Silhouette Score.
    """
    #Load Data
    df = preprocess_data(path)
    
    # PREPARE DATA FOR SILHOUETTE SCORE
    le = LabelEncoder()
    df_encoded = df.apply(lambda col: le.fit_transform(col))
    
    costs = []
    silhouettes = []
    K_range = range(2, max_k + 1)
    
    print(f"Evaluating clusters for k = 2 to {max_k}...")
    
    for k in K_range:
        # Train K-Modes (n_init=50 ensures stability!)
        km = KModes(n_clusters=k, init='Huang', n_init=50, verbose=0, random_state=42)
        labels = km.fit_predict(df)
        
        costs.append(km.cost_)
        
        # Calculate Silhouette Score
        score = silhouette_score(df_encoded, labels, metric='hamming')
        silhouettes.append(score)
        
        print(f"k={k}: Cost={km.cost_:.0f}, Silhouette={score:.4f}")

    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Cost (Dissimilarity)', color=color, fontsize=12)
    
    ln1 = ax1.plot(K_range, costs, color=color, marker='o', linewidth=3, 
                   markersize=8, label='Cost (Elbow)')
    
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', labelsize=11)
    
    # Clean borders
    ax1.spines['top'].set_visible(False)
    ax1.grid(axis='x', linestyle='--', alpha=0.5) # Vertical grid helps read 'k'

    # SECONDARY AXIS (SILHOUETTE - RED)
    ax2 = ax1.twinx()
    color = 'tab:purple'
    ax2.set_ylabel('Silhouette Score (Hamming)', color=color, fontsize=12)
    
    # Dashed line for contrast
    ln2 = ax2.plot(K_range, silhouettes, color=color, marker='s', linestyle='--', 
                   linewidth=2, markersize=8, label='Silhouette')
    
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.spines['top'].set_visible(False)

    # COMBINED LEGEND (The Professional Touch)
    # We grab handles from both axes and put them in one box
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), 
               ncol=2, frameon=False, fontsize=11)

    plt.title('Optimization: Elbow Method vs. Silhouette Score', 
              fontsize=14, fontweight='bold', pad=30)
    
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({'k': K_range, 'cost': costs, 'silhouette': silhouettes})

# Usage
# evaluate_clusters("data/coffee_survey.csv", max_k=10)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
from methods.data_clean import preprocess_data

def evaluate_stability(path, k=4, n_runs=20):
    """
    Evaluates stability by running the cluster algorithm numerous times 
    and plotting the Cost to check for variance.
    """
    df = preprocess_data(path)
    costs = []
    
    print(f"Testing stability for k={k} over {n_runs} runs...")
    
    for i in range(n_runs):
        # n_init=50 ensures we are finding the best local minimum each time
        km = KModes(n_clusters=k, init='Huang', n_init=50, verbose=0, random_state=i)
        km.fit(df)
        costs.append(km.cost_)
        
    mean_cost = np.mean(costs)
    std_cost = np.std(costs)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Main Data Line (Using 'tab:blue' to match your Cost/Elbow plot)
    ax.plot(costs, marker='o', linestyle='-', linewidth=2, markersize=6, 
            color='tab:purple', label='Run Cost')
    
    # Mean Line (To show the "Center" of stability)
    ax.axhline(y=mean_cost, color='tab:red', linestyle='--', alpha=0.7, 
               label=f'Mean Cost ({mean_cost:.0f})')
    
    # Clean up the "Box" (Remove top and right spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(53000, 58000)
    
    # Titles and Labels
    ax.set_title(f'Model Stability Check (k={k}, n_init=50)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Run Number (Random Seed)', fontsize=11)
    ax.set_ylabel('Cost (Dissimilarity)', fontsize=11, color='tab:blue')
    
    # legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Mean Cost: {mean_cost:.0f} | Std Dev: {std_cost:.0f}")

# Usage
# evaluate_stability("data/coffee_survey.csv", k=4)
