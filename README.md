# Consumer Segmentation (Quantitative Market Analysis) 
## K-Modes Clustering for Coffee Customer Survey Data
Author: Luke Catalano

Affiliation: University of California, Santa Cruz, M.S. Quantitative Economics & Finance

# Overview 
This project develops an unsupervised consumer segmentation pipeline using K-Modes clustering to analyze mixed categorical survey data for a coffee retailer. The primary objective is to identify distinct customer segments based on purchasing preferences, price sensitivity, and loyalty behaviors, and to translate these segments into actionable pricing and retention strategies aimed at improving customer lifetime value (CLV).

Traditional clustering methods such as K-Means are not well-suited for categorical data due to data type mismatches and the non-numerical natrue of the majority of responses. This project instead applies K-Modes, which replaces Euclidean distance with a mode-based dissimilarity measure (hamming distance), allowing for meaningful segmentation of survey-based customer attributes.

## Economic Motivation
Retailers often collect rich survey data through mobile app or web-based surveys, but struggle to convert qualitative responses into quantitative insights. Without segmentation, pricing and marketing strategies are typically applied uniformly, leading to:
	•	Over-discounting low-sensitivity customers
	•	Under-serving high-value loyalists
	•	Inefficient allocation of marketing spend

This project demonstrates how unsupervised learning can uncover latent structure in customer preferences and spending behavior, enabling segment-specific decision-making. In economic terms, such segmentation supports third-degree price discrimination, where consumers are grouped by quantifiable characteristics and firms tailor prices or offerings across segments to more efficiently capture surplus.

## Tools & Models
  • Python
  
  • K-Modes clustering
  
  • Pandas, NumPy
  
  • Seaborn 

# Methodologies

## Optimal Cluster Selection

To determine the proper cluster count, the within-cluster dissimilarity (cost function) and Silhouette Score were both analyzed to balance statistical fit and interpretability. While lower cluster counts produced marginally higher Silhouette Scores, they failed to capture meaningful behavioral heterogeneity (near-identical clusters). A four-cluster solution represented the point at which additional clusters yielded diminishing reductions in cost while preserving interpretable segment structure.
<img width="729" height="437" alt="Screenshot 2026-01-21 at 10 43 17 AM" src="https://github.com/user-attachments/assets/459a0c2c-9051-45e0-a4ea-b6361d92f995" />


## Cluster Stability

To assess robustness, the K-Modes algorithm was run 20 independent times using different random seeds and Huang initialization (Convergance Speed Optimization). Across all runs, the model converged to nearly identical solutions with minimal variation in total cost, indicating that the identified segments are structurally stable rather than by-products of random initialization.
<img width="630" height="364" alt="Screenshot 2026-01-21 at 10 42 52 AM" src="https://github.com/user-attachments/assets/796d4e15-1447-4333-8a56-6fa3b8b23cd7" />


## Results (Segment Profiles) 
The final model identified four distinct customer segments, each exhibiting consistent and interpretable behavioral patterns:

• Premium Customers (Cluster 0)

Price-insensitive, high-spending consumers with strong engagement across product categories. This segment represents the highest revenue potential and is well-suited for premium offerings, subscriptions, and high-margin upsells.

• Core Regulars (Cluster 1)

Moderate-to-high spenders with stable purchasing behavior. These customers form the revenue backbone of the business and benefit most from retention-focused strategies such as loyalty programs and consistency-based rewards.

• Budget-Conscious Consumers (Cluster 3)

Highly price-sensitive customers concentrated in lower spending brackets. Engagement can be increased through targeted promotions, value bundles, and limited-time discounts designed to raise basket size without eroding margins elsewhere.

• Minimalists (Cluster 2) 

Low-frequency, low-spend customers with limited engagement. This segment exhibits the highest churn risk, and marketing investment here yields the lowest marginal returns.

<img width="881" height="808" alt="Screenshot 2026-01-21 at 10 44 03 AM" src="https://github.com/user-attachments/assets/ebe26c5e-f876-4358-ba28-054b5ae51ccb" />

Note: spending bounds are averaged and normalized for analysis

# Key Takeaway

The results demonstrate that unsupervised learning can reliably uncover latent preference and spending structure in categorical survey data. The identified segments are reproducible, interpretable, and directly actionable, supporting differentiated pricing, retention, and marketing strategies aligned with heterogeneous consumer behavior.
