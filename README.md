# Coffee Consumer Segmentation
**Unsupervised Learning | K-Modes Clustering | Python**

---

## Overview

This project develops a consumer segmentation pipeline using **K-Modes clustering** to analyze mixed categorical survey data collected from a coffee retailer's customer base. The objective is to identify distinct customer segments based on purchasing preferences, price sensitivity, and loyalty behaviors — and translate those segments into actionable pricing and retention strategies aimed at improving customer lifetime value (CLV).

Traditional clustering methods like K-Means are not suited for categorical data due to Euclidean distance assumptions that break down on non-numerical survey responses. K-Modes replaces Euclidean distance with a **Hamming dissimilarity measure**, enabling meaningful segmentation of survey-based attributes without requiring encoding assumptions.

---

## Economic Motivation

Retailers commonly collect rich survey data through mobile or web-based platforms but struggle to convert qualitative responses into quantitative strategy. Without segmentation, pricing and marketing are applied uniformly, leading to:

- Over-discounting low-sensitivity customers
- Under-serving high-value loyalists
- Inefficient allocation of marketing spend

In economic terms, this project operationalizes **third-degree price discrimination** — grouping consumers by quantifiable behavioral characteristics so that pricing and offerings can be differentiated across segments to more efficiently capture surplus.

---

## Cluster Results

Four distinct consumer profiles were identified:

| Segment | Profile | Strategy |
|---|---|---|
| **Premium Customers** | Price-insensitive, high-spending, broad product engagement | Premium offerings, subscriptions, high-margin upsells |
| **Core Regulars** | Moderate-to-high spend, stable purchase behavior | Loyalty programs, consistency-based rewards |
| **Budget-Conscious** | Highly price-sensitive, lower spending brackets | Value bundles, targeted promotions, limited-time discounts |
| **Minimalists** | Low-frequency, low-spend, limited engagement | Lowest marketing ROI — highest churn risk |

Spending bounds are averaged and normalized across clusters for comparability.

---

## Methodology

**Algorithm:** K-Modes with Huang initialization (`n_init=50` for convergence stability), Hamming distance metric

**Optimal k selection:** Elbow method (within-cluster dissimilarity cost) and Silhouette Score evaluated jointly across k = 2–10. A four-cluster solution was selected as the point of diminishing cost reduction while preserving interpretable segment structure.

**Stability validation:** Algorithm run 20 independent times across different random seeds. Costs converged to near-identical solutions across all runs, confirming the segments are structurally stable rather than artifacts of initialization.

**Preprocessing:** Columns with >15% missing values dropped; remaining rows with any missing values removed; `submission_id` and unnamed index columns excluded. Cleaned data written to `data/cleaned/`.

**Spend parsing:** Raw spend range strings (e.g., `"$20-$40"`) parsed into `min_total_spend` / `max_total_spend` numeric bounds for cluster profiling and visualization.

---

## Repository Structure

```
├── analysis/
│   ├── src/
│   │   ├── data_clean.py          # Preprocessing pipeline (missing value filter, export)
│   │   ├── cluster_algo.py        # K-Modes clustering (Huang init, n_init=50)
│   │   └── cluster_evaluation.py  # Elbow + Silhouette optimization; stability analysis
│   └── scripts/
│       ├── 01_eda.ipynb           # Cluster profiling, spend parsing, EDA visualizations
│       └── 02_evaluation.ipynb    # Optimal k selection and stability validation
├── data/
│   ├── raw/
│   │   └── coffee_survey.csv      # Raw survey data
│   └── cleaned/
│       └── data_clean.csv         # Preprocessed output (auto-generated)
├── output/
│   ├── cluster_results.png        # Normalized spend by segment (bar chart)
│   ├── Customer_Age_Distribution.png
│   └── Favorite_Drink_Distribution.png
├── literature/
│   └── Project_Slides.pdf         # Presentation deck
└── README.md
```

**To run:**
```bash
pip install pandas numpy matplotlib seaborn kmodes scikit-learn
# From analysis/scripts/:
jupyter notebook 01_eda.ipynb       # Cluster profiling and EDA
jupyter notebook 02_evaluation.ipynb # k optimization and stability
```

---

## Key Takeaway

The results demonstrate that unsupervised learning can reliably uncover latent preference and spending structure in categorical survey data. The four identified segments are reproducible, interpretable, and directly actionable — supporting differentiated pricing, retention, and marketing strategies aligned with heterogeneous consumer behavior.

---

## Tech Stack

Python — `kmodes`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
